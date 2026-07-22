from __future__ import annotations

import ctypes
import re
import time
from pathlib import Path


def apply_triposplat_native_qkv_packed_attention_helpers_patch(
    flow_model,
    *,
    postprocess_library: str,
    sdpa_library: str,
    threads: int,
    include_regex: str = r"^blocks[.][0-9]+[.]attn$",
    strict: bool = True,
):
    import torch
    import triposplat_attention_patch

    postprocess_path = Path(postprocess_library).resolve()
    sdpa_path = Path(sdpa_library).resolve()
    if not postprocess_path.is_file() or not sdpa_path.is_file():
        raise FileNotFoundError(f"packed Attention libraries not found: {postprocess_path}, {sdpa_path}")

    postprocess_lib = ctypes.CDLL(postprocess_path.as_posix())
    postprocess = postprocess_lib.triposplat_qkv_rope_rmsnorm_pack_f32_avx512_v2
    postprocess.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 7
    postprocess.restype = ctypes.c_int
    postprocess_selected = postprocess_lib.triposplat_q_kv_selected_rope_rmsnorm_pack_f32_avx512_v2
    postprocess_selected.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int] * 8
    postprocess_selected.restype = ctypes.c_int
    sdpa_lib = ctypes.CDLL(sdpa_path.as_posix())
    sdpa = sdpa_lib.triposplat_sdpa_f32_avx512_exact_q8t512_packed_blhd
    sdpa.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 9
    sdpa.restype = ctypes.c_int

    include = re.compile(include_regex)
    selected_names = []
    selected_ids = set()
    for name, module in flow_model.named_modules():
        if module.__class__.__name__ != "RopeMultiHeadAttention" or include.search(name) is None:
            continue
        if module._type == "self" and module.use_rope and module.qk_rms_norm and module.head_dim == 64:
            selected_names.append(name)
            selected_ids.add(id(module))
    if strict and not selected_names:
        raise RuntimeError(f"packed Attention helper patch selected no modules with {include_regex!r}")

    workspace = {name: torch.empty(0, dtype=torch.float32) for name in ("q", "k", "v", "out")}
    runtime = {
        "calls": 0,
        "selected_row_calls": 0,
        "key_bias_calls": 0,
        "postprocess_seconds": 0.0,
        "sdpa_seconds": 0.0,
        "workspace_allocations": 0,
        "workspace_capacity_bytes": 0,
        "qkv_input_copies": 0,
        "frequency_copies": 0,
        "selected_index_copies": 0,
        "fallbacks": 0,
    }

    def acquire(name: str, elements: int):
        tensor = workspace[name]
        if tensor.numel() < elements:
            tensor = torch.empty(elements, dtype=torch.float32, device="cpu")
            workspace[name] = tensor
            runtime["workspace_allocations"] += 1
            runtime["workspace_capacity_bytes"] = sum(value.numel() * 4 for value in workspace.values())
        return tensor[:elements]

    def stage(timing_callback, name: str, function, shape):
        if timing_callback is None:
            return function()
        started = time.perf_counter()
        try:
            return function()
        finally:
            timing_callback(name, time.perf_counter() - started, shape=shape)

    def frequency_view(rope_emb, b: int, length: int, heads: int, dim: int):
        frequency = torch.view_as_real(rope_emb).reshape(*rope_emb.shape[:-1], dim)
        if not frequency.is_contiguous():
            frequency = frequency.contiguous()
            runtime["frequency_copies"] += 1
        frequency_batches = int(frequency.shape[0])
        if frequency_batches not in (1, b) or frequency.numel() != frequency_batches * length * heads * dim:
            raise RuntimeError(f"packed Attention frequency mismatch: {tuple(rope_emb.shape)}")
        return frequency, frequency_batches

    def run(attn, x, rope_emb, key_bias, timing_callback):
        b, length, channels = (int(value) for value in x.shape)
        heads, dim = int(attn.num_heads), int(attn.head_dim)
        if (
            id(attn) not in selected_ids
            or x.device.type != "cpu"
            or x.dtype != torch.float32
            or rope_emb is None
            or rope_emb.device.type != "cpu"
            or rope_emb.dtype != torch.complex64
        ):
            return None
        length_padded = (length + 15) & ~15
        qkv = stage(timing_callback, "attention.qkv_projection", lambda: attn.qkv(x), [b, length, 3 * channels])
        if not qkv.is_contiguous():
            qkv = qkv.contiguous()
            runtime["qkv_input_copies"] += 1
        frequency, frequency_batches = frequency_view(rope_emb, b, length, heads, dim)

        q = acquire("q", b * heads * length * dim).view(b, heads, length, dim)
        packed_elements = b * heads * dim * length_padded
        packed_k = acquire("k", packed_elements).view(b, heads, dim, length_padded)
        packed_v = acquire("v", packed_elements).view(b, heads, dim, length_padded)
        output = acquire("out", b * length * heads * dim).view(b, length, heads, dim)

        started = time.perf_counter()
        status = int(postprocess(
            qkv.data_ptr(), frequency.data_ptr(), attn.q_norm.gamma.data_ptr(), attn.k_norm.gamma.data_ptr(),
            q.data_ptr(), packed_k.data_ptr(), packed_v.data_ptr(),
            b, length, heads, dim, length_padded, frequency_batches, int(threads),
        ))
        elapsed = time.perf_counter() - started
        runtime["postprocess_seconds"] += elapsed
        if timing_callback is not None:
            timing_callback("attention.rope_qknorm_layout_pack", elapsed, shape=[b, heads, length, dim])
        if status != 0:
            raise RuntimeError(f"packed QKV postprocess returned {status}")

        bias_pointer = 0
        has_bias = 0
        bias_length = 0
        if key_bias is not None:
            bias = key_bias.to(dtype=torch.float32).reshape(-1).contiguous()
            if bias.numel() != length:
                raise RuntimeError(f"packed Attention key bias mismatch: {bias.numel()} != {length}")
            bias_pointer, has_bias, bias_length = bias.data_ptr(), 1, length
            runtime["key_bias_calls"] += 1

        started = time.perf_counter()
        status = int(sdpa(
            q.data_ptr(), packed_k.data_ptr(), packed_v.data_ptr(), bias_pointer, output.data_ptr(),
            b, heads, length, length, dim, length_padded, has_bias, bias_length, int(threads),
        ))
        elapsed = time.perf_counter() - started
        runtime["sdpa_seconds"] += elapsed
        if timing_callback is not None:
            timing_callback("attention.sdpa", elapsed, shape=[b, heads, length, dim])
        if status != 0:
            raise RuntimeError(f"packed SDPA returned {status}")
        runtime["calls"] += 1
        return stage(
            timing_callback,
            "attention.out_projection",
            lambda: attn.out(output.view(b, length, channels)),
            [b, length, channels],
        )

    def run_selected(attn, x, selected_idx, rope_emb, timing_callback):
        b, length, channels = (int(value) for value in x.shape)
        selected_length = int(selected_idx.numel())
        heads, dim = int(attn.num_heads), int(attn.head_dim)
        native_range = getattr(attn.qkv, "_native_avx512_forward_range", None)
        if (
            id(attn) not in selected_ids
            or native_range is None
            or x.device.type != "cpu"
            or x.dtype != torch.float32
            or rope_emb is None
            or rope_emb.device.type != "cpu"
            or rope_emb.dtype != torch.complex64
            or selected_idx.device.type != "cpu"
            or selected_idx.dtype != torch.int64
        ):
            return None

        if not selected_idx.is_contiguous():
            selected_idx = selected_idx.contiguous()
            runtime["selected_index_copies"] += 1
        x_selected = stage(
            timing_callback,
            "attention.selected_index",
            lambda: x.index_select(1, selected_idx),
            [b, selected_length, channels],
        )
        q = stage(
            timing_callback,
            "attention.q_projection",
            lambda: native_range(x_selected, 0, channels),
            [b, selected_length, channels],
        )
        kv = stage(
            timing_callback,
            "attention.kv_projection",
            lambda: native_range(x, channels, 2 * channels),
            [b, length, 2 * channels],
        )
        if not q.is_contiguous():
            q = q.contiguous()
            runtime["qkv_input_copies"] += 1
        if not kv.is_contiguous():
            kv = kv.contiguous()
            runtime["qkv_input_copies"] += 1
        frequency, frequency_batches = frequency_view(rope_emb, b, length, heads, dim)
        length_padded = (length + 15) & ~15
        q_packed = acquire("q", b * heads * selected_length * dim).view(b, heads, selected_length, dim)
        packed_elements = b * heads * dim * length_padded
        packed_k = acquire("k", packed_elements).view(b, heads, dim, length_padded)
        packed_v = acquire("v", packed_elements).view(b, heads, dim, length_padded)
        output = acquire("out", b * selected_length * heads * dim).view(b, selected_length, heads, dim)

        started = time.perf_counter()
        status = int(postprocess_selected(
            q.data_ptr(), kv.data_ptr(), frequency.data_ptr(), selected_idx.data_ptr(),
            attn.q_norm.gamma.data_ptr(), attn.k_norm.gamma.data_ptr(),
            q_packed.data_ptr(), packed_k.data_ptr(), packed_v.data_ptr(),
            b, selected_length, length, heads, dim, length_padded, frequency_batches, int(threads),
        ))
        elapsed = time.perf_counter() - started
        runtime["postprocess_seconds"] += elapsed
        if timing_callback is not None:
            timing_callback(
                "attention.rope_qknorm_layout_pack",
                elapsed,
                shape={"q": [b, heads, selected_length, dim], "k": [b, heads, length, dim]},
            )
        if status != 0:
            raise RuntimeError(f"selected packed Q/KV postprocess returned {status}")

        started = time.perf_counter()
        status = int(sdpa(
            q_packed.data_ptr(), packed_k.data_ptr(), packed_v.data_ptr(), 0, output.data_ptr(),
            b, heads, selected_length, length, dim, length_padded, 0, 0, int(threads),
        ))
        elapsed = time.perf_counter() - started
        runtime["sdpa_seconds"] += elapsed
        if timing_callback is not None:
            timing_callback(
                "attention.sdpa",
                elapsed,
                shape={"q": [b, heads, selected_length, dim], "k": [b, heads, length, dim]},
            )
        if status != 0:
            raise RuntimeError(f"selected packed SDPA returned {status}")
        runtime["calls"] += 1
        runtime["selected_row_calls"] += 1
        return stage(
            timing_callback,
            "attention.out_projection",
            lambda: attn.out(output.view(b, selected_length, channels)),
            [b, selected_length, channels],
        )

    original_standard = triposplat_attention_patch._rope_self_attention_standard
    original_key_bias = triposplat_attention_patch._rope_self_attention_with_key_bias
    original_selected = triposplat_attention_patch._rope_self_attention_selected_rows

    def packed_standard(attn, x, rope_emb, *, backend="default", compute_dtype="model", query_chunk_size=128, contiguous_qkv=True, timing_callback=None):
        if backend == "native_avx512_exact" and compute_dtype == "model":
            result = run(attn, x, rope_emb, None, timing_callback)
            if result is not None:
                return result
        runtime["fallbacks"] += 1
        if strict and id(attn) in selected_ids:
            raise RuntimeError("selected standard Attention fell back from packed path")
        return original_standard(
            attn, x, rope_emb, backend=backend, compute_dtype=compute_dtype,
            query_chunk_size=query_chunk_size, contiguous_qkv=contiguous_qkv,
            timing_callback=timing_callback,
        )

    def packed_key_bias(attn, x, rope_emb, key_bias, *, backend="default", timing_callback=None):
        if backend == "native_avx512_exact":
            result = run(attn, x, rope_emb, key_bias, timing_callback)
            if result is not None:
                return result
        runtime["fallbacks"] += 1
        if strict and id(attn) in selected_ids:
            raise RuntimeError("selected key-bias Attention fell back from packed path")
        return original_key_bias(attn, x, rope_emb, key_bias, backend=backend, timing_callback=timing_callback)

    def packed_selected(
        attn,
        x,
        selected_idx,
        rope_emb,
        *,
        backend,
        compute_dtype,
        query_chunk_size,
        round_qkv_to_fp16=False,
        round_v_to_fp16=False,
        round_attn_core_to_fp16=False,
        half_sequence=False,
        key_bias=None,
        timing_callback=None,
    ):
        exact = (
            backend == "native_avx512_exact"
            and compute_dtype == "model"
            and not round_qkv_to_fp16
            and not round_v_to_fp16
            and not round_attn_core_to_fp16
            and not half_sequence
            and key_bias is None
        )
        if exact:
            result = run_selected(attn, x, selected_idx, rope_emb, timing_callback)
            if result is not None:
                return result
            runtime["fallbacks"] += 1
            if strict and id(attn) in selected_ids:
                raise RuntimeError("selected-row Attention fell back from packed path")
        return original_selected(
            attn,
            x,
            selected_idx,
            rope_emb,
            backend=backend,
            compute_dtype=compute_dtype,
            query_chunk_size=query_chunk_size,
            round_qkv_to_fp16=round_qkv_to_fp16,
            round_v_to_fp16=round_v_to_fp16,
            round_attn_core_to_fp16=round_attn_core_to_fp16,
            half_sequence=half_sequence,
            key_bias=key_bias,
            timing_callback=timing_callback,
        )

    triposplat_attention_patch._rope_self_attention_standard = packed_standard
    triposplat_attention_patch._rope_self_attention_with_key_bias = packed_key_bias
    triposplat_attention_patch._rope_self_attention_selected_rows = packed_selected
    return {
        "enabled": bool(selected_names),
        "kind": "native_attention_helpers_qkv_packed_exact_v3",
        "postprocess_library": postprocess_path.as_posix(),
        "sdpa_library": sdpa_path.as_posix(),
        "threads": int(threads),
        "include_regex": include_regex,
        "selected_count": len(selected_names),
        "selected": selected_names,
        "runtime": runtime,
        "coverage": "positive standard, negative key-bias, and final positive selected-query helpers",
        "semantics": "Exact RoPE/RMSNorm and q8t512 arithmetic; Q is computed only for consumed rows in the final selected-query helper.",
    }
