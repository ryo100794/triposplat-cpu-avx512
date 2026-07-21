#!/usr/bin/env python3
"""Runtime attention patching utilities for TripoSplat experiments.

The official TripoSplat model calls a module-level
``model.scaled_dot_product_attention`` helper. These utilities replace that
helper at runtime so experiments can change SDPA backend and compute dtype
without editing the upstream repository.
"""

from __future__ import annotations

from contextlib import nullcontext
import json
import math
import os
import re
import types
from typing import Any


def _streaming_backends() -> dict[str, str]:
    return {
        "streaming": "two_pass",
        "streaming_online": "online",
        "streaming_m4": "online_m4",
        "streaming_m4_d64": "online_m4_d64",
        "streaming_m4_d64_k64": "online_m4_d64_k64",
        "streaming_m8": "online_m8",
    }


def _valid_backends() -> set[str]:
    return {"default", "math", "flash", "chunked", "aten_flash_direct", "aten_flash_direct_scale", "aten_flash_direct_auto", "f_explicit_scale", "native_avx512_exact", *_streaming_backends().keys()}


def _is_streaming_backend(backend: str) -> bool:
    return backend in _streaming_backends()


def _is_native_sdpa_backend(backend: str) -> bool:
    return backend == "native_avx512_exact"


def _native_add_tensor_inplace(owner, base, delta):
    native = getattr(owner, "_native_avx512_add_inplace", None)
    if native is not None:
        return native(base, delta)
    return base + delta


def _native_shift_table_add(owner, modulation, shift_table):
    native = getattr(owner, "_native_avx512_add_inplace", None)
    if native is not None:
        out = modulation.contiguous().clone()
        return native(out, shift_table.contiguous())
    return modulation + shift_table


def _native_functional_layer_norm(owner, value, eps: float = 1.0e-5):
    native = getattr(owner, "_native_avx512_functional_layernorm", None)
    if native is not None:
        return native(value, eps=eps)
    import torch.nn.functional as F
    return F.layer_norm(value.float(), value.shape[-1:], eps=eps).type_as(value)


def _native_modulate_tensor(owner, value, scale, shift):
    native = getattr(owner, "_native_avx512_modulate_inplace", None)
    if native is not None:
        return native(value, scale, shift)
    return value * (1 + scale) + shift


def _native_final_modulation_params(owner, shift_table, t_emb):
    native = getattr(owner, "_native_avx512_add_inplace", None)
    if native is not None:
        expanded = t_emb.unsqueeze(1).expand_as(shift_table).contiguous()
        return native(shift_table.clone(), expanded).chunk(2, dim=1)
    return (shift_table + t_emb.unsqueeze(1)).chunk(2, dim=1)


def _streaming_mode(backend: str) -> str:
    try:
        return _streaming_backends()[backend]
    except KeyError as exc:
        raise ValueError(f"unsupported streaming attention backend: {backend}") from exc


def _chunk_size(default: int = 128) -> int:
    raw = os.environ.get("TRIPOSPLAT_CHUNKED_SDPA_Q", "")
    if raw.strip():
        return max(1, int(raw))
    return int(default)


def streaming_exact_scaled_dot_product_attention_bhld(q, k, v, *, backend: str = "streaming_m4_d64", threads: int | None = None):
    """Exact full attention through the float32 ctypes streaming SDPA kernel.

    q/k/v are shaped [B, H, L, D]. The kernel computes dense full attention
    without materializing the LxL score/probability matrix. It is CPU-only and
    not an approximate, sparse, or low-rank method. Inputs are upcast to
    float32 for the kernel and the result is cast back to the original dtype.
    """
    import torch

    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q/k/v must be [B,H,L,D]")
    if q.shape != k.shape or k.shape != v.shape:
        raise ValueError(f"streaming SDPA requires matching q/k/v shapes: q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")
    if q.device.type != "cpu" or k.device.type != "cpu" or v.device.type != "cpu":
        raise ValueError("streaming SDPA backend is CPU-only")
    if int(q.shape[-1]) > 256:
        raise ValueError("streaming SDPA prototype supports head_dim <= 256")

    mode = _streaming_mode(backend)
    orig_dtype = q.dtype
    q_f = q.to(dtype=torch.float32).contiguous()
    k_f = k.to(dtype=torch.float32).contiguous()
    v_f = v.to(dtype=torch.float32).contiguous()

    from streaming_sdpa_ctypes import (
        streaming_sdpa_f32,
        streaming_sdpa_f32_online,
        streaming_sdpa_f32_online_m4,
        streaming_sdpa_f32_online_m4_d64,
        streaming_sdpa_f32_online_m4_d64_k64,
        streaming_sdpa_f32_online_m8,
    )

    if mode == "two_pass":
        out = streaming_sdpa_f32(q_f, k_f, v_f, threads=threads)
    elif mode == "online":
        out = streaming_sdpa_f32_online(q_f, k_f, v_f, threads=threads)
    elif mode == "online_m4":
        out = streaming_sdpa_f32_online_m4(q_f, k_f, v_f, threads=threads)
    elif mode == "online_m4_d64":
        out = streaming_sdpa_f32_online_m4_d64(q_f, k_f, v_f, threads=threads)
    elif mode == "online_m4_d64_k64":
        out = streaming_sdpa_f32_online_m4_d64_k64(q_f, k_f, v_f, threads=threads)
    elif mode == "online_m8":
        out = streaming_sdpa_f32_online_m8(q_f, k_f, v_f, threads=threads)
    else:  # pragma: no cover - guarded by _streaming_mode
        raise ValueError(f"unsupported streaming mode: {mode}")
    return out.to(dtype=orig_dtype)


def native_exact_scaled_dot_product_attention_bhld(q, k, v, *, key_bias=None, threads: int | None = None):
    """Native AVX-512 exact dense attention for TripoSplat D=64 CPU shapes."""
    from native_sdpa_ctypes import native_sdpa_f32

    return native_sdpa_f32(q, k, v, key_bias=key_bias, threads=threads)


def chunked_exact_scaled_dot_product_attention_bhld(q, k, v, *, query_chunk_size: int = 128, compute_dtype: str = "model"):
    """Exact full attention with query-axis chunking.

    Args:
        q/k/v: tensors shaped [B, H, L, D].
        query_chunk_size: number of query tokens per chunk.
        compute_dtype: ``model`` keeps the model dtype for matmul inputs;
            ``float32`` upcasts q/k/v before matmul. Softmax is always computed
            in float32, then cast back to the matmul dtype for the second matmul.

    This preserves dense full-attention semantics. It is not sparse, linear, or
    low-rank attention. The first Python/Torch version is primarily a correctness
    and memory-control reference for a future C++/SIMD kernel.
    """
    import torch

    if compute_dtype not in {"model", "float32"}:
        raise ValueError(f"unsupported attention compute dtype: {compute_dtype}")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q/k/v must be [B,H,L,D]")
    if q.shape[:2] != k.shape[:2] or k.shape != v.shape:
        raise ValueError(f"unsupported q/k/v shapes: q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")
    orig_dtype = q.dtype
    work_dtype = torch.float32 if compute_dtype == "float32" else q.dtype
    q_work = q.to(dtype=work_dtype).contiguous()
    k_work = k.to(dtype=work_dtype).contiguous()
    v_work = v.to(dtype=work_dtype).contiguous()
    kt = k_work.transpose(-2, -1).contiguous()
    scale = 1.0 / math.sqrt(float(q_work.shape[-1]))
    chunks = []
    q_chunk = max(1, int(query_chunk_size))
    for start in range(0, q_work.shape[-2], q_chunk):
        end = min(start + q_chunk, q_work.shape[-2])
        scores = torch.matmul(q_work[..., start:end, :], kt) * scale
        probs = torch.softmax(scores.float(), dim=-1).to(dtype=work_dtype)
        chunks.append(torch.matmul(probs, v_work))
    return torch.cat(chunks, dim=-2).to(dtype=orig_dtype)


def apply_triposplat_attention_patch(
    *,
    backend: str = "default",
    compute_dtype: str = "model",
    query_chunk_size: int = 128,
    contiguous_qkv: bool = False,
) -> dict[str, Any]:
    """Patch TripoSplat's SDPA wrapper and return manifest metadata.

    Args:
        backend: one of default, math, flash, chunked, or streaming*.
        compute_dtype: one of model, float32.
            ``model`` preserves the original input dtype. ``float32`` upcasts
            q/k/v for attention math, then casts output back to the original
            dtype before the following Linear layer.
        query_chunk_size: only used by backend=chunked.
        contiguous_qkv: make permuted [B,H,L,D] q/k/v contiguous before
            calling torch SDPA. This is exact and targets CPU flash layout cost.
    """
    if backend not in _valid_backends():
        raise ValueError(f"unsupported attention backend: {backend}")
    if compute_dtype not in {"model", "float32"}:
        raise ValueError(f"unsupported attention compute dtype: {compute_dtype}")
    if backend == "default" and compute_dtype == "model" and not contiguous_qkv:
        return {"enabled": False, "backend": backend, "compute_dtype": compute_dtype, "contiguous_qkv": False}

    import torch.nn.functional as F

    try:
        import model as triposplat_model
    except Exception as exc:  # pragma: no cover - depends on runtime sys.path
        raise RuntimeError("could not import TripoSplat model module for attention patch") from exc

    if not hasattr(triposplat_model, "_original_scaled_dot_product_attention"):
        triposplat_model._original_scaled_dot_product_attention = triposplat_model.scaled_dot_product_attention

    def patched_scaled_dot_product_attention(qkv=None, q=None, k=None, v=None, kv=None):
        if qkv is not None:
            q, k, v = qkv.unbind(dim=2)
        elif kv is not None:
            k, v = kv.unbind(dim=2)
        if q is None or k is None or v is None:
            raise ValueError("q/k/v must be provided")
        return _attention_core(
            q,
            k,
            v,
            backend=backend,
            compute_dtype=compute_dtype,
            query_chunk_size=int(query_chunk_size),
            contiguous_qkv=bool(contiguous_qkv),
        )

    triposplat_model.scaled_dot_product_attention = patched_scaled_dot_product_attention
    return {
        "enabled": True,
        "backend": backend,
        "compute_dtype": compute_dtype,
        "contiguous_qkv": bool(contiguous_qkv),
        "query_chunk_size": int(query_chunk_size) if backend == "chunked" else None,
        "streaming_mode": _streaming_mode(backend) if _is_streaming_backend(backend) else None,
        "streaming_threads_env": os.environ.get("STREAMING_SDPA_THREADS"),
        "effective_compute_dtype": "float32" if _is_streaming_backend(backend) else compute_dtype,
        "cross_attention_fallback": "chunked_float32_exact" if _is_streaming_backend(backend) else None,
        "note": "Runtime patch of model.scaled_dot_product_attention; dense full attention semantics are preserved. contiguous_qkv only changes memory layout before torch SDPA.",
    }



def _backend_context(backend: str):
    if backend in {"default", "aten_flash_direct", "aten_flash_direct_scale", "aten_flash_direct_auto", "f_explicit_scale", "native_avx512_exact"}:
        return nullcontext()
    if backend == "chunked" or _is_streaming_backend(backend) or _is_native_sdpa_backend(backend):
        raise ValueError(f"{backend} backend does not use torch sdpa_kernel context")
    from torch.nn.attention import SDPBackend, sdpa_kernel

    backend_enum = SDPBackend.MATH if backend == "math" else SDPBackend.FLASH_ATTENTION
    return sdpa_kernel(backend_enum)


def _attention_core(q, k, v, *, backend: str, compute_dtype: str, query_chunk_size: int = 128, contiguous_qkv: bool = False):
    import torch.nn.functional as F

    orig_dtype = q.dtype
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    if contiguous_qkv:
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
    if backend == "chunked":
        out = chunked_exact_scaled_dot_product_attention_bhld(
            q, k, v, query_chunk_size=int(query_chunk_size), compute_dtype=compute_dtype
        )
    elif _is_native_sdpa_backend(backend):
        out = native_exact_scaled_dot_product_attention_bhld(q, k, v)
    elif _is_streaming_backend(backend):
        if q.shape[-2] != k.shape[-2]:
            out = chunked_exact_scaled_dot_product_attention_bhld(
                q, k, v, query_chunk_size=int(query_chunk_size), compute_dtype="float32"
            )
        else:
            out = streaming_exact_scaled_dot_product_attention_bhld(q, k, v, backend=backend)
    else:
        if compute_dtype == "float32":
            q = q.float()
            k = k.float()
            v = v.float()
        if backend in {"aten_flash_direct", "aten_flash_direct_scale", "aten_flash_direct_auto"}:
            import torch
            direct_scale = None
            if backend == "aten_flash_direct_scale" or (backend == "aten_flash_direct_auto" and q.shape[-2] == k.shape[-2]):
                direct_scale = 1.0 / math.sqrt(float(q.shape[-1]))
            out = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(
                q, k, v, 0.0, False, attn_mask=None, scale=direct_scale
            )[0]
        elif backend == "f_explicit_scale":
            out = F.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(float(q.shape[-1])))
        else:
            with _backend_context(backend):
                out = F.scaled_dot_product_attention(q, k, v)
    return out.permute(0, 2, 1, 3).to(dtype=orig_dtype)



def _modulate_addcmul(h, scale, shift):
    import torch

    return torch.addcmul(shift.unsqueeze(1), h, (1 + scale).unsqueeze(1))


def _residual_gate_addcmul(x, h, gate):
    import torch

    return torch.addcmul(x, h, gate.unsqueeze(1))


def _modulate_inplace_preserve_order(h, scale, shift):
    h.mul_(1 + scale.unsqueeze(1))
    h.add_(shift.unsqueeze(1))
    return h


def _residual_gate_inplace_preserve_order(x, h, gate):
    h.mul_(gate.unsqueeze(1))
    x.add_(h)
    return x


def _key_bias_lse_adjust_meta(key_bias):
    if key_bias is None or not bool(getattr(key_bias, "_triposplat_logbias_lse_adjust", False)):
        return None
    key_index = getattr(key_bias, "_triposplat_logbias_index", None)
    multiplicity = getattr(key_bias, "_triposplat_logbias_multiplicity", None)
    if key_index is None or multiplicity is None:
        return None
    return int(key_index), int(multiplicity)


def _direct_flash_attention_with_key_bias(q, k, v, key_bias, *, direct_scale):
    import math
    import torch

    meta = _key_bias_lse_adjust_meta(key_bias)
    if meta is None:
        return torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(
            q, k, v, 0.0, False, attn_mask=key_bias, scale=direct_scale
        )[0]
    key_index, multiplicity = meta
    if multiplicity <= 1 or key_index < 0 or key_index >= int(k.shape[-2]):
        return torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(
            q, k, v, 0.0, False, attn_mask=key_bias, scale=direct_scale
        )[0]
    effective_scale = 1.0 / math.sqrt(float(q.shape[-1])) if direct_scale is None else float(direct_scale)
    out, logsumexp = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(
        q, k, v, 0.0, False, attn_mask=None, scale=direct_scale
    )
    key = k[..., key_index : key_index + 1, :]
    value = v[..., key_index : key_index + 1, :]
    prob = torch.exp((q * key).sum(dim=-1) * effective_scale - logsumexp).unsqueeze(-1)
    alpha = float(multiplicity - 1)
    denom = 1.0 + alpha * prob
    out.addcmul_(prob, value, value=alpha)
    out.div_(denom)
    return out


def _rope_self_attention_selected_rows(
    attn,
    x,
    selected_idx,
    rope_emb,
    *,
    backend: str,
    compute_dtype: str,
    query_chunk_size: int,
    round_qkv_to_fp16: bool = False,
    round_v_to_fp16: bool = False,
    round_attn_core_to_fp16: bool = False,
    half_sequence: bool = False,
    key_bias=None,
    timing_callback=None,
):
    import time
    import torch
    import torch.nn.functional as F

    try:
        import model as triposplat_model
    except Exception as exc:  # pragma: no cover - depends on runtime sys.path
        raise RuntimeError("could not import TripoSplat model module for selective final block patch") from exc

    def _stage(name: str, fn, *, shape=None):
        if timing_callback is None:
            return fn()
        started = time.perf_counter()
        try:
            return fn()
        finally:
            timing_callback(name, time.perf_counter() - started, shape=shape)

    if attn._type != "self":
        raise ValueError("selective final block patch only supports self-attention")
    B, L, C = x.shape
    H = attn.num_heads
    D = attn.head_dim
    native_range = getattr(attn.qkv, "_native_avx512_forward_range", None)
    if native_range is None:
        weight = attn.qkv.weight
        bias = attn.qkv.bias
        w_q = weight[:C]
        w_kv = weight[C:]
        b_q = None if bias is None else bias[:C]
        b_kv = None if bias is None else bias[C:]

    x_sel = _stage(
        "attention.selected_index",
        lambda: x.index_select(1, selected_idx),
        shape=[int(B), int(selected_idx.numel()), int(C)],
    )
    if native_range is not None:
        q = _stage(
            "attention.q_projection",
            lambda: native_range(x_sel, 0, C),
            shape=[int(B), int(selected_idx.numel()), int(C)],
        )
        kv = _stage(
            "attention.kv_projection",
            lambda: native_range(x, C, 2 * C),
            shape=[int(B), int(L), int(2 * C)],
        )
    else:
        linear_dtype = weight.dtype if torch.is_tensor(weight) and weight.is_floating_point() else x.dtype
        q_input = x_sel.to(dtype=linear_dtype) if x_sel.dtype != linear_dtype else x_sel
        kv_input = x.to(dtype=linear_dtype) if x.dtype != linear_dtype else x
        q = _stage(
            "attention.q_projection",
            lambda: F.linear(q_input, w_q, b_q),
            shape=[int(B), int(selected_idx.numel()), int(C)],
        )
        kv = _stage(
            "attention.kv_projection",
            lambda: F.linear(kv_input, w_kv, b_kv),
            shape=[int(B), int(L), int(2 * C)],
        )
    if q.dtype != x.dtype:
        q = q.to(dtype=x.dtype)
    if kv.dtype != x.dtype:
        kv = kv.to(dtype=x.dtype)
    if half_sequence or round_qkv_to_fp16:
        q = q.to(dtype=torch.float16).to(dtype=q.dtype)
        kv = kv.to(dtype=torch.float16).to(dtype=kv.dtype)
    q = q.reshape(B, selected_idx.numel(), H, D)
    kv = kv.reshape(B, L, 2, H, D)
    k, v = kv.unbind(2)
    if half_sequence or round_v_to_fp16:
        v = v.to(dtype=torch.float16).to(dtype=v.dtype)
    if attn.use_rope:
        started = time.perf_counter() if timing_callback is not None else None
        q = triposplat_model.apply_rotary_emb(q, rope_emb.index_select(1, selected_idx))
        k = triposplat_model.apply_rotary_emb(k, rope_emb)
        if half_sequence:
            q = q.to(dtype=torch.float16).to(dtype=q.dtype)
            k = k.to(dtype=torch.float16).to(dtype=k.dtype)
        if started is not None:
            timing_callback(
                "attention.rope",
                time.perf_counter() - started,
                shape={"q": [int(B), int(selected_idx.numel()), int(H), int(D)], "k": [int(B), int(L), int(H), int(D)]},
            )
    if attn.qk_rms_norm:
        started = time.perf_counter() if timing_callback is not None else None
        q = attn.q_norm(q)
        k = attn.k_norm(k)
        if half_sequence:
            q = q.to(dtype=torch.float16).to(dtype=q.dtype)
            k = k.to(dtype=torch.float16).to(dtype=k.dtype)
        if started is not None:
            timing_callback(
                "attention.qk_norm",
                time.perf_counter() - started,
                shape={"q": [int(B), int(selected_idx.numel()), int(H), int(D)], "k": [int(B), int(L), int(H), int(D)]},
            )
    if key_bias is None:
        h = _stage(
            "attention.sdpa",
            lambda: _attention_core(
                q,
                k,
                v,
                backend=backend,
                compute_dtype=compute_dtype,
                query_chunk_size=int(query_chunk_size),
                contiguous_qkv=True,
            ),
            shape={"q": [int(B), int(selected_idx.numel()), int(H), int(D)], "k": [int(B), int(L), int(H), int(D)]},
        )
    else:
        if backend == "chunked" or _is_streaming_backend(backend):
            raise ValueError("selected-row key-bias attention supports torch SDPA backends only")
        orig_dtype = q.dtype
        started = time.perf_counter() if timing_callback is not None else None
        q_bhld = q.permute(0, 2, 1, 3).contiguous()
        k_bhld = k.permute(0, 2, 1, 3).contiguous()
        v_bhld = v.permute(0, 2, 1, 3).contiguous()
        if started is not None:
            timing_callback(
                "attention.layout",
                time.perf_counter() - started,
                shape={"q": [int(B), int(H), int(selected_idx.numel()), int(D)], "k": [int(B), int(H), int(L), int(D)]},
            )
        if compute_dtype == "float32":
            q_bhld = q_bhld.float()
            k_bhld = k_bhld.float()
            v_bhld = v_bhld.float()
        if _is_native_sdpa_backend(backend):
            h = _stage(
                "attention.sdpa",
                lambda: native_exact_scaled_dot_product_attention_bhld(q_bhld, k_bhld, v_bhld, key_bias=key_bias),
                shape={"q": [int(B), int(H), int(selected_idx.numel()), int(D)], "k": [int(B), int(H), int(L), int(D)]},
            )
        elif backend in {"aten_flash_direct", "aten_flash_direct_scale", "aten_flash_direct_auto"}:
            direct_scale = 1.0 / math.sqrt(float(q_bhld.shape[-1])) if backend == "aten_flash_direct_scale" else None
            h = _stage(
                "attention.sdpa",
                lambda: _direct_flash_attention_with_key_bias(q_bhld, k_bhld, v_bhld, key_bias, direct_scale=direct_scale),
                shape={"q": [int(B), int(H), int(selected_idx.numel()), int(D)], "k": [int(B), int(H), int(L), int(D)]},
            )
        elif backend == "f_explicit_scale":
            h = _stage(
                "attention.sdpa",
                lambda: F.scaled_dot_product_attention(
                    q_bhld,
                    k_bhld,
                    v_bhld,
                    attn_mask=key_bias,
                    scale=1.0 / math.sqrt(float(q_bhld.shape[-1])),
                ),
                shape={"q": [int(B), int(H), int(selected_idx.numel()), int(D)], "k": [int(B), int(H), int(L), int(D)]},
            )
        else:
            with _backend_context(backend):
                h = _stage(
                    "attention.sdpa",
                    lambda: F.scaled_dot_product_attention(q_bhld, k_bhld, v_bhld, attn_mask=key_bias),
                    shape={"q": [int(B), int(H), int(selected_idx.numel()), int(D)], "k": [int(B), int(H), int(L), int(D)]},
                )
        h = h.permute(0, 2, 1, 3).to(dtype=orig_dtype)
    if half_sequence or round_attn_core_to_fp16:
        h = h.to(dtype=torch.float16).to(dtype=h.dtype)
    out = _stage(
        "attention.out_projection",
        lambda: attn.out(h.reshape(B, selected_idx.numel(), C)),
        shape=[int(B), int(selected_idx.numel()), int(C)],
    )
    if half_sequence:
        out = out.to(dtype=torch.float16).to(dtype=out.dtype)
    return out


def apply_triposplat_selective_final_block_patch(
    flow_model,
    *,
    enabled: bool = False,
    backend: str = "default",
    compute_dtype: str = "model",
    query_chunk_size: int = 128,
    round_qkv_to_fp16: bool = False,
    round_v_to_fp16: bool = False,
    round_attn_core_to_fp16: bool = False,
    half_sequence: bool = False,
    elementwise_compute_dtype: str = "roundtrip",
    inplace_output: bool = False,
    latent_length: int | None = None,
    camera_length: int | None = None,
) -> dict[str, Any]:
    """Patch the final main block to compute only consumed output rows.

    LatentSeqMMFlowModel concatenates [latent, condition, camera] for the main
    blocks, but after the final block it only consumes latent and camera rows.
    This patch preserves the final block return shape and leaves unused
    condition rows as their input values.
    """
    if not enabled:
        return {"enabled": False, "backend": backend, "compute_dtype": compute_dtype}
    if backend not in _valid_backends():
        raise ValueError(f"unsupported selective final block backend: {backend}")
    if compute_dtype not in {"model", "float32"}:
        raise ValueError(f"unsupported selective final block compute dtype: {compute_dtype}")
    if elementwise_compute_dtype not in {"roundtrip", "float16"}:
        raise ValueError(f"unsupported selective final block elementwise compute dtype: {elementwise_compute_dtype}")
    if not hasattr(flow_model, "blocks") or len(flow_model.blocks) == 0:
        raise ValueError("flow_model has no main blocks to patch")

    import torch

    block = flow_model.blocks[-1]
    q_token_length = int(latent_length if latent_length is not None else flow_model.q_token_length)
    cam_len = int(camera_length if camera_length is not None else (1 if getattr(flow_model, "cam_channels", None) is not None else 0))

    def patched_forward(self, x, mod=None, rotary_emb=None):
        L = int(x.shape[1])
        if q_token_length + cam_len > L:
            raise ValueError(f"invalid selective rows for L={L}: latent={q_token_length} camera={cam_len}")
        parts = [torch.arange(0, q_token_length, device=x.device, dtype=torch.long)]
        if cam_len:
            parts.append(torch.arange(L - cam_len, L, device=x.device, dtype=torch.long))
        selected_idx = torch.cat(parts)

        x_work = _fp16_roundtrip_tensor(x) if half_sequence else x
        if mod is not None and half_sequence:
            mod = _fp16_roundtrip_tensor(mod)

        if self.modulation:
            mod_work = self.adaLN_modulation(mod) if not self.share_mod else mod
            if half_sequence:
                mod_work = _fp16_roundtrip_tensor(mod_work)
            if hasattr(self, "shift_table") and self.shift_table is not None:
                mod_work = _native_shift_table_add(self, mod_work, self.shift_table.type(mod_work.dtype))
                if half_sequence:
                    mod_work = _fp16_roundtrip_tensor(mod_work)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod_work.chunk(6, dim=1)
            h1 = self.norm1(x_work)
            if half_sequence:
                h1 = _fp16_roundtrip_tensor(h1)
            if half_sequence and elementwise_compute_dtype == "float16":
                h1 = _modulated_half_compute(h1, scale_msa, shift_msa)
            else:
                h1 = h1 * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
                if half_sequence:
                    h1 = _fp16_roundtrip_tensor(h1)
            attn_sel = _rope_self_attention_selected_rows(
                self.attn,
                h1,
                selected_idx,
                rotary_emb,
                backend=backend,
                compute_dtype=compute_dtype,
                query_chunk_size=int(query_chunk_size),
                round_qkv_to_fp16=bool(round_qkv_to_fp16),
                round_v_to_fp16=bool(round_v_to_fp16),
                round_attn_core_to_fp16=bool(round_attn_core_to_fp16),
                half_sequence=bool(half_sequence),
            )
            x_base_sel = x_work.index_select(1, selected_idx)
            if half_sequence and elementwise_compute_dtype == "float16":
                x_sel = _residual_gate_half_compute(x_base_sel, attn_sel, gate_msa)
            else:
                x_sel = x_base_sel + attn_sel * gate_msa.unsqueeze(1)
                if half_sequence:
                    x_sel = _fp16_roundtrip_tensor(x_sel)
            h2 = self.norm2(x_sel)
            if half_sequence:
                h2 = _fp16_roundtrip_tensor(h2)
            if half_sequence and elementwise_compute_dtype == "float16":
                h2 = _modulated_half_compute(h2, scale_mlp, shift_mlp)
            else:
                h2 = h2 * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
                if half_sequence:
                    h2 = _fp16_roundtrip_tensor(h2)
            if half_sequence:
                mlp_out = _feed_forward_half_sequence(self.mlp, h2)
            else:
                mlp_out = self.mlp(h2)
            if half_sequence and elementwise_compute_dtype == "float16":
                selected_out = _residual_gate_half_compute(x_sel, mlp_out, gate_mlp)
            else:
                selected_out = x_sel + mlp_out * gate_mlp.unsqueeze(1)
                if half_sequence:
                    selected_out = _fp16_roundtrip_tensor(selected_out)
        else:
            h1 = self.norm1(x_work)
            if half_sequence:
                h1 = _fp16_roundtrip_tensor(h1)
            attn_sel = _rope_self_attention_selected_rows(
                self.attn,
                h1,
                selected_idx,
                rotary_emb,
                backend=backend,
                compute_dtype=compute_dtype,
                query_chunk_size=int(query_chunk_size),
                round_qkv_to_fp16=bool(round_qkv_to_fp16),
                round_v_to_fp16=bool(round_v_to_fp16),
                round_attn_core_to_fp16=bool(round_attn_core_to_fp16),
                half_sequence=bool(half_sequence),
            )
            x_base_sel = x_work.index_select(1, selected_idx)
            if half_sequence and elementwise_compute_dtype == "float16":
                x_sel = _residual_add_half_compute(x_base_sel, attn_sel)
                selected_out = _residual_add_half_compute(x_sel, _feed_forward_half_sequence(self.mlp, _fp16_roundtrip_tensor(self.norm2(x_sel))))
            elif half_sequence:
                x_sel = _fp16_roundtrip_tensor(x_base_sel + attn_sel)
                selected_out = _fp16_roundtrip_tensor(x_sel + _feed_forward_half_sequence(self.mlp, _fp16_roundtrip_tensor(self.norm2(x_sel))))
            else:
                x_sel = x_base_sel + attn_sel
                selected_out = x_sel + self.mlp(self.norm2(x_sel))

        if inplace_output:
            x.index_copy_(1, selected_idx, selected_out)
            return x
        out = x.clone()
        out.index_copy_(1, selected_idx, selected_out)
        return out

    if not hasattr(block, "_original_forward"):
        block._original_forward = block.forward
    block.forward = types.MethodType(patched_forward, block)
    return {
        "enabled": True,
        "kind": "final_main_block_selective_consumed_rows_patch",
        "backend": backend,
        "compute_dtype": compute_dtype,
        "query_chunk_size": int(query_chunk_size) if backend == "chunked" else None,
        "round_qkv_to_fp16": bool(round_qkv_to_fp16),
        "round_v_to_fp16": bool(round_v_to_fp16),
        "round_attn_core_to_fp16": bool(round_attn_core_to_fp16),
        "half_sequence": bool(half_sequence),
        "elementwise_compute_dtype": elementwise_compute_dtype,
        "inplace_output": bool(inplace_output),
        "latent_length": q_token_length,
        "camera_length": cam_len,
        "condition_rows_left_uncomputed": True,
        "selected_rows_qkv_weight_dtype_aware": True,
        "selected_rows_attention_contiguous_qkv": True,
        "patched_module": f"blocks.{len(flow_model.blocks) - 1}",
        "note": "Only the final block is patched. Returned tensor shape is preserved; unused condition rows are left as input values. If inplace_output is true, the final block input tensor is updated in place after selected_out has been computed.",
    }



def apply_triposplat_late_selective_condition_freeze_patch(
    flow_model,
    *,
    enabled: bool = False,
    block_count: int = 0,
    backend: str = "default",
    compute_dtype: str = "model",
    query_chunk_size: int = 128,
    latent_length: int | None = None,
    camera_length: int | None = None,
) -> dict[str, Any]:
    """Patch late main blocks to update only latent/camera rows.

    The selected rows still attend to all K/V tokens, including condition rows.
    Condition rows are kept at the block input value instead of being updated.
    For the final block this is an exact consumed-output optimization; for
    earlier late blocks it is a non-equivalent speed mode because subsequent
    blocks see frozen condition rows.
    """
    count = int(block_count)
    if not enabled or count <= 0:
        return {"enabled": False, "block_count": count, "backend": backend, "compute_dtype": compute_dtype}
    if backend not in _valid_backends():
        raise ValueError(f"unsupported late selective backend: {backend}")
    if compute_dtype not in {"model", "float32"}:
        raise ValueError(f"unsupported late selective compute dtype: {compute_dtype}")
    if not hasattr(flow_model, "blocks") or len(flow_model.blocks) == 0:
        raise ValueError("flow_model has no main blocks to patch")

    import torch

    total_blocks = len(flow_model.blocks)
    count = min(count, total_blocks)
    start = total_blocks - count
    q_token_length = int(latent_length if latent_length is not None else flow_model.q_token_length)
    cam_len = int(camera_length if camera_length is not None else (1 if getattr(flow_model, "cam_channels", None) is not None else 0))
    patched = []

    def make_forward(block_index: int):
        def patched_forward(self, x, mod=None, rotary_emb=None):
            L = int(x.shape[1])
            if q_token_length + cam_len > L:
                raise ValueError(f"invalid selective rows for L={L}: latent={q_token_length} camera={cam_len}")
            parts = [torch.arange(0, q_token_length, device=x.device, dtype=torch.long)]
            if cam_len:
                parts.append(torch.arange(L - cam_len, L, device=x.device, dtype=torch.long))
            selected_idx = torch.cat(parts)

            if self.modulation:
                mod_work = self.adaLN_modulation(mod) if not self.share_mod else mod
                if hasattr(self, "shift_table") and self.shift_table is not None:
                    mod_work = _native_shift_table_add(self, mod_work, self.shift_table.type(mod_work.dtype))
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod_work.chunk(6, dim=1)
                h1 = self.norm1(x)
                h1 = h1 * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
                attn_sel = _rope_self_attention_selected_rows(
                    self.attn,
                    h1,
                    selected_idx,
                    rotary_emb,
                    backend=backend,
                    compute_dtype=compute_dtype,
                    query_chunk_size=int(query_chunk_size),
                )
                x_base_sel = x.index_select(1, selected_idx)
                x_sel = x_base_sel + attn_sel * gate_msa.unsqueeze(1)
                h2 = self.norm2(x_sel)
                h2 = h2 * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
                selected_out = x_sel + self.mlp(h2) * gate_mlp.unsqueeze(1)
            else:
                h1 = self.norm1(x)
                attn_sel = _rope_self_attention_selected_rows(
                    self.attn,
                    h1,
                    selected_idx,
                    rotary_emb,
                    backend=backend,
                    compute_dtype=compute_dtype,
                    query_chunk_size=int(query_chunk_size),
                )
                x_sel = x.index_select(1, selected_idx) + attn_sel
                selected_out = x_sel + self.mlp(self.norm2(x_sel))

            out = x.clone()
            out.index_copy_(1, selected_idx, selected_out)
            return out

        patched_forward.__name__ = f"late_selective_condition_freeze_block_{block_index}"
        return patched_forward

    for idx in range(start, total_blocks):
        block = flow_model.blocks[idx]
        if not hasattr(block, "_original_forward_late_selective_condition_freeze"):
            block._original_forward_late_selective_condition_freeze = block.forward
        block.forward = types.MethodType(make_forward(idx), block)
        patched.append(f"blocks.{idx}")

    return {
        "enabled": True,
        "kind": "late_main_block_selective_condition_freeze_patch",
        "backend": backend,
        "compute_dtype": compute_dtype,
        "query_chunk_size": int(query_chunk_size) if backend == "chunked" else None,
        "block_count": count,
        "start_block": start,
        "patched_blocks": patched,
        "latent_length": q_token_length,
        "camera_length": cam_len,
        "selected_rows": q_token_length + cam_len,
        "condition_rows_left_uncomputed": True,
        "non_equivalent_blocks": patched[:-1],
        "semantics": "N=1 is exact for consumed final output; N>1 is a non-equivalent speed mode that freezes condition rows in late blocks while latent/camera rows attend to all tokens.",
    }


def make_triposplat_cached_condition(flow_model, cond: dict) -> tuple[dict, dict[str, Any]]:
    """Return a condition dict with the t/noise-independent context cached.

    LatentSeqMMFlowModel refines condition tokens before concatenating them with
    latent/camera tokens. That path depends only on condition tensors and model
    weights, not on the sampler state or timestep, so it can be computed once
    and reused across flow steps without changing the model output.
    """
    import torch

    required = {"feature1"}
    missing = sorted(required - set(cond))
    if missing:
        raise ValueError(f"condition cache requires missing keys: {missing}")
    if not hasattr(flow_model, "context_refiner") or not hasattr(flow_model, "cond_embedder"):
        raise ValueError("flow_model does not look like LatentSeqMMFlowModel")
    cache_key = "_triposplat_cached_h_cond"
    if cache_key in cond:
        return cond, {"enabled": True, "already_cached": True, "cache_key": cache_key}

    d = flow_model.dtype
    started = __import__("time").time()
    with torch.no_grad():
        feat1 = cond["feature1"].to(d)
        feat2 = cond["feature2"].to(d) if getattr(flow_model, "cond_embedder2", None) is not None and "feature2" in cond else None
        h_cond = flow_model.cond_embedder(feat1)
        if feat2 is not None:
            h_cond = _native_add_tensor_inplace(flow_model, h_cond, flow_model.cond_embedder2(feat2))
        for i, block in enumerate(flow_model.context_refiner):
            h_cond = block(h_cond, mod=None, rotary_emb=flow_model.context_repo_layers[i](h_cond))
    out = dict(cond)
    out[cache_key] = h_cond.detach()
    return out, {
        "enabled": True,
        "kind": "static_condition_context_cache",
        "cache_key": cache_key,
        "condition_tokens": int(h_cond.shape[1]),
        "model_channels": int(h_cond.shape[2]),
        "batch": int(h_cond.shape[0]),
        "dtype": str(h_cond.dtype).replace("torch.", ""),
        "device": str(h_cond.device),
        "elapsed_sec": __import__("time").time() - started,
        "semantics": "Caches cond_embedder + context_refiner output, which is independent of sampler state and timestep.",
    }


def apply_triposplat_static_condition_cache_patch(
    flow_model,
    *,
    enabled: bool = False,
) -> dict[str, Any]:
    """Patch LatentSeqMMFlowModel.forward to consume cached condition context."""
    if not enabled:
        return {"enabled": False}
    if not hasattr(flow_model, "context_refiner") or not hasattr(flow_model, "blocks"):
        raise ValueError("flow_model does not look like LatentSeqMMFlowModel")

    import torch
    import torch.nn.functional as F

    cache_key = "_triposplat_cached_h_cond"

    def patched_forward(self, x_t, t, cond):
        d = self.dtype
        z = x_t["latent"].to(d)
        feat1 = cond["feature1"].to(d)
        feat2 = cond["feature2"].to(d) if self.cond_embedder2 is not None and "feature2" in cond else None
        self.pos_pe = self.pos_pe.to(z.device)

        h_x = self.input_layer(z)
        if cache_key in cond:
            h_cond = cond[cache_key].to(device=z.device, dtype=d)
        else:
            h_cond = self.cond_embedder(feat1)
            if feat2 is not None:
                h_cond = _native_add_tensor_inplace(self, h_cond, self.cond_embedder2(feat2))
        t_emb = self.t_embedder(t)
        t_mod = self.adaLN_modulation(t_emb) if self.share_mod else t_emb

        h_x = _native_add_tensor_inplace(self, h_x, self.pos_embedder(self.pos_pe).to(d))

        for i, block in enumerate(self.noise_refiner):
            h_x = block(h_x, mod=t_mod, rotary_emb=self.noise_repo_layers[i](h_x))

        if cache_key not in cond:
            for i, block in enumerate(self.context_refiner):
                h_cond = block(h_cond, mod=None, rotary_emb=self.context_repo_layers[i](h_cond))

        if self.cam_channels is not None:
            cam = x_t.get("camera").to(d)
            h_cam = self.cam_refiner(cam)

        h = torch.cat([h_x, h_cond], dim=1)
        if self.cam_channels is not None:
            h = torch.cat([h, h_cam], dim=1)

        for i, block in enumerate(self.blocks):
            h = block(h, mod=t_mod, rotary_emb=self.repo_layers[i](h))

        h_x = _native_functional_layer_norm(self, h[:, : z.shape[1]])
        if self.cam_channels is not None:
            h_cam = _native_functional_layer_norm(self, h[:, -cam.shape[1] :])

        if self.use_shift_table:
            shift, scale = _native_final_modulation_params(self, self.shift_table, t_emb)
            h_x = _native_modulate_tensor(self, h_x, scale, shift)
            if self.cam_channels is not None:
                h_cam = _native_modulate_tensor(self, h_cam, scale, shift)

        out = {"latent": self.out_layer(h_x)}
        if self.cam_channels is not None:
            out["camera"] = self.cam_out_layer(h_cam)
        return out

    if not hasattr(flow_model, "_original_forward_static_condition_cache"):
        flow_model._original_forward_static_condition_cache = flow_model.forward
    flow_model.forward = types.MethodType(patched_forward, flow_model)
    return {
        "enabled": True,
        "kind": "static_condition_context_cache_forward_patch",
        "cache_key": cache_key,
        "semantics": "Uses a precomputed condition context when present; falls back to the original condition path otherwise.",
    }


def apply_triposplat_cfg_duplicate_state_patch(
    flow_model,
    *,
    enabled: bool = False,
    assume_duplicated: bool = False,
) -> dict[str, Any]:
    """Patch flow forward to avoid duplicate CFG state-side work.

    The non-split CFG sampler calls the model with [state, state] and
    [cond, neg_cond]. The condition path differs, but input_layer,
    pos_embedder, noise_refiner, cam_refiner, and timestep modulation see
    identical state/timestep values in the two halves. This patch computes that
    state-side prefix for the first half and repeats it before the main blocks.
    If the runtime batch does not have identical halves, it falls back to the
    normal full-batch state-side computation. For runner-controlled non-split
    CFG, assume_duplicated can skip the torch.equal runtime guard so
    torch.compile sees a static branch.
    """
    if not enabled:
        return {"enabled": False}
    if not hasattr(flow_model, "noise_refiner") or not hasattr(flow_model, "blocks"):
        raise ValueError("flow_model does not look like LatentSeqMMFlowModel")

    import torch
    import torch.nn.functional as F

    cache_key = "_triposplat_cached_h_cond"

    def _duplicated_halves_tensor(x):
        if not torch.is_tensor(x) or int(x.shape[0]) < 2 or int(x.shape[0]) % 2 != 0:
            return False
        half = int(x.shape[0]) // 2
        return bool(torch.equal(x[:half], x[half:]))

    def _cond_batch(cond):
        if cache_key in cond and torch.is_tensor(cond[cache_key]):
            return int(cond[cache_key].shape[0])
        if "feature1" in cond and torch.is_tensor(cond["feature1"]):
            return int(cond["feature1"].shape[0])
        return None

    def _can_dedup_state(x_t, t):
        if not _duplicated_halves_tensor(t):
            return False
        if not _duplicated_halves_tensor(x_t["latent"]):
            return False
        if self_cam_channels is not None and "camera" in x_t and not _duplicated_halves_tensor(x_t["camera"]):
            return False
        return True

    def _can_single_state_batched_cfg(x_t, t, cond):
        state_batch = int(x_t["latent"].shape[0])
        cond_batch = _cond_batch(cond)
        if cond_batch is None or cond_batch != state_batch * 2:
            return False
        if not torch.is_tensor(t) or int(t.shape[0]) != cond_batch:
            return False
        return bool(torch.equal(t[:state_batch], t[state_batch:]))

    self_cam_channels = getattr(flow_model, "cam_channels", None)

    def patched_forward(self, x_t, t, cond):
        d = self.dtype
        z_full = x_t["latent"].to(d)
        state_batch = int(z_full.shape[0])
        single_state_batched_cfg = _can_single_state_batched_cfg(x_t, t, cond)
        if assume_duplicated:
            if single_state_batched_cfg:
                dedup_state = True
            elif state_batch >= 2 and state_batch % 2 == 0:
                dedup_state = True
            else:
                raise RuntimeError("assume_duplicated CFG state requires either even duplicated state batch or single-state batched CFG")
        else:
            dedup_state = single_state_batched_cfg or _can_dedup_state(x_t, t)
        half = state_batch if single_state_batched_cfg else (state_batch // 2 if dedup_state else state_batch)
        z_work = z_full[:half] if dedup_state else z_full
        t_work = t[:half] if dedup_state else t

        feat1 = cond["feature1"].to(d)
        feat2 = cond["feature2"].to(d) if self.cond_embedder2 is not None and "feature2" in cond else None
        self.pos_pe = self.pos_pe.to(z_full.device)

        h_x = self.input_layer(z_work)
        if cache_key in cond:
            h_cond = cond[cache_key].to(device=z_full.device, dtype=d)
        else:
            h_cond = self.cond_embedder(feat1)
            if feat2 is not None:
                h_cond = _native_add_tensor_inplace(self, h_cond, self.cond_embedder2(feat2))
        t_emb_work = self.t_embedder(t_work)
        t_mod_work = self.adaLN_modulation(t_emb_work) if self.share_mod else t_emb_work

        h_x = _native_add_tensor_inplace(self, h_x, self.pos_embedder(self.pos_pe).to(d))

        for i, block in enumerate(self.noise_refiner):
            h_x = block(h_x, mod=t_mod_work, rotary_emb=self.noise_repo_layers[i](h_x))

        if dedup_state:
            h_x = h_x.repeat(2, *([1] * (h_x.dim() - 1)))
            t_emb = t_emb_work.repeat(2, *([1] * (t_emb_work.dim() - 1)))
            t_mod = t_mod_work.repeat(2, *([1] * (t_mod_work.dim() - 1)))
        else:
            t_emb = t_emb_work
            t_mod = t_mod_work

        if cache_key not in cond:
            for i, block in enumerate(self.context_refiner):
                h_cond = block(h_cond, mod=None, rotary_emb=self.context_repo_layers[i](h_cond))

        if self.cam_channels is not None:
            cam_full = x_t.get("camera").to(d)
            cam_work = cam_full[:half] if dedup_state else cam_full
            h_cam = self.cam_refiner(cam_work)
            if dedup_state:
                h_cam = h_cam.repeat(2, *([1] * (h_cam.dim() - 1)))

        h = torch.cat([h_x, h_cond], dim=1)
        if self.cam_channels is not None:
            h = torch.cat([h, h_cam], dim=1)

        for i, block in enumerate(self.blocks):
            h = block(h, mod=t_mod, rotary_emb=self.repo_layers[i](h))

        h_x = _native_functional_layer_norm(self, h[:, : z_full.shape[1]])
        if self.cam_channels is not None:
            h_cam = _native_functional_layer_norm(self, h[:, -cam_full.shape[1] :])

        if self.use_shift_table:
            shift, scale = _native_final_modulation_params(self, self.shift_table, t_emb)
            h_x = _native_modulate_tensor(self, h_x, scale, shift)
            if self.cam_channels is not None:
                h_cam = _native_modulate_tensor(self, h_cam, scale, shift)

        out = {"latent": self.out_layer(h_x)}
        if self.cam_channels is not None:
            out["camera"] = self.cam_out_layer(h_cam)
        return out

    if not hasattr(flow_model, "_original_forward_cfg_duplicate_state"):
        flow_model._original_forward_cfg_duplicate_state = flow_model.forward
    flow_model.forward = types.MethodType(patched_forward, flow_model)
    return {
        "enabled": True,
        "kind": "cfg_duplicate_state_forward_patch",
        "cache_key_honored": cache_key,
        "runtime_guard": "skipped under assume_duplicated; otherwise deduplicate only when latent/timestep/camera batch halves are bit-identical or when state batch is N and condition/timestep batch is 2N",
        "assume_duplicated": bool(assume_duplicated),
        "single_state_batched_cfg": "supported when state batch is N and condition/timestep batch is 2N with duplicated timesteps",
        "semantics": "Exact for runner-controlled non-split CFG batches produced either as [state, state] with [cond, neg_cond], or as single state batch N with condition/timestep batch 2N.",
    }


def apply_triposplat_position_embed_cache_patch(
    flow_model,
    *,
    enabled: bool = False,
) -> dict[str, Any]:
    """Cache the fixed absolute position embedding used for latent tokens.

    This only wraps ``flow_model.pos_embedder``. The RePo/rotary embeddings are
    not cached because they depend on the current hidden-state values.
    """
    if not enabled:
        return {"enabled": False}
    if not hasattr(flow_model, "pos_embedder") or not hasattr(flow_model, "pos_pe"):
        raise ValueError("flow_model has no pos_embedder/pos_pe to cache")

    import types

    module = flow_model.pos_embedder
    if not hasattr(module, "_original_forward_position_embed_cache"):
        module._original_forward_position_embed_cache = module.forward

    cache: dict[tuple[Any, ...], Any] = {}
    stats = {"hits": 0, "misses": 0}

    def patched_forward(self, x):
        key = (
            id(x),
            int(getattr(x, "_version", 0)),
            tuple(int(v) for v in x.shape),
            str(x.device),
            str(x.dtype).replace("torch.", ""),
            int(getattr(self, "channels", -1)),
            int(getattr(self, "freq_dim", -1)),
            int(getattr(self, "max_res", -1)),
        )
        if key in cache:
            stats["hits"] += 1
            return cache[key]
        stats["misses"] += 1
        out = self._original_forward_position_embed_cache(x)
        cache.clear()
        cache[key] = out
        return out

    module.forward = types.MethodType(patched_forward, module)
    return {
        "enabled": True,
        "kind": "fixed_absolute_position_embedding_cache",
        "patched_module": "pos_embedder",
        "cache_limit": 1,
        "stats": stats,
        "semantics": "Exact cache for pos_embedder(pos_pe). RePo/rotary embeddings are intentionally not cached because they depend on hidden states.",
    }

def apply_triposplat_repo_polar_cos_sin_patch(
    flow_model,
    *,
    enabled: bool = False,
) -> dict[str, Any]:
    """Patch RePo rotary frequency construction to avoid torch.polar.

    RePo3DRotaryEmbedding depends on the current hidden state, so the embedding
    itself cannot be cached. This patch keeps the same angle computation and
    constructs the complex unit phasor from cos/sin instead of
    torch.polar(ones_like(angle), angle), which is faster on the current CPU
    microbench and differs only by normal fp32 transcendental roundoff.
    """
    if not enabled:
        return {"enabled": False}

    import types
    import torch

    patched: list[str] = []

    def _clamp_mul(x, f):
        f_t = f.tanh()
        return x * f_t + x.detach() * (f - f_t)

    def patched_forward(self, hidden_states):
        h = self.norm(hidden_states)
        feat = self.act(self.gate_map(h)) * self.content_map(h)
        out = self.final_map(feat)
        B, L, _ = out.shape
        delta_pos = out.reshape(B, L, self.num_heads, 3)
        ang_0 = _clamp_mul(delta_pos[..., 0].unsqueeze(-1), self.freqs_0) * torch.pi
        ang_1 = _clamp_mul(delta_pos[..., 1].unsqueeze(-1), self.freqs_1) * torch.pi
        ang_2 = _clamp_mul(delta_pos[..., 2].unsqueeze(-1), self.freqs_2) * torch.pi
        ang = torch.cat([ang_0, ang_1, ang_2], dim=-1).float()
        return torch.complex(ang.cos(), ang.sin()).type(torch.complex64)

    for name, module in flow_model.named_modules():
        if module.__class__.__name__ != "RePo3DRotaryEmbedding":
            continue
        if not hasattr(module, "_original_forward_repo_polar_cos_sin"):
            module._original_forward_repo_polar_cos_sin = module.forward
        module.forward = types.MethodType(patched_forward, module)
        patched.append(name)

    return {
        "enabled": True,
        "kind": "repo_rotary_polar_cos_sin_patch",
        "selected_count": len(patched),
        "selected": patched,
        "semantics": "Replaces torch.polar(ones_like(angle), angle) with torch.complex(cos(angle), sin(angle)) inside RePo3DRotaryEmbedding.forward.",
    }


def apply_triposplat_module_attention_patch(
    flow_model,
    *,
    include_regex: str | None,
    exclude_regex: str | None = None,
    backend: str = "flash",
    compute_dtype: str = "float32",
    query_chunk_size: int = 128,
    linear_dtype: str = "model",
) -> dict[str, Any]:
    """Patch selected RopeMultiHeadAttention modules by module name.

    This keeps dense full attention semantics while allowing precision/backend
    experiments to target noise_refiner, context_refiner, or main blocks
    separately.
    """
    if not include_regex:
        return {
            "enabled": False,
            "backend": backend,
            "compute_dtype": compute_dtype,
            "query_chunk_size": int(query_chunk_size) if backend == "chunked" else None,
            "streaming_mode": _streaming_mode(backend) if _is_streaming_backend(backend) else None,
            "streaming_threads_env": os.environ.get("STREAMING_SDPA_THREADS"),
            "effective_compute_dtype": "float32" if _is_streaming_backend(backend) else compute_dtype,
            "linear_dtype": linear_dtype,
            "include_regex": include_regex,
            "exclude_regex": exclude_regex,
            "selected_count": 0,
        }
    if backend not in _valid_backends():
        raise ValueError(f"unsupported attention backend: {backend}")
    if compute_dtype not in {"model", "float32"}:
        raise ValueError(f"unsupported attention compute dtype: {compute_dtype}")
    if linear_dtype not in {"model", "float32"}:
        raise ValueError(f"unsupported attention module linear dtype: {linear_dtype}")

    import torch

    try:
        import model as triposplat_model
    except Exception as exc:  # pragma: no cover - depends on runtime sys.path
        raise RuntimeError("could not import TripoSplat model module for module attention patch") from exc

    include = re.compile(include_regex)
    exclude = re.compile(exclude_regex) if exclude_regex else None
    selected: list[str] = []
    skipped: list[str] = []

    def make_forward(module_name: str):
        def patched_forward(self, x, context=None, rope_emb=None):
            B, L, C = x.shape
            orig_dtype = x.dtype
            work_dtype = torch.float32 if linear_dtype == "float32" else orig_dtype
            x_work = x.to(dtype=work_dtype) if x.dtype != work_dtype else x
            if self._type == "self":
                qkv = self.qkv(x_work).reshape(B, L, 3, self.num_heads, self.head_dim)
                q, k, v = qkv.unbind(2)
                if self.use_rope:
                    q = triposplat_model.apply_rotary_emb(q, rope_emb)
                    k = triposplat_model.apply_rotary_emb(k, rope_emb)
            else:
                q = self.q(x_work).reshape(B, L, self.num_heads, self.head_dim)
                if context is None:
                    raise ValueError("Context must be provided for cross attention")
                context_work = context.to(dtype=work_dtype) if context.dtype != work_dtype else context
                kv = self.kv(context_work).reshape(B, context.shape[1], 2, self.num_heads, self.head_dim)
                k, v = kv.unbind(2)
            if self.qk_rms_norm:
                q = self.q_norm(q)
                k = self.k_norm(k)
            h = _attention_core(
                q,
                k,
                v,
                backend=backend,
                compute_dtype=compute_dtype,
                query_chunk_size=int(query_chunk_size),
            )
            out = self.out(h.reshape(B, L, C))
            return out.to(dtype=orig_dtype) if out.dtype != orig_dtype else out

        patched_forward.__name__ = f"patched_forward_{module_name.replace('.', '_')}"
        return patched_forward

    for name, module in flow_model.named_modules():
        if module.__class__.__name__ != "RopeMultiHeadAttention":
            continue
        if include.search(name) is None:
            skipped.append(name)
            continue
        if exclude is not None and exclude.search(name) is not None:
            skipped.append(name)
            continue
        if linear_dtype == "float32":
            module.to(dtype=torch.float32)
        if not hasattr(module, "_original_forward"):
            module._original_forward = module.forward
        module.forward = types.MethodType(make_forward(name), module)
        selected.append(name)

    return {
        "enabled": bool(selected),
        "kind": "module_rope_multi_head_attention_runtime_patch",
        "backend": backend,
        "compute_dtype": compute_dtype,
        "query_chunk_size": int(query_chunk_size) if backend == "chunked" else None,
        "streaming_mode": _streaming_mode(backend) if _is_streaming_backend(backend) else None,
        "streaming_threads_env": os.environ.get("STREAMING_SDPA_THREADS"),
        "effective_compute_dtype": "float32" if _is_streaming_backend(backend) else compute_dtype,
        "linear_dtype": linear_dtype,
        "include_regex": include_regex,
        "exclude_regex": exclude_regex,
        "selected_count": len(selected),
        "selected": selected,
        "skipped_count": len(skipped),
        "skipped_sample": skipped[:50],
        "cross_attention_fallback": "chunked_float32_exact" if _is_streaming_backend(backend) else None,
        "note": "Selected RopeMultiHeadAttention.forward methods are patched; dense full attention semantics are preserved.",
    }



def _fp16_roundtrip_tensor(x):
    import torch

    if torch.is_tensor(x) and x.is_floating_point():
        return x.to(dtype=torch.float16).to(dtype=x.dtype)
    return x


def _fp16_compute_tensor(x):
    import torch

    if torch.is_tensor(x) and x.is_floating_point():
        return x.to(dtype=torch.float16)
    return x


def _fp16_restore_tensor(x, dtype):
    import torch

    if torch.is_tensor(x) and x.is_floating_point() and dtype is not None:
        return x.to(dtype=dtype)
    return x


def _modulated_half_compute(h, scale, shift):
    import torch

    dtype = h.dtype
    h16 = _fp16_compute_tensor(h)
    scale16 = _fp16_compute_tensor(scale).unsqueeze(1)
    shift16 = _fp16_compute_tensor(shift).unsqueeze(1)
    return _fp16_restore_tensor(h16 * (torch.ones_like(scale16) + scale16) + shift16, dtype)


def _residual_gate_half_compute(x, h, gate):
    dtype = x.dtype
    x16 = _fp16_compute_tensor(x)
    h16 = _fp16_compute_tensor(h)
    gate16 = _fp16_compute_tensor(gate).unsqueeze(1)
    return _fp16_restore_tensor(x16 + h16 * gate16, dtype)


def _residual_add_half_compute(x, h):
    dtype = x.dtype
    return _fp16_restore_tensor(_fp16_compute_tensor(x) + _fp16_compute_tensor(h), dtype)


def _feed_forward_half_sequence(mlp, x):
    """Run FeedForwardNet in official order with fp16-value roundtrips."""
    import torch
    native_dtype = getattr(mlp, "_native_sequence_compute_dtype", None)
    if native_dtype is not None:
        if not hasattr(mlp, "mlp"):
            return mlp(x.to(dtype=native_dtype)).to(dtype=torch.float32)
        h = x.to(dtype=native_dtype)
        for layer in mlp.mlp:
            h = layer(h)
        return h.to(dtype=torch.float32)
    if not hasattr(mlp, "mlp"):
        return _fp16_roundtrip_tensor(mlp(x))
    layers = list(mlp.mlp)
    if not layers:
        return x
    h = x
    for layer in layers:
        weight = getattr(layer, "weight", None)
        if torch.is_tensor(weight) and weight.is_floating_point() and h.dtype != weight.dtype:
            h = h.to(dtype=weight.dtype)
        h = _fp16_roundtrip_tensor(layer(h))
    return h


def _rope_attention_half_sequence(attn, x, rope_emb, *, backend: str, compute_dtype: str, query_chunk_size: int):
    try:
        import model as triposplat_model
    except Exception as exc:  # pragma: no cover - depends on runtime sys.path
        raise RuntimeError("could not import TripoSplat model module for half-sequence patch") from exc

    if attn._type != "self":
        return _fp16_roundtrip_tensor(attn(x, rope_emb=rope_emb))
    B, L, C = x.shape
    qkv = attn.qkv(x).reshape(B, L, 3, attn.num_heads, attn.head_dim)
    qkv = _fp16_roundtrip_tensor(qkv)
    q, k, v = qkv.unbind(2)
    if attn.use_rope:
        q = _fp16_roundtrip_tensor(triposplat_model.apply_rotary_emb(q, rope_emb))
        k = _fp16_roundtrip_tensor(triposplat_model.apply_rotary_emb(k, rope_emb))
    if attn.qk_rms_norm:
        q = _fp16_roundtrip_tensor(attn.q_norm(q))
        k = _fp16_roundtrip_tensor(attn.k_norm(k))
    v = _fp16_roundtrip_tensor(v)
    h = _attention_core(
        q,
        k,
        v,
        backend=backend,
        compute_dtype=compute_dtype,
        query_chunk_size=int(query_chunk_size),
    )
    h = _fp16_roundtrip_tensor(h)
    return _fp16_roundtrip_tensor(attn.out(h.reshape(B, L, C)))


def apply_triposplat_unified_block_half_sequence_patch(
    flow_model,
    *,
    enabled: bool = False,
    include_regex: str | None = None,
    exclude_regex: str | None = None,
    backend: str = "default",
    compute_dtype: str = "model",
    query_chunk_size: int = 128,
    elementwise_compute_dtype: str = "roundtrip",
) -> dict[str, Any]:
    """Patch UnifiedTransformerBlock.forward with explicit half-value sequence.

    Tensors stay in the runtime dtype, but selected intermediates are
    roundtripped through float16 at the official operation boundaries.
    """
    if not enabled:
        return {
            "enabled": False,
            "backend": backend,
            "compute_dtype": compute_dtype,
            "include_regex": include_regex,
            "exclude_regex": exclude_regex,
        }
    if backend not in _valid_backends():
        raise ValueError(f"unsupported half-sequence attention backend: {backend}")
    if compute_dtype not in {"model", "float32"}:
        raise ValueError(f"unsupported half-sequence attention compute dtype: {compute_dtype}")
    if elementwise_compute_dtype not in {"roundtrip", "float16"}:
        raise ValueError(f"unsupported half-sequence elementwise compute dtype: {elementwise_compute_dtype}")

    include = re.compile(include_regex) if include_regex else None
    exclude = re.compile(exclude_regex) if exclude_regex else None
    selected: list[str] = []
    skipped: list[str] = []

    def make_forward(module_name: str):
        def patched_forward(self, x, mod=None, rotary_emb=None):
            x = _fp16_roundtrip_tensor(x)
            if mod is not None:
                mod = _fp16_roundtrip_tensor(mod)
            if self.modulation:
                if not self.share_mod:
                    mod_work = _fp16_roundtrip_tensor(self.adaLN_modulation(mod))
                else:
                    mod_work = mod
                if hasattr(self, "shift_table") and self.shift_table is not None:
                    mod_work = _fp16_roundtrip_tensor(mod_work + self.shift_table.type(mod_work.dtype))
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod_work.chunk(6, dim=1)
                h = _fp16_roundtrip_tensor(self.norm1(x))
                if elementwise_compute_dtype == "float16":
                    h = _modulated_half_compute(h, scale_msa, shift_msa)
                else:
                    h = _fp16_roundtrip_tensor(h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1))
                h = _rope_attention_half_sequence(
                    self.attn,
                    h,
                    rotary_emb,
                    backend=backend,
                    compute_dtype=compute_dtype,
                    query_chunk_size=int(query_chunk_size),
                )
                if elementwise_compute_dtype == "float16":
                    x = _residual_gate_half_compute(x, h, gate_msa)
                else:
                    x = _fp16_roundtrip_tensor(x + h * gate_msa.unsqueeze(1))
                h = _fp16_roundtrip_tensor(self.norm2(x))
                if elementwise_compute_dtype == "float16":
                    h = _modulated_half_compute(h, scale_mlp, shift_mlp)
                else:
                    h = _fp16_roundtrip_tensor(h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1))
                h = _feed_forward_half_sequence(self.mlp, h)
                if elementwise_compute_dtype == "float16":
                    x = _residual_gate_half_compute(x, h, gate_mlp)
                else:
                    x = _fp16_roundtrip_tensor(x + h * gate_mlp.unsqueeze(1))
            else:
                h = _fp16_roundtrip_tensor(self.norm1(x))
                h = _rope_attention_half_sequence(
                    self.attn,
                    h,
                    rotary_emb,
                    backend=backend,
                    compute_dtype=compute_dtype,
                    query_chunk_size=int(query_chunk_size),
                )
                if elementwise_compute_dtype == "float16":
                    x = _residual_add_half_compute(x, h)
                else:
                    x = _fp16_roundtrip_tensor(x + h)
                h = _fp16_roundtrip_tensor(self.norm2(x))
                h = _feed_forward_half_sequence(self.mlp, h)
                if elementwise_compute_dtype == "float16":
                    x = _residual_add_half_compute(x, h)
                else:
                    x = _fp16_roundtrip_tensor(x + h)
            return x

        patched_forward.__name__ = f"half_sequence_forward_{module_name.replace('.', '_')}"
        return patched_forward

    for name, module in flow_model.named_modules():
        if module.__class__.__name__ != "UnifiedTransformerBlock":
            continue
        if include is not None and include.search(name) is None:
            skipped.append(name)
            continue
        if exclude is not None and exclude.search(name) is not None:
            skipped.append(name)
            continue
        if not hasattr(module, "_original_forward"):
            module._original_forward = module.forward
        module.forward = types.MethodType(make_forward(name), module)
        selected.append(name)

    return {
        "enabled": bool(selected),
        "kind": "unified_transformer_block_half_sequence_runtime_patch",
        "backend": backend,
        "compute_dtype": compute_dtype,
        "query_chunk_size": int(query_chunk_size) if backend == "chunked" else None,
        "elementwise_compute_dtype": elementwise_compute_dtype,
        "streaming_mode": _streaming_mode(backend) if _is_streaming_backend(backend) else None,
        "streaming_threads_env": os.environ.get("STREAMING_SDPA_THREADS"),
        "effective_compute_dtype": "float32" if _is_streaming_backend(backend) else compute_dtype,
        "include_regex": include_regex,
        "exclude_regex": exclude_regex,
        "selected_count": len(selected),
        "selected": selected,
        "skipped_count": len(skipped),
        "skipped_sample": skipped[:50],
        "note": "Selected UnifiedTransformerBlock.forward methods are patched with explicit fp16-value roundtrips in official operation order.",
    }


def apply_triposplat_addcmul_elementwise_patch(
    flow_model,
    *,
    enabled: bool = False,
    include_regex: str | None = r"^blocks[.][0-9]+$",
    exclude_regex: str | None = None,
) -> dict[str, Any]:
    """Patch UnifiedTransformerBlock.forward to use torch.addcmul for gate/modulation elementwise ops.

    This keeps the same high-level block order as the official implementation,
    but uses fused elementwise kernels for ``h * (1 + scale) + shift`` and
    ``x + h * gate``. Floating-point accumulation order is not bit-exact, so the
    candidate must pass the latent quality gate before adoption.
    """
    if not enabled:
        return {"enabled": False, "include_regex": include_regex, "exclude_regex": exclude_regex}
    include = re.compile(include_regex) if include_regex else None
    exclude = re.compile(exclude_regex) if exclude_regex else None
    selected: list[str] = []
    skipped: list[str] = []

    def make_forward(module_name: str):
        def patched_forward(self, x, mod=None, rotary_emb=None):
            if self.modulation:
                if not self.share_mod:
                    mod = self.adaLN_modulation(mod)
                if hasattr(self, "shift_table") and self.shift_table is not None:
                    mod = mod + self.shift_table.type(mod.dtype)
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
                h = self.norm1(x)
                h = _modulate_addcmul(h, scale_msa, shift_msa)
                h = self.attn(h, rope_emb=rotary_emb)
                x = _residual_gate_addcmul(x, h, gate_msa)
                h = self.norm2(x)
                h = _modulate_addcmul(h, scale_mlp, shift_mlp)
                x = _residual_gate_addcmul(x, self.mlp(h), gate_mlp)
            else:
                x = x + self.attn(self.norm1(x), rope_emb=rotary_emb)
                x = x + self.mlp(self.norm2(x))
            return x

        patched_forward.__name__ = f"addcmul_elementwise_forward_{module_name.replace('.', '_')}"
        return patched_forward

    for name, module in flow_model.named_modules():
        if module.__class__.__name__ != "UnifiedTransformerBlock":
            continue
        if include is not None and include.search(name) is None:
            skipped.append(name)
            continue
        if exclude is not None and exclude.search(name) is not None:
            skipped.append(name)
            continue
        if not hasattr(module, "_original_forward_addcmul_elementwise"):
            module._original_forward_addcmul_elementwise = module.forward
        module.forward = types.MethodType(make_forward(name), module)
        selected.append(name)

    return {
        "enabled": bool(selected),
        "kind": "unified_transformer_block_addcmul_elementwise_patch",
        "include_regex": include_regex,
        "exclude_regex": exclude_regex,
        "selected_count": len(selected),
        "selected": selected,
        "skipped_count": len(skipped),
        "skipped_sample": skipped[:50],
        "semantics": "Uses torch.addcmul for modulation and gated residual elementwise ops. Operation order is close but not bit-exact; quality gate decides adoption.",
    }

def apply_triposplat_mkldnn_fused_mlp_patch(
    flow_model,
    *,
    enabled: bool = False,
    include_regex: str | None = None,
    exclude_regex: str | None = None,
) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False}

    import re
    import torch
    import torch.nn as nn
    import types

    if not hasattr(torch.ops, "mkldnn") or not hasattr(torch.ops.mkldnn, "_linear_pointwise"):
        raise RuntimeError("mkldnn._linear_pointwise is not available in this PyTorch build")

    include = re.compile(include_regex) if include_regex else None
    exclude = re.compile(exclude_regex) if exclude_regex else None
    patched: list[str] = []
    skipped: list[str] = []

    def matches(name: str) -> bool:
        if include is not None and include.search(name) is None:
            return False
        if exclude is not None and exclude.search(name) is not None:
            return False
        return True

    def make_forward(module_name: str):
        def patched_forward(self, x):
            l0 = self.mlp[0]
            l2 = self.mlp[2]
            h = torch.ops.mkldnn._linear_pointwise(
                x,
                l0.weight,
                l0.bias,
                "gelu",
                [],
                "tanh",
            )
            return l2(h)

        patched_forward.__name__ = f"mkldnn_fused_mlp_forward_{module_name.replace(chr(46), chr(95))}"
        return patched_forward

    for name, module in flow_model.named_modules():
        if not matches(name):
            continue
        if not hasattr(module, "mlp") or not isinstance(module.mlp, nn.Sequential) or len(module.mlp) != 3:
            continue
        l0, act, l2 = module.mlp[0], module.mlp[1], module.mlp[2]
        if not isinstance(l0, nn.Linear) or not isinstance(l2, nn.Linear):
            skipped.append(name)
            continue
        if not isinstance(act, nn.GELU) or getattr(act, "approximate", None) != "tanh":
            skipped.append(name)
            continue
        if not hasattr(module, "_original_forward_mkldnn_fused_mlp"):
            module._original_forward_mkldnn_fused_mlp = module.forward
        module.forward = types.MethodType(make_forward(name), module)
        patched.append(name)

    return {
        "enabled": True,
        "kind": "mkldnn_linear_pointwise_gelu_tanh_mlp_patch",
        "include_regex": include_regex,
        "exclude_regex": exclude_regex,
        "selected_count": len(patched),
        "selected": patched,
        "skipped_count": len(skipped),
        "skipped_sample": skipped[:32],
        "semantics": "Fuses first Linear plus GELU tanh in FeedForwardNet style MLPs through oneDNN. Close numeric agreement but not bit-identical to torch.nn.GELU.",
    }




def apply_triposplat_gelu_out_buffer_mlp_patch(
    flow_model,
    *,
    enabled: bool = False,
    include_regex: str | None = None,
    exclude_regex: str | None = None,
) -> dict[str, Any]:
    """Patch FeedForwardNet-style MLPs to write GELU into the first Linear output.

    This is an exact CPU float32 candidate for modules shaped as
    Sequential(Linear, GELU(approximate="tanh"), Linear). The first Linear
    output is not reused by the original MLP, so writing GELU back into that
    tensor preserves the operation order while avoiding a separate GELU output
    allocation.
    """
    if not enabled:
        return {"enabled": False}

    import re
    import torch
    import torch.nn as nn
    import types

    include = re.compile(include_regex) if include_regex else None
    exclude = re.compile(exclude_regex) if exclude_regex else None
    patched: list[str] = []
    skipped: list[str] = []

    if not hasattr(torch.ops.aten.gelu, "out"):
        raise RuntimeError("aten.gelu.out is not available in this PyTorch build")

    def matches(name: str) -> bool:
        if include is not None and include.search(name) is None:
            return False
        if exclude is not None and exclude.search(name) is not None:
            return False
        return True

    def make_forward(module_name: str):
        def patched_forward(self, x):
            l0 = self.mlp[0]
            act = self.mlp[1]
            l2 = self.mlp[2]
            h = l0(x)
            torch.ops.aten.gelu.out(h, approximate=getattr(act, "approximate", "none"), out=h)
            return l2(h)

        patched_forward.__name__ = f"gelu_out_buffer_mlp_forward_{module_name.replace(chr(46), chr(95))}"
        return patched_forward

    for name, module in flow_model.named_modules():
        if not matches(name):
            continue
        if not hasattr(module, "mlp") or not isinstance(module.mlp, nn.Sequential) or len(module.mlp) != 3:
            continue
        l0, act, l2 = module.mlp[0], module.mlp[1], module.mlp[2]
        if not isinstance(l0, nn.Linear) or not isinstance(l2, nn.Linear):
            skipped.append(name)
            continue
        if not isinstance(act, nn.GELU) or getattr(act, "approximate", None) != "tanh":
            skipped.append(name)
            continue
        if not hasattr(module, "_original_forward_gelu_out_buffer_mlp"):
            module._original_forward_gelu_out_buffer_mlp = module.forward
        module.forward = types.MethodType(make_forward(name), module)
        patched.append(name)

    return {
        "enabled": bool(patched),
        "kind": "gelu_out_buffer_mlp_patch",
        "include_regex": include_regex,
        "exclude_regex": exclude_regex,
        "selected_count": len(patched),
        "selected": patched[:100],
        "skipped_count": len(skipped),
        "skipped_sample": skipped[:32],
        "semantics": "Exact CPU float32 candidate. Computes Linear -> GELU(tanh) -> Linear but writes GELU into the first Linear output tensor via aten.gelu.out.",
    }


def apply_triposplat_chunked_mlp_patch(
    flow_model,
    *,
    enabled: bool = False,
    include_regex: str | None = None,
    exclude_regex: str | None = None,
    large_chunk_rows: int = 4096,
    small_chunk_rows: int = 2048,
    large_row_threshold: int = 10000,
    min_rows: int = 4096,
    cache_weight_t: bool = True,
) -> dict[str, Any]:
    """Patch FeedForwardNet-style MLPs to run row-chunked addmm(out) calls.

    The patch keeps the same float32 operation graph at the MLP level:
    Linear -> GELU(tanh) -> Linear. It avoids returning pooled outputs, because
    those tensors escape into the transformer block. Only the hidden chunk
    buffer and optional transposed weights are cached on the module.
    """
    if not enabled:
        return {"enabled": False}

    import re
    import torch
    import torch.nn as nn
    import types

    include = re.compile(include_regex) if include_regex else None
    exclude = re.compile(exclude_regex) if exclude_regex else None
    patched: list[str] = []
    skipped: list[str] = []
    selected_dims: dict[str, str] = {}
    large_chunk = max(1, int(large_chunk_rows))
    small_chunk = max(1, int(small_chunk_rows))
    large_threshold = max(1, int(large_row_threshold))
    min_m = max(1, int(min_rows))

    def matches(name: str) -> bool:
        if include is not None and include.search(name) is None:
            return False
        if exclude is not None and exclude.search(name) is not None:
            return False
        return True

    def choose_chunk(rows: int) -> int:
        if rows >= large_threshold:
            return large_chunk
        return small_chunk

    def cached_weight_t(module, attr_name: str):
        cached = getattr(module, attr_name, None)
        if (
            cached is None
            or cached.device != module.weight.device
            or cached.dtype != module.weight.dtype
            or tuple(cached.shape) != (int(module.in_features), int(module.out_features))
        ):
            cached = module.weight.detach().t().contiguous()
            setattr(module, attr_name, cached)
        return cached

    def hidden_buffer(module, rows: int, hidden: int, *, device, dtype):
        cached = getattr(module, "_chunked_mlp_hidden_buffer", None)
        if (
            cached is None
            or cached.device != device
            or cached.dtype != dtype
            or int(cached.shape[0]) < int(rows)
            or int(cached.shape[1]) != int(hidden)
        ):
            cached = torch.empty((int(rows), int(hidden)), device=device, dtype=dtype)
            module._chunked_mlp_hidden_buffer = cached
        return cached

    def make_forward(module_name: str):
        def patched_forward(self, x):
            if torch.is_grad_enabled() or x.device.type != "cpu" or x.dtype != torch.float32:
                return self._original_forward_chunked_mlp(x)
            if not x.is_contiguous():
                return self._original_forward_chunked_mlp(x)
            l0 = self.mlp[0]
            act = self.mlp[1]
            l2 = self.mlp[2]
            in_features = int(l0.in_features)
            hidden_features = int(l0.out_features)
            out_features = int(l2.out_features)
            if int(l2.in_features) != hidden_features or int(out_features) != in_features:
                return self._original_forward_chunked_mlp(x)
            rows = int(x.numel() // in_features)
            if rows < min_m or int(x.shape[-1]) != in_features:
                return self._original_forward_chunked_mlp(x)
            x2 = x.reshape(rows, in_features)
            chunk = choose_chunk(rows)
            hidden = hidden_buffer(self, min(chunk, rows), hidden_features, device=x.device, dtype=x.dtype)
            out = torch.empty((rows, out_features), device=x.device, dtype=x.dtype)
            if cache_weight_t:
                w0_t = cached_weight_t(l0, "_chunked_mlp_weight_t_contiguous")
                w2_t = cached_weight_t(l2, "_chunked_mlp_weight_t_contiguous")
            else:
                w0_t = l0.weight.t()
                w2_t = l2.weight.t()
            approximate = getattr(act, "approximate", "none")
            for start in range(0, rows, chunk):
                end = min(start + chunk, rows)
                count = end - start
                h_view = hidden[:count]
                torch.addmm(l0.bias, x2[start:end], w0_t, out=h_view)
                torch.ops.aten.gelu.out(h_view, approximate=approximate, out=h_view)
                torch.addmm(l2.bias, h_view, w2_t, out=out[start:end])
            return out.view(*x.shape[:-1], out_features)

        patched_forward.__name__ = f"chunked_mlp_forward_{module_name.replace(chr(46), chr(95))}"
        return patched_forward

    for name, module in flow_model.named_modules():
        if not matches(name):
            continue
        if not hasattr(module, "mlp") or not isinstance(module.mlp, nn.Sequential) or len(module.mlp) != 3:
            continue
        l0, act, l2 = module.mlp[0], module.mlp[1], module.mlp[2]
        if not isinstance(l0, nn.Linear) or not isinstance(l2, nn.Linear):
            skipped.append(name)
            continue
        if not isinstance(act, nn.GELU) or getattr(act, "approximate", None) != "tanh":
            skipped.append(name)
            continue
        if int(l0.in_features) == 1024 and int(l0.out_features) == 4096 and int(l2.in_features) == 4096 and int(l2.out_features) == 1024:
            if cache_weight_t:
                cached_weight_t(l0, "_chunked_mlp_weight_t_contiguous")
                cached_weight_t(l2, "_chunked_mlp_weight_t_contiguous")
            if not hasattr(module, "_original_forward_chunked_mlp"):
                module._original_forward_chunked_mlp = module.forward
            module.forward = types.MethodType(make_forward(name), module)
            patched.append(name)
            selected_dims[name] = "1024x4096x1024"
        else:
            skipped.append(name)

    return {
        "enabled": bool(patched),
        "kind": "chunked_mlp_addmm_out_patch",
        "include_regex": include_regex,
        "exclude_regex": exclude_regex,
        "large_chunk_rows": large_chunk,
        "small_chunk_rows": small_chunk,
        "large_row_threshold": large_threshold,
        "min_rows": min_m,
        "cache_weight_t": bool(cache_weight_t),
        "selected_count": len(patched),
        "selected": patched[:100],
        "selected_dims": selected_dims,
        "skipped_count": len(skipped),
        "skipped_sample": skipped[:50],
        "semantics": "CPU float32 candidate. Computes Linear -> GELU(tanh) -> Linear in row chunks using torch.addmm(out=...) and returns a fresh output tensor; hidden chunk buffer only is reused.",
    }

def apply_triposplat_linear_output_buffer_patch(
    flow_model,
    *,
    enabled: bool = False,
    include_regex: str | None = None,
    exclude_regex: str | None = None,
    ring_depth: int = 4,
    min_rows: int = 4096,
) -> dict[str, Any]:
    """Patch selected CPU float32 Linear calls to use addmm(out=...).

    The target is the current-best low-resource CPU path where large Linear
    calls are pure inference. Buffers are shared by output shape rather than by
    module so memory use stays bounded. This is an admission candidate; the
    full-flow quality gate decides whether the buffer lifetime assumptions hold.
    """
    if not enabled:
        return {"enabled": False}

    import re
    import torch
    import torch.nn as nn
    import types
    from collections import defaultdict

    include = re.compile(include_regex) if include_regex else None
    exclude = re.compile(exclude_regex) if exclude_regex else None
    pool: dict[tuple[Any, ...], list[torch.Tensor]] = {}
    pool_cursor: defaultdict[tuple[Any, ...], int] = defaultdict(int)
    patched: list[str] = []
    skipped: list[str] = []
    selected_dims: dict[str, str] = {}
    depth = max(2, int(ring_depth))
    min_m = int(min_rows)

    def matches(name: str) -> bool:
        if include is not None and include.search(name) is None:
            return False
        if exclude is not None and exclude.search(name) is not None:
            return False
        return True

    def mode_for(module: nn.Linear, rows: int) -> str | None:
        if rows < min_m:
            return None
        in_features = int(module.in_features)
        out_features = int(module.out_features)
        if in_features == 1024 and out_features == 4096:
            return "addmm_contiguous_out"
        if in_features == 1024 and out_features == 3072:
            return "addmm_view_out"
        if in_features == 1024 and out_features == 1024:
            return "addmm_contiguous_out"
        return None

    def take_buffer(x2: torch.Tensor, out_features: int, mode: str) -> torch.Tensor:
        key = (
            str(x2.device),
            str(x2.dtype),
            int(x2.shape[0]),
            int(out_features),
            mode,
        )
        buffers = pool.get(key)
        if buffers is None:
            buffers = [torch.empty((int(x2.shape[0]), int(out_features)), device=x2.device, dtype=x2.dtype) for _ in range(depth)]
            pool[key] = buffers
        index = pool_cursor[key] % len(buffers)
        pool_cursor[key] += 1
        return buffers[index]

    def make_forward(module_name: str):
        def patched_forward(self, x):
            if torch.is_grad_enabled() or x.device.type != "cpu" or x.dtype != torch.float32:
                return self._original_forward_linear_output_buffer(x)
            if int(x.shape[-1]) != int(self.in_features) or self.bias is None:
                return self._original_forward_linear_output_buffer(x)
            if not x.is_contiguous():
                return self._original_forward_linear_output_buffer(x)
            rows = int(x.numel() // int(self.in_features))
            mode = mode_for(self, rows)
            if mode is None:
                return self._original_forward_linear_output_buffer(x)
            x2 = x.reshape(rows, int(self.in_features))
            out = take_buffer(x2, int(self.out_features), mode)
            if mode == "addmm_contiguous_out":
                cached = getattr(self, "_linear_output_buffer_weight_t_contiguous", None)
                if cached is None or cached.device != self.weight.device or cached.dtype != self.weight.dtype:
                    cached = self.weight.detach().t().contiguous()
                    self._linear_output_buffer_weight_t_contiguous = cached
                torch.addmm(self.bias, x2, cached, out=out)
            else:
                torch.addmm(self.bias, x2, self.weight.t(), out=out)
            return out.view(*x.shape[:-1], int(self.out_features))

        patched_forward.__name__ = f"linear_output_buffer_forward_{module_name.replace(chr(46), chr(95))}"
        return patched_forward

    for name, module in flow_model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not matches(name):
            skipped.append(name)
            continue
        if int(module.in_features) == 1024 and int(module.out_features) in {1024, 3072, 4096}:
            if not hasattr(module, "_original_forward_linear_output_buffer"):
                module._original_forward_linear_output_buffer = module.forward
            module.forward = types.MethodType(make_forward(name), module)
            patched.append(name)
            selected_dims[name] = f"{int(module.in_features)}x{int(module.out_features)}"
        else:
            skipped.append(name)

    return {
        "enabled": bool(patched),
        "kind": "linear_output_buffer_addmm_out_patch",
        "include_regex": include_regex,
        "exclude_regex": exclude_regex,
        "ring_depth": depth,
        "min_rows": min_m,
        "selected_count": len(patched),
        "selected": patched[:100],
        "selected_dims": selected_dims,
        "skipped_count": len(skipped),
        "skipped_sample": skipped[:50],
        "modes": {
            "1024x4096": "addmm_contiguous_out",
            "1024x3072": "addmm_view_out",
            "1024x1024": "addmm_contiguous_out",
        },
        "semantics": "Inference-only CPU float32 candidate. Uses torch.addmm(out=shared_ring_buffer) for current-best large Linear shapes; full-flow quality gate validates output lifetime safety.",
    }


def apply_triposplat_numpy_linear_patch(
    flow_model,
    *,
    enabled: bool = False,
    include_regex: str | None = None,
    exclude_regex: str | None = None,
    min_rows: int = 4096,
    patch_mlp: bool = True,
) -> dict[str, Any]:
    """Patch selected CPU float32 Linear calls through NumPy BLAS.

    This is an admission candidate for fixed TripoSplat CPU shapes. It is not
    bit-exact against torch addmm because BLAS accumulation order differs, so a
    full-flow quality gate is required before adoption.
    """
    if not enabled:
        return {"enabled": False}

    import numpy as np
    import torch
    import torch.nn as nn
    import types

    include = re.compile(include_regex) if include_regex else None
    exclude = re.compile(exclude_regex) if exclude_regex else None
    patched: list[str] = []
    skipped: list[str] = []
    selected_dims: dict[str, str] = {}
    min_m = int(min_rows)

    def matches(name: str) -> bool:
        if include is not None and include.search(name) is None:
            return False
        if exclude is not None and exclude.search(name) is not None:
            return False
        return True

    def layout_for(module_name: str, module: nn.Linear, rows: int) -> str | None:
        if rows < min_m:
            return None
        in_features = int(module.in_features)
        out_features = int(module.out_features)
        if in_features == 1024 and out_features == 3072:
            return "f_contiguous_weight_t"
        if in_features == 1024 and out_features == 1024:
            return "c_contiguous_weight_t"
        if patch_mlp and in_features == 1024 and out_features == 4096:
            return "c_contiguous_weight_t"
        if patch_mlp and in_features == 4096 and out_features == 1024:
            return "c_contiguous_weight_t"
        return None

    def cached_weight_t(module: nn.Linear, layout: str):
        attr = "_numpy_linear_weight_t_f" if layout == "f_contiguous_weight_t" else "_numpy_linear_weight_t_c"
        cached = getattr(module, attr, None)
        weight_version = getattr(module.weight, "_version", None)
        cached_version = getattr(module, attr + "_version", None)
        if cached is None or cached_version != weight_version:
            wt = module.weight.detach().t().contiguous().numpy()
            if layout == "f_contiguous_weight_t":
                wt = np.asfortranarray(wt)
            setattr(module, attr, wt)
            setattr(module, attr + "_version", weight_version)
            cached = wt
        return cached

    def make_forward(module_name: str):
        def patched_forward(self, x):
            if torch.is_grad_enabled() or x.device.type != "cpu" or x.dtype != torch.float32:
                return self._original_forward_numpy_linear(x)
            if self.bias is None or int(x.shape[-1]) != int(self.in_features):
                return self._original_forward_numpy_linear(x)
            if not x.is_contiguous():
                return self._original_forward_numpy_linear(x)
            rows = int(x.numel() // int(self.in_features))
            layout = layout_for(module_name, self, rows)
            if layout is None:
                return self._original_forward_numpy_linear(x)
            x2 = x.reshape(rows, int(self.in_features))
            y = x2.detach().numpy() @ cached_weight_t(self, layout)
            y += self.bias.detach().numpy()
            out = torch.from_numpy(y)
            return out.view(*x.shape[:-1], int(self.out_features))

        patched_forward.__name__ = f"numpy_linear_forward_{module_name.replace(chr(46), chr(95))}"
        return patched_forward

    for name, module in flow_model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not matches(name):
            skipped.append(name)
            continue
        dims = (int(module.in_features), int(module.out_features))
        supported = dims in {(1024, 3072), (1024, 1024)}
        if patch_mlp:
            supported = supported or dims in {(1024, 4096), (4096, 1024)}
        if supported:
            if not hasattr(module, "_original_forward_numpy_linear"):
                module._original_forward_numpy_linear = module.forward
            module.forward = types.MethodType(make_forward(name), module)
            patched.append(name)
            selected_dims[name] = f"{dims[0]}x{dims[1]}"
        else:
            skipped.append(name)

    return {
        "enabled": bool(patched),
        "kind": "numpy_blas_linear_patch",
        "include_regex": include_regex,
        "exclude_regex": exclude_regex,
        "min_rows": min_m,
        "patch_mlp": bool(patch_mlp),
        "selected_count": len(patched),
        "selected": patched[:100],
        "selected_dims": selected_dims,
        "skipped_count": len(skipped),
        "skipped_sample": skipped[:50],
        "layouts": {
            "1024x3072": "f_contiguous_weight_t",
            "1024x1024": "c_contiguous_weight_t",
            "1024x4096": "c_contiguous_weight_t when patch_mlp=true",
            "4096x1024": "c_contiguous_weight_t when patch_mlp=true",
        },
        "semantics": "CPU float32 inference candidate. Computes Y = X W^T + b via NumPy BLAS for large fixed TripoSplat Linear shapes. Not bit-exact; quality gate required.",
    }

def apply_triposplat_u7s8_linear_patch(
    flow_model,
    *,
    enabled: bool = False,
    include_regex: str | None = None,
    exclude_regex: str | None = None,
    min_rows: int = 4096,
    patch_mlp: bool = True,
    library_path: str | None = None,
    threads: int = 2,
) -> dict[str, Any]:
    """Patch selected CPU float32 Linear calls through the u7/s8 AVX2 GEMM."""
    if not enabled:
        return {"enabled": False}

    import ctypes
    import time
    from pathlib import Path

    import torch
    import torch.nn as nn

    include = re.compile(include_regex) if include_regex else None
    exclude = re.compile(exclude_regex) if exclude_regex else None
    patched: list[str] = []
    skipped: list[str] = []
    selected_dims: dict[str, str] = {}
    min_m = int(min_rows)
    thread_count = int(threads)
    lib_path = Path(library_path or "artifacts/backends/libtriposplat_gemm_i8_avx2.so")
    if not lib_path.exists():
        raise FileNotFoundError(f"u7/s8 GEMM library not found: {lib_path}")
    lib = ctypes.CDLL(lib_path.as_posix())
    kernel = lib.triposplat_gemm_i8u7_avx2_blocked
    kernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    kernel.restype = ctypes.c_int

    def matches(name: str) -> bool:
        if include is not None and include.search(name) is None:
            return False
        if exclude is not None and exclude.search(name) is not None:
            return False
        return True

    def supported_dims(module: nn.Linear) -> bool:
        dims = (int(module.in_features), int(module.out_features))
        if dims in {(1024, 3072), (1024, 1024)}:
            return True
        if patch_mlp and dims in {(1024, 4096), (4096, 1024)}:
            return True
        return False

    def cached_quantized_weight(module: nn.Linear):
        version = getattr(module.weight, "_version", None)
        cached_version = getattr(module, "_u7s8_weight_version", None)
        cached = getattr(module, "_u7s8_weight_q", None)
        cached_scale = getattr(module, "_u7s8_weight_scale", None)
        cached_bias = getattr(module, "_u7s8_bias", None)
        if cached is None or cached_scale is None or cached_bias is None or cached_version != version:
            w = module.weight.detach().to(dtype=torch.float32).contiguous()
            max_abs = w.abs().amax(dim=1)
            scale = torch.clamp(max_abs / 127.0, min=1.0e-30).contiguous()
            q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8).contiguous()
            bias = module.bias.detach().to(dtype=torch.float32).contiguous()
            module._u7s8_weight_q = q
            module._u7s8_weight_scale = scale
            module._u7s8_bias = bias
            module._u7s8_weight_version = version
            cached = q
            cached_scale = scale
            cached_bias = bias
        return cached, cached_scale, cached_bias

    def make_forward(module_name: str):
        def patched_forward(self, x):
            if torch.is_grad_enabled() or x.device.type != "cpu" or x.dtype != torch.float32:
                return self._original_forward_u7s8_linear(x)
            if self.bias is None or int(x.shape[-1]) != int(self.in_features):
                return self._original_forward_u7s8_linear(x)
            if not x.is_contiguous():
                return self._original_forward_u7s8_linear(x)
            rows = int(x.numel() // int(self.in_features))
            if rows < min_m:
                return self._original_forward_u7s8_linear(x)
            weight_q, weight_scale, bias = cached_quantized_weight(self)
            x2 = x.reshape(rows, int(self.in_features))
            out = torch.empty((rows, int(self.out_features)), dtype=torch.float32, device=x.device)
            status = int(kernel(
                int(x2.data_ptr()),
                int(weight_q.data_ptr()),
                int(weight_scale.data_ptr()),
                int(bias.data_ptr()),
                int(out.data_ptr()),
                rows,
                int(self.in_features),
                int(self.out_features),
                int(self.in_features),
                int(self.out_features),
                thread_count,
            ))
            if status != 0:
                raise RuntimeError(f"triposplat_gemm_i8u7_avx2_blocked returned status {status} for {module_name}")
            return out.view(*x.shape[:-1], int(self.out_features))

        patched_forward.__name__ = f"u7s8_linear_forward_{module_name.replace(chr(46), chr(95))}"
        return patched_forward

    for name, module in flow_model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not matches(name) or not supported_dims(module):
            skipped.append(name)
            continue
        if not hasattr(module, "_original_forward_u7s8_linear"):
            module._original_forward_u7s8_linear = module.forward
        module.forward = types.MethodType(make_forward(name), module)
        patched.append(name)
        selected_dims[name] = f"{int(module.in_features)}x{int(module.out_features)}"

    return {
        "enabled": bool(patched),
        "kind": "u7s8_avx2_ctypes_linear_patch",
        "library_path": lib_path.as_posix(),
        "symbol": "triposplat_gemm_i8u7_avx2_blocked",
        "include_regex": include_regex,
        "exclude_regex": exclude_regex,
        "min_rows": min_m,
        "patch_mlp": bool(patch_mlp),
        "threads": thread_count,
        "selected_count": len(patched),
        "selected": patched[:100],
        "selected_dims": selected_dims,
        "skipped_count": len(skipped),
        "skipped_sample": skipped[:50],
        "tile": {"MR": 2, "NR": 4},
        "semantics": "Approximate CPU inference candidate. Uses external AVX2 u7/s8 GEMM for selected large nn.Linear.forward calls; F.linear-only custom paths are not patched by this hook.",
    }



def apply_triposplat_mixed_pc8_linear_patch(
    flow_model,
    *,
    enabled: bool = False,
    include_regex: str | None = None,
    exclude_regex: str | None = None,
    group_mask_npz: str | None = None,
    group_size: int = 32,
    library_path: str | None = None,
    kernel_variant: str = "n8_oneq_mask",
    min_rows: int = 1,
    threads: int = 2,
    step_context: dict[str, Any] | None = None,
    fallback_steps: str | None = None,
    residual_correction_rank: int = 0,
    residual_correction_mode: str = "none",
    residual_correction_calibration_npz: str | None = None,
    residual_correction_factors_npz: str | None = None,
    residual_correction_save_factors_npz: str | None = None,
    residual_correction_gemm_library: str | None = None,
    residual_correction_gemm_symbol: str = "triposplat_gemm_f32_avx2",
    residual_correction_fused_library: str | None = None,
    residual_correction_fused_symbol: str = "triposplat_lowrank_residual_f32_avx2_add",
) -> dict[str, Any]:
    """Patch selected CPU float32 Linear calls through packed mixed pc8 GEMM.

    This is the runtime counterpart of the dequantized-weight quality probe:
    pc8 groups and float-kept groups are stored separately and selected by the
    same row x input-group keep mask. It is intentionally CPU-only.
    """
    if not enabled:
        return {"enabled": False}
    if not group_mask_npz:
        raise ValueError("mixed pc8 Linear requires group_mask_npz")
    if int(group_size) <= 0:
        raise ValueError(f"group_size must be > 0, got {group_size}")
    if int(residual_correction_rank) < 0:
        raise ValueError(f"residual_correction_rank must be >= 0, got {residual_correction_rank}")
    if residual_correction_mode not in {"none", "svd", "activation_svd"}:
        raise ValueError(f"unsupported mixed pc8 residual_correction_mode: {residual_correction_mode}")
    if int(residual_correction_rank) > 0 and residual_correction_mode == "none":
        raise ValueError("residual_correction_rank > 0 requires residual_correction_mode=svd or activation_svd")
    if residual_correction_mode == "activation_svd" and not (
        residual_correction_calibration_npz or residual_correction_factors_npz
    ):
        raise ValueError("activation_svd residual correction requires residual_correction_calibration_npz or residual_correction_factors_npz")
    if kernel_variant not in {"v0_scalar_n", "n4_blocked", "n8_blocked", "n8_oneq_mask", "n8_oneq_mask_mr2", "n8_twq_mask", "n8_tile_allkeep_mr2", "n8_tile_hot_mr2", "n8_tile_hot_replace_mr2", "n8_tile_hot_replace_mr4", "n8_tile_hot_replace_mr4_mblock", "n8_tile_hot_replace_mr4_cold_mr4", "n8_tile_hot_replace_mr8", "n8_tile_hot_replace_mr8_runs", "n8_oneq_schedule"}:
        raise ValueError(f"unsupported mixed pc8 kernel_variant: {kernel_variant}")

    import ctypes
    import time
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn as nn

    include = re.compile(include_regex) if include_regex else None
    exclude = re.compile(exclude_regex) if exclude_regex else None
    group = int(group_size)
    min_m = int(min_rows)
    thread_count = int(threads)
    lib_path = Path(library_path or "artifacts/backends/libtriposplat_mixed_pc8_avx2.so")
    if not lib_path.exists():
        raise FileNotFoundError(f"mixed pc8 GEMM library not found: {lib_path}")
    residual_gemm_path = Path(residual_correction_gemm_library) if residual_correction_gemm_library else None
    if residual_gemm_path is not None and not residual_gemm_path.exists():
        raise FileNotFoundError(f"mixed pc8 residual correction GEMM library not found: {residual_gemm_path}")
    residual_fused_path = Path(residual_correction_fused_library) if residual_correction_fused_library else None
    if residual_fused_path is not None and not residual_fused_path.exists():
        raise FileNotFoundError(f"mixed pc8 residual correction fused library not found: {residual_fused_path}")

    def parse_step_set(raw: str | None) -> set[int]:
        steps: set[int] = set()
        for part in (raw or "").split(","):
            text = part.strip()
            if not text:
                continue
            if "-" in text:
                start_text, end_text = text.split("-", 1)
                start = int(start_text.strip())
                end = int(end_text.strip())
                lo, hi = (start, end) if start <= end else (end, start)
                steps.update(range(lo, hi + 1))
            else:
                steps.add(int(text))
        return steps

    fallback_step_set = parse_step_set(fallback_steps)

    def current_step() -> int:
        return int((step_context or {}).get("step", 0))

    def load_group_keep_mask_npz(path: str, *, expected_group_size: int) -> tuple[dict[str, Any], dict[str, Any]]:
        calibration = np.load(path, allow_pickle=False)
        calibration_group_size = int(calibration["group_size"].reshape(-1)[0])
        if calibration_group_size != int(expected_group_size):
            raise ValueError(
                f"group keep mask group_size mismatch: file={calibration_group_size} requested={expected_group_size}"
            )
        names = [str(value) for value in calibration["module_names"].tolist()]
        by_name: dict[str, Any] = {}
        counts: dict[str, int] = {}
        shapes: dict[str, list[int]] = {}
        if "keep_masks" in calibration.files or "masks" in calibration.files:
            key = "keep_masks" if "keep_masks" in calibration.files else "masks"
            masks = calibration[key]
            if masks.ndim != 3:
                raise ValueError(f"group keep masks must be [module, out, group], got shape={masks.shape}")
            for idx, name in enumerate(names):
                mask = masks[idx].astype(np.bool_)
                by_name[name] = mask
                counts[name] = int(mask.sum())
                shapes[name] = [int(v) for v in mask.shape]
            mask_format = "stacked"
        elif "mask_keys" in calibration.files:
            key = "mask_keys"
            mask_keys = [str(value) for value in calibration["mask_keys"].tolist()]
            if len(mask_keys) != len(names):
                raise ValueError(f"mask_keys length mismatch: names={len(names)} mask_keys={len(mask_keys)}")
            for name, mask_key in zip(names, mask_keys):
                if mask_key not in calibration.files:
                    raise ValueError(f"group keep mask key missing for {name}: {mask_key}")
                mask = calibration[mask_key]
                if mask.ndim != 2:
                    raise ValueError(f"group keep mask for {name} must be [out, group], got shape={mask.shape}")
                mask = mask.astype(np.bool_)
                by_name[name] = mask
                counts[name] = int(mask.sum())
                shapes[name] = [int(v) for v in mask.shape]
            mask_format = "per_module_arrays"
        else:
            raise ValueError("group keep mask NPZ requires keep_masks/masks or mask_keys")
        meta: dict[str, Any] = {
            "enabled": True,
            "path": str(path),
            "module_count": len(names),
            "module_names": names[:32],
            "group_size": calibration_group_size,
            "mask_key": key,
            "format": mask_format,
            "counts": counts,
            "shapes": shapes,
        }
        if "metadata_json" in calibration.files:
            try:
                meta["metadata"] = json.loads(str(calibration["metadata_json"].reshape(-1)[0]))
            except Exception as exc:
                meta["metadata_parse_error"] = repr(exc)
        return by_name, meta

    group_mask_by_name, group_mask_meta = load_group_keep_mask_npz(group_mask_npz, expected_group_size=group)

    residual_activation_cov_by_name: dict[str, Any] = {}
    residual_activation_count_by_name: dict[str, int] = {}
    residual_activation_cov_meta: dict[str, Any] = {"enabled": False, "path": residual_correction_calibration_npz}

    def load_activation_covariance_npz(path: str) -> tuple[dict[str, Any], dict[str, int], dict[str, Any]]:
        calibration = np.load(path, allow_pickle=False)
        calibration_group_size = int(calibration["group_size"].reshape(-1)[0])
        names = [str(value) for value in calibration["module_names"].tolist()]
        counts = calibration["counts"]
        covariance_keys = [str(value) for value in calibration["covariance_keys"].tolist()] if "covariance_keys" in calibration.files else []
        if covariance_keys:
            if len(covariance_keys) != len(names):
                raise ValueError(f"covariance_keys length mismatch: names={len(names)} keys={len(covariance_keys)}")
            covariances_by_module = []
            for name, covariance_key in zip(names, covariance_keys):
                if covariance_key not in calibration.files:
                    raise ValueError(f"covariance key missing for {name}: {covariance_key}")
                covariances_by_module.append(calibration[covariance_key])
            covariance_format = "per_module_arrays"
        else:
            covariances = calibration["covariances"]
            covariances_by_module = [covariances[idx] for idx in range(len(names))]
            covariance_format = "stacked"
        by_name: dict[str, Any] = {}
        count_by_name: dict[str, int] = {}
        count_meta: dict[str, int] = {}
        shapes: dict[str, list[int]] = {}
        for idx, name in enumerate(names):
            by_name[name] = covariances_by_module[idx]
            count_value = int(np.sum(counts[idx])) if getattr(counts[idx], "ndim", 0) else int(counts[idx])
            count_by_name[name] = count_value
            count_meta[name] = count_value
            shapes[name] = [int(v) for v in covariances_by_module[idx].shape]
        meta = {
            "enabled": True,
            "path": str(path),
            "module_count": len(names),
            "module_names": names[:32],
            "group_size": calibration_group_size,
            "counts": count_meta,
            "format": covariance_format,
            "covariance_keys": covariance_keys[:32],
            "shapes": shapes,
        }
        return by_name, count_by_name, meta

    if residual_correction_calibration_npz:
        residual_activation_cov_by_name, residual_activation_count_by_name, residual_activation_cov_meta = load_activation_covariance_npz(
            residual_correction_calibration_npz,
        )

    residual_factor_by_name: dict[str, dict[str, Any]] = {}
    residual_factor_npz_meta: dict[str, Any] = {"enabled": False, "path": residual_correction_factors_npz}

    def load_residual_factor_npz(path: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        archive = np.load(path, allow_pickle=False)
        names = [str(value) for value in archive["module_names"].tolist()]
        by_name: dict[str, dict[str, Any]] = {}
        shapes: dict[str, dict[str, list[int]]] = {}
        for idx, name in enumerate(names):
            left_key = f"left_{idx}"
            right_key = f"right_{idx}"
            if left_key not in archive.files or right_key not in archive.files:
                raise ValueError(f"residual factor archive missing {left_key}/{right_key} for {name}")
            left = np.ascontiguousarray(archive[left_key].astype(np.float32, copy=False))
            right = np.ascontiguousarray(archive[right_key].astype(np.float32, copy=False))
            if left.ndim != 2 or right.ndim != 2:
                raise ValueError(f"residual factor arrays must be 2D for {name}: left={left.shape} right={right.shape}")
            if int(left.shape[1]) != int(right.shape[0]):
                raise ValueError(f"residual factor rank mismatch for {name}: left={left.shape} right={right.shape}")
            by_name[name] = {"left": left, "right": right}
            shapes[name] = {"left": [int(v) for v in left.shape], "right": [int(v) for v in right.shape]}
        meta: dict[str, Any] = {
            "enabled": True,
            "path": str(path),
            "module_count": len(names),
            "module_names": names[:32],
            "shapes": shapes,
        }
        if "metadata_json" in archive.files:
            try:
                meta["metadata"] = json.loads(str(archive["metadata_json"].reshape(-1)[0]))
            except Exception as exc:
                meta["metadata_parse_error"] = repr(exc)
        return by_name, meta

    if residual_correction_factors_npz:
        residual_factor_by_name, residual_factor_npz_meta = load_residual_factor_npz(residual_correction_factors_npz)

    residual_factors_to_save: dict[str, dict[str, Any]] = {}

    lib = ctypes.CDLL(lib_path.as_posix())
    common_argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    schedule_argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    symbol_by_variant = {
        "v0_scalar_n": "triposplat_gemm_mixed_pc8_f32_avx2",
        "n4_blocked": "triposplat_gemm_mixed_pc8_f32_avx2_n4",
        "n8_blocked": "triposplat_gemm_mixed_pc8_f32_avx2_n8",
        "n8_oneq_mask": "triposplat_gemm_mixed_pc8_f32_avx2_n8_oneq_mask",
        "n8_oneq_mask_mr2": "triposplat_gemm_mixed_pc8_f32_avx2_n8_oneq_mask_mr2",
        "n8_twq_mask": "triposplat_gemm_mixed_pc8_f32_avx2_n8_twq_mask",
        "n8_tile_allkeep_mr2": "triposplat_gemm_mixed_pc8_f32_avx2_n8_tile_allkeep_mr2",
        "n8_tile_hot_mr2": "triposplat_gemm_mixed_pc8_f32_avx2_n8_tile_hot_mr2",
        "n8_tile_hot_replace_mr2": "triposplat_gemm_mixed_pc8_f32_avx2_n8_tile_hot_mr2",
        "n8_tile_hot_replace_mr4": "triposplat_gemm_mixed_pc8_f32_avx2_n8_tile_hot_mr4",
        "n8_tile_hot_replace_mr4_mblock": "triposplat_gemm_mixed_pc8_f32_avx2_n8_tile_hot_mr4_mblock",
        "n8_tile_hot_replace_mr4_cold_mr4": "triposplat_gemm_mixed_pc8_f32_avx2_n8_tile_hot_mr4_cold_mr4",
        "n8_tile_hot_replace_mr8": "triposplat_gemm_mixed_pc8_f32_avx2_n8_tile_hot_mr8",
        "n8_tile_hot_replace_mr8_runs": "triposplat_gemm_mixed_pc8_f32_avx2_n8_tile_hot_mr8_runs",
        "n8_oneq_schedule": "triposplat_gemm_mixed_pc8_f32_avx2_n8_oneq_schedule",
    }
    symbol = symbol_by_variant[kernel_variant]
    kernel = getattr(lib, symbol)
    kernel.argtypes = schedule_argtypes if kernel_variant in {"n8_oneq_schedule", "n8_tile_allkeep_mr2", "n8_tile_hot_mr2", "n8_tile_hot_replace_mr2", "n8_tile_hot_replace_mr4", "n8_tile_hot_replace_mr4_mblock", "n8_tile_hot_replace_mr4_cold_mr4", "n8_tile_hot_replace_mr8", "n8_tile_hot_replace_mr8_runs"} else common_argtypes
    kernel.restype = ctypes.c_int
    residual_gemm = None
    residual_gemm_meta: dict[str, Any] = {"enabled": False, "library_path": None, "symbol": residual_correction_gemm_symbol}
    if residual_gemm_path is not None:
        residual_lib = ctypes.CDLL(residual_gemm_path.as_posix())
        residual_gemm = getattr(residual_lib, residual_correction_gemm_symbol)
        residual_gemm.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        residual_gemm.restype = ctypes.c_int
        residual_gemm_meta = {
            "enabled": True,
            "library_path": residual_gemm_path.as_posix(),
            "symbol": residual_correction_gemm_symbol,
            "runtime_equivalent": "two custom f32 GEMMs: x @ right.T, then mid @ left.T",
        }

    def run_residual_gemm(
        x_tensor: torch.Tensor,
        weight_t: torch.Tensor,
        bias_tensor: torch.Tensor,
        out_tensor: torch.Tensor,
    ) -> None:
        if residual_gemm is None:
            raise RuntimeError("residual correction GEMM was not initialized")
        m = int(x_tensor.shape[0])
        k = int(x_tensor.shape[1])
        n = int(out_tensor.shape[1])
        status = int(
            residual_gemm(
                int(x_tensor.data_ptr()),
                int(weight_t.data_ptr()),
                int(bias_tensor.data_ptr()),
                int(out_tensor.data_ptr()),
                m,
                k,
                n,
                k,
                n,
                thread_count,
            )
        )
        if status != 0:
            raise RuntimeError(
                f"{residual_correction_gemm_symbol} returned status {status} "
                f"for residual correction M={m} K={k} N={n}"
            )

    residual_fused = None
    residual_fused_meta: dict[str, Any] = {"enabled": False, "library_path": None, "symbol": residual_correction_fused_symbol}
    if residual_fused_path is not None:
        residual_fused_lib = ctypes.CDLL(residual_fused_path.as_posix())
        residual_fused = getattr(residual_fused_lib, residual_correction_fused_symbol)
        residual_fused.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        residual_fused.restype = ctypes.c_int
        residual_fused_meta = {
            "enabled": True,
            "library_path": residual_fused_path.as_posix(),
            "symbol": residual_correction_fused_symbol,
            "runtime_equivalent": "one custom AVX2 function: mid = x @ right.T; out += mid @ left.T",
        }

    def run_residual_fused(
        x_tensor: torch.Tensor,
        right_t: torch.Tensor,
        left_t: torch.Tensor,
        out_tensor: torch.Tensor,
        mid_tensor: torch.Tensor,
    ) -> None:
        if residual_fused is None:
            raise RuntimeError("residual correction fused kernel was not initialized")
        m = int(x_tensor.shape[0])
        k = int(x_tensor.shape[1])
        rank = int(mid_tensor.shape[1])
        n = int(out_tensor.shape[1])
        status = int(
            residual_fused(
                int(x_tensor.data_ptr()),
                int(right_t.data_ptr()),
                int(left_t.data_ptr()),
                int(out_tensor.data_ptr()),
                int(mid_tensor.data_ptr()),
                m,
                k,
                rank,
                n,
                k,
                n,
                thread_count,
            )
        )
        if status != 0:
            raise RuntimeError(
                f"{residual_correction_fused_symbol} returned status {status} "
                f"for residual correction M={m} K={k} R={rank} N={n}"
            )

    def matches(name: str) -> bool:
        if include is not None and include.search(name) is None:
            return False
        if exclude is not None and exclude.search(name) is not None:
            return False
        return True

    def supported_dims(module: nn.Linear) -> bool:
        return (int(module.in_features), int(module.out_features)) in {
            (1024, 1024),
            (1024, 3072),
            (1024, 4096),
            (4096, 1024),
        }

    def build_lowrank_residual_correction(module_name: str, original: torch.Tensor, mixed_weight: torch.Tensor) -> tuple[dict[str, torch.Tensor] | None, dict[str, Any]]:
        rank_requested = int(residual_correction_rank)
        if rank_requested <= 0 or residual_correction_mode == "none":
            return None, {"enabled": False, "mode": residual_correction_mode, "rank_requested": rank_requested, "rank_effective": 0}
        residual = original - mixed_weight
        max_rank = min(int(residual.shape[0]), int(residual.shape[1]))
        rank = min(rank_requested, max_rank)
        pre_rmse = torch.sqrt(torch.mean((mixed_weight - original) * (mixed_weight - original)))
        weighted_pre_rmse = None
        weighted_post_rmse = None
        covariance_group_size = None
        covariance_count = None
        prepacked = residual_factor_by_name.get(module_name)
        if prepacked is not None:
            left_np = prepacked["left"]
            right_np = prepacked["right"]
            if int(left_np.shape[0]) != int(original.shape[0]) or int(right_np.shape[1]) != int(original.shape[1]):
                raise ValueError(
                    f"residual factor shape mismatch for {module_name}: "
                    f"left={left_np.shape} right={right_np.shape} expected_out_in={tuple(original.shape)}"
                )
            available_rank = min(int(left_np.shape[1]), int(right_np.shape[0]))
            rank = min(rank, available_rank)
            left = torch.from_numpy(left_np[:, :rank]).to(device=original.device, dtype=torch.float32).contiguous()
            right = torch.from_numpy(right_np[:rank]).to(device=original.device, dtype=torch.float32).contiguous()
            correction = left.to(dtype=original.dtype).matmul(right.to(dtype=original.dtype))
            corrected = mixed_weight + correction
            post_diff = corrected - original
            post_rmse = torch.sqrt(torch.mean(post_diff * post_diff))
            params = int(rank * (int(original.shape[0]) + int(original.shape[1]) + 1))
            meta = {
                "enabled": True,
                "mode": residual_correction_mode,
                "rank_requested": int(rank_requested),
                "rank_effective": int(rank),
                "pre_rmse": float(pre_rmse.item()),
                "post_rmse": float(post_rmse.item()),
                "energy_fraction": None,
                "approx_params": int(params),
                "dense_weight_params": int(original.numel()),
                "param_ratio_vs_dense": float(params / original.numel()),
                "runtime_equivalent": "mixed pc8 GEMM output plus (x @ right.T) @ left.T low-rank correction",
                "source": "prepacked_npz",
                "factors_npz": residual_factor_npz_meta,
            }
            if residual_correction_mode == "activation_svd":
                meta.update({
                    "weighted_pre_rmse": None,
                    "weighted_post_rmse": None,
                    "covariance_group_size": None,
                    "covariance_count": None,
                    "calibration": residual_activation_cov_meta,
                })
            return {"left": left, "right": right}, meta
        if residual_correction_mode == "activation_svd":
            calibration_cov = residual_activation_cov_by_name.get(module_name)
            if calibration_cov is None:
                raise ValueError(f"activation residual calibration missing for module: {module_name}")
            activation_cov = torch.from_numpy(calibration_cov).to(device=original.device, dtype=original.dtype)
            covariance_count = int(residual_activation_count_by_name.get(module_name, 0))
            normalize = float(max(covariance_count, 1))
            covariance_group_size = int(activation_cov.shape[-1])
            weighted_chunks = []
            invsqrt_chunks = []
            widths = []
            for group_idx, start in enumerate(range(0, int(residual.shape[1]), covariance_group_size)):
                end = min(start + covariance_group_size, int(residual.shape[1]))
                width = end - start
                cov = activation_cov[group_idx, :width, :width] / normalize
                cov = (cov + cov.transpose(0, 1)) * 0.5
                evals, evecs = torch.linalg.eigh(cov)
                max_eval = torch.clamp(evals.max(), min=1.0e-30)
                evals = torch.clamp(evals, min=max_eval * 1.0e-6)
                sqrt_cov = (evecs * torch.sqrt(evals)[None, :]) @ evecs.transpose(0, 1)
                invsqrt_cov = (evecs * torch.rsqrt(evals)[None, :]) @ evecs.transpose(0, 1)
                weighted_chunks.append(residual[:, start:end] @ sqrt_cov)
                invsqrt_chunks.append(invsqrt_cov)
                widths.append(width)
            weighted_residual = torch.cat(weighted_chunks, dim=1)
            u, s, vh = torch.linalg.svd(weighted_residual, full_matrices=False)
            weighted_correction = (u[:, :rank] * s[:rank]) @ vh[:rank]
            correction_chunks = []
            offset = 0
            for width, invsqrt_cov in zip(widths, invsqrt_chunks):
                correction_chunks.append(weighted_correction[:, offset:offset + width] @ invsqrt_cov)
                offset += width
            correction = torch.cat(correction_chunks, dim=1)
            weighted_pre_rmse = torch.sqrt(torch.mean(weighted_residual * weighted_residual))
            weighted_post = weighted_residual - weighted_correction
            weighted_post_rmse = torch.sqrt(torch.mean(weighted_post * weighted_post))
        else:
            u, s, vh = torch.linalg.svd(residual, full_matrices=False)
            correction = (u[:, :rank] * s[:rank]) @ vh[:rank]
        corrected = mixed_weight + correction
        post_diff = corrected - original
        post_rmse = torch.sqrt(torch.mean(post_diff * post_diff))
        denom = torch.clamp(torch.sum(s * s), min=1.0e-30)
        energy_fraction = torch.sum(s[:rank] * s[:rank]) / denom
        u2, s2, vh2 = torch.linalg.svd(correction, full_matrices=False)
        left = (u2[:, :rank] * s2[:rank]).to(dtype=torch.float32).contiguous()
        right = vh2[:rank].to(dtype=torch.float32).contiguous()
        params = int(rank * (int(original.shape[0]) + int(original.shape[1]) + 1))
        meta = {
            "enabled": True,
            "mode": residual_correction_mode,
            "rank_requested": int(rank_requested),
            "rank_effective": int(rank),
            "pre_rmse": float(pre_rmse.item()),
            "post_rmse": float(post_rmse.item()),
            "energy_fraction": float(energy_fraction.item()),
            "approx_params": int(params),
            "dense_weight_params": int(original.numel()),
            "param_ratio_vs_dense": float(params / original.numel()),
            "runtime_equivalent": "mixed pc8 GEMM output plus (x @ right.T) @ left.T low-rank correction",
        }
        if residual_correction_mode == "activation_svd":
            meta.update({
                "weighted_pre_rmse": float(weighted_pre_rmse.item()) if weighted_pre_rmse is not None else None,
                "weighted_post_rmse": float(weighted_post_rmse.item()) if weighted_post_rmse is not None else None,
                "covariance_group_size": int(covariance_group_size) if covariance_group_size is not None else None,
                "covariance_count": int(covariance_count) if covariance_count is not None else None,
                "calibration": residual_activation_cov_meta,
            })
        return {"left": left, "right": right}, meta

    def pack_weight(module_name: str, module: nn.Linear) -> tuple[dict[str, Any], dict[str, Any]]:
        mask_np = group_mask_by_name.get(module_name)
        if mask_np is None:
            raise ValueError(f"group keep mask missing for module: {module_name}")
        weight = module.weight.detach().to(dtype=torch.float32).contiguous()
        bias = (
            module.bias.detach().to(dtype=torch.float32).contiguous()
            if module.bias is not None
            else torch.zeros((int(module.out_features),), dtype=torch.float32)
        )
        out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
        group_count = int(math.ceil(in_features / group))
        if tuple(mask_np.shape) != (out_features, group_count):
            raise ValueError(
                f"group keep mask shape mismatch for {module_name}: mask={tuple(mask_np.shape)} "
                f"expected={(out_features, group_count)}"
            )
        if group_count > 255:
            raise ValueError(f"mixed pc8 schedule stores group index as uint8; group_count={group_count} is too large")

        levels = 127.0
        scale = torch.clamp(weight.abs().amax(dim=1) / levels, min=1.0e-30).contiguous()
        q = torch.round(weight / scale[:, None]).clamp(-levels, levels).to(torch.int8).contiguous()
        weight_np = weight.cpu().numpy()
        q_np = q.cpu().numpy()
        group_kind = np.zeros((out_features, group_count), dtype=np.uint8)
        group_offsets = np.full((out_features, group_count), -1, dtype=np.int32)
        quant_chunks: list[Any] = []
        keep_chunks: list[Any] = []
        quant_groups = 0
        keep_groups = 0
        quant_offset = 0
        keep_offset = 0
        for row in range(out_features):
            for group_idx in range(group_count):
                start = group_idx * group
                end = min(start + group, in_features)
                width = end - start
                if bool(mask_np[row, group_idx]):
                    group_kind[row, group_idx] = 1
                    group_offsets[row, group_idx] = keep_offset
                    keep_chunks.append(np.ascontiguousarray(weight_np[row, start:end], dtype=np.float32))
                    keep_offset += width
                    keep_groups += 1
                else:
                    group_offsets[row, group_idx] = quant_offset
                    quant_chunks.append(np.ascontiguousarray(q_np[row, start:end], dtype=np.int8))
                    quant_offset += width
                    quant_groups += 1

        quant_values_np = np.concatenate(quant_chunks).astype(np.int8, copy=False) if quant_chunks else np.zeros((1,), dtype=np.int8)
        keep_values_np = np.concatenate(keep_chunks).astype(np.float32, copy=False) if keep_chunks else np.zeros((1,), dtype=np.float32)
        mixed_weight = weight.clone()
        for row in range(out_features):
            for group_idx in range(group_count):
                if bool(mask_np[row, group_idx]):
                    continue
                start = group_idx * group
                end = min(start + group, in_features)
                mixed_weight[row, start:end] = q[row, start:end].to(dtype=torch.float32) * scale[row]
        residual_factors, residual_meta = build_lowrank_residual_correction(module_name, weight, mixed_weight)
        if residual_factors is not None and (residual_gemm is not None or residual_fused is not None):
            rank = int(residual_factors["right"].shape[0])
            residual_factors = {
                **residual_factors,
                "right_t": residual_factors["right"].transpose(0, 1).contiguous(),
                "left_t": residual_factors["left"].transpose(0, 1).contiguous(),
                "gemm_backend": "ctypes_fused_add" if residual_fused is not None else "ctypes_f32",
            }
            if residual_gemm is not None:
                residual_factors["bias_rank"] = torch.zeros((rank,), dtype=torch.float32)
                residual_factors["bias_out"] = torch.zeros((out_features,), dtype=torch.float32)
            residual_meta = {**residual_meta, "runtime_gemm": residual_gemm_meta, "runtime_fused": residual_fused_meta}
        if residual_factors is not None and residual_correction_save_factors_npz:
            residual_factors_to_save[module_name] = {
                "left": residual_factors["left"].detach().cpu().numpy().astype(np.float32, copy=False),
                "right": residual_factors["right"].detach().cpu().numpy().astype(np.float32, copy=False),
                "meta": residual_meta,
            }
        n8_group_count = int(math.ceil(out_features / 8))
        n8_kind_masks = np.zeros((n8_group_count, group_count), dtype=np.uint8)
        for nb_i in range(n8_group_count):
            for group_idx in range(group_count):
                bits = 0
                for lane in range(8):
                    row = nb_i * 8 + lane
                    if row < out_features and bool(group_kind[row, group_idx]):
                        bits |= 1 << lane
                n8_kind_masks[nb_i, group_idx] = bits

        n8_allkeep_tile_offsets = np.full((n8_group_count, group_count), -1, dtype=np.int32)
        allkeep_tile_chunks: list[Any] = []
        allkeep_tile_offset = 0
        for nb_i in range(n8_group_count):
            nb = nb_i * 8
            if nb + 8 > out_features:
                continue
            for group_idx in range(group_count):
                if int(n8_kind_masks[nb_i, group_idx]) != 0xFF:
                    continue
                start = group_idx * group
                end = min(start + group, in_features)
                tile = np.ascontiguousarray(weight_np[nb:nb + 8, start:end].T, dtype=np.float32)
                n8_allkeep_tile_offsets[nb_i, group_idx] = allkeep_tile_offset
                allkeep_tile_chunks.append(tile.reshape(-1))
                allkeep_tile_offset += int(tile.size)
        n8_allkeep_tile_values_np = (
            np.concatenate(allkeep_tile_chunks).astype(np.float32, copy=False)
            if allkeep_tile_chunks else np.zeros((1,), dtype=np.float32)
        )

        scale_np = scale.detach().float().cpu().numpy().astype(np.float32, copy=False)
        n8_hot_tile_offsets = np.full((n8_group_count, group_count), -1, dtype=np.int32)
        hot_tile_chunks: list[Any] = []
        hot_tile_offset = 0
        for nb_i in range(n8_group_count):
            nb = nb_i * 8
            if nb + 8 > out_features:
                continue
            for group_idx in range(group_count):
                bits = int(n8_kind_masks[nb_i, group_idx])
                if bin(bits).count("1") < 6:
                    continue
                start = group_idx * group
                end = min(start + group, in_features)
                width = end - start
                tile = np.empty((width, 8), dtype=np.float32)
                for lane in range(8):
                    row = nb + lane
                    if bits & (1 << lane):
                        tile[:, lane] = weight_np[row, start:end]
                    else:
                        tile[:, lane] = q_np[row, start:end].astype(np.float32, copy=False) * scale_np[row]
                n8_hot_tile_offsets[nb_i, group_idx] = hot_tile_offset
                hot_tile_chunks.append(np.ascontiguousarray(tile).reshape(-1))
                hot_tile_offset += int(tile.size)
        n8_hot_tile_values_np = (
            np.concatenate(hot_tile_chunks).astype(np.float32, copy=False)
            if hot_tile_chunks else np.zeros((1,), dtype=np.float32)
        )

        group_offsets_hot_replace = np.full((out_features, group_count), -1, dtype=np.int32)
        quant_chunks_hot_replace: list[Any] = []
        keep_chunks_hot_replace: list[Any] = []
        quant_groups_hot_replace = 0
        keep_groups_hot_replace = 0
        quant_offset_hot_replace = 0
        keep_offset_hot_replace = 0
        for row in range(out_features):
            nb_i = row // 8
            full_n8 = (nb_i * 8 + 8) <= out_features
            for group_idx in range(group_count):
                bits = int(n8_kind_masks[nb_i, group_idx]) if full_n8 else 0
                if full_n8 and bin(bits).count("1") >= 6:
                    continue
                start = group_idx * group
                end = min(start + group, in_features)
                width = end - start
                if bool(group_kind[row, group_idx]):
                    group_offsets_hot_replace[row, group_idx] = keep_offset_hot_replace
                    keep_chunks_hot_replace.append(np.ascontiguousarray(weight_np[row, start:end], dtype=np.float32))
                    keep_offset_hot_replace += width
                    keep_groups_hot_replace += 1
                else:
                    group_offsets_hot_replace[row, group_idx] = quant_offset_hot_replace
                    quant_chunks_hot_replace.append(np.ascontiguousarray(q_np[row, start:end], dtype=np.int8))
                    quant_offset_hot_replace += width
                    quant_groups_hot_replace += 1
        quant_values_hot_replace_np = (
            np.concatenate(quant_chunks_hot_replace).astype(np.int8, copy=False)
            if quant_chunks_hot_replace else np.zeros((1,), dtype=np.int8)
        )
        keep_values_hot_replace_np = (
            np.concatenate(keep_chunks_hot_replace).astype(np.float32, copy=False)
            if keep_chunks_hot_replace else np.zeros((1,), dtype=np.float32)
        )

        schedule_groups = np.zeros((n8_group_count, group_count), dtype=np.uint8)
        schedule_offsets = np.zeros((n8_group_count, 12), dtype=np.int32)
        oneq_masks = {0xFE: 2, 0xFD: 3, 0xFB: 4, 0xF7: 5, 0xEF: 6, 0xDF: 7, 0xBF: 8, 0x7F: 9}
        for nb_i in range(n8_group_count):
            buckets = [[] for _ in range(11)]
            for group_idx in range(group_count):
                bits = int(n8_kind_masks[nb_i, group_idx])
                if bits == 0xFF:
                    cat = 0
                elif bits == 0x00:
                    cat = 1
                else:
                    cat = oneq_masks.get(bits, 10)
                buckets[cat].append(group_idx)
            pos = 0
            for cat, bucket in enumerate(buckets):
                schedule_offsets[nb_i, cat] = pos
                if bucket:
                    schedule_groups[nb_i, pos:pos + len(bucket)] = np.asarray(bucket, dtype=np.uint8)
                pos += len(bucket)
            schedule_offsets[nb_i, len(buckets)] = pos

        packed = {
            "group_kind": torch.from_numpy(np.ascontiguousarray(group_kind.reshape(-1))).to(torch.uint8).contiguous(),
            "n8_kind_masks": torch.from_numpy(np.ascontiguousarray(n8_kind_masks.reshape(-1))).to(torch.uint8).contiguous(),
            "n8_schedule_groups": torch.from_numpy(np.ascontiguousarray(schedule_groups.reshape(-1))).to(torch.uint8).contiguous(),
            "n8_schedule_offsets": torch.from_numpy(np.ascontiguousarray(schedule_offsets.reshape(-1))).to(torch.int32).contiguous(),
            "n8_allkeep_tile_offsets": torch.from_numpy(np.ascontiguousarray(n8_allkeep_tile_offsets.reshape(-1))).to(torch.int32).contiguous(),
            "n8_allkeep_tile_values": torch.from_numpy(n8_allkeep_tile_values_np).to(torch.float32).contiguous(),
            "n8_hot_tile_offsets": torch.from_numpy(np.ascontiguousarray(n8_hot_tile_offsets.reshape(-1))).to(torch.int32).contiguous(),
            "n8_hot_tile_values": torch.from_numpy(n8_hot_tile_values_np).to(torch.float32).contiguous(),
            "group_offsets_hot_replace": torch.from_numpy(np.ascontiguousarray(group_offsets_hot_replace.reshape(-1))).to(torch.int32).contiguous(),
            "quant_values_hot_replace": torch.from_numpy(quant_values_hot_replace_np).to(torch.int8).contiguous(),
            "keep_values_hot_replace": torch.from_numpy(keep_values_hot_replace_np).to(torch.float32).contiguous(),
            "group_offsets": torch.from_numpy(np.ascontiguousarray(group_offsets.reshape(-1))).to(torch.int32).contiguous(),
            "quant_values": torch.from_numpy(quant_values_np).to(torch.int8).contiguous(),
            "keep_values": torch.from_numpy(keep_values_np).to(torch.float32).contiguous(),
            "weight_scale": scale,
            "bias": bias,
            "group_count": group_count,
            "residual_correction": residual_factors,
        }
        storage_bytes = {
            "group_kind": int(group_kind.size),
            "n8_kind_masks": int(n8_kind_masks.size),
            "n8_schedule_groups": int(schedule_groups.size),
            "n8_schedule_offsets": int(schedule_offsets.size * 4),
            "group_offsets": int(group_offsets.size * 4),
            "quant_values": int(quant_values_np.size),
            "keep_values": int(keep_values_np.size * 4),
            "weight_scale": int(scale.numel() * 4),
            "n8_allkeep_tile_offsets": int(n8_allkeep_tile_offsets.size * 4),
            "n8_allkeep_tile_values": int(n8_allkeep_tile_values_np.size * 4),
            "n8_hot_tile_offsets": int(n8_hot_tile_offsets.size * 4),
            "n8_hot_tile_values": int(n8_hot_tile_values_np.size * 4),
            "group_offsets_hot_replace": int(group_offsets_hot_replace.size * 4),
            "quant_values_hot_replace": int(quant_values_hot_replace_np.size),
            "keep_values_hot_replace": int(keep_values_hot_replace_np.size * 4),
        }
        base_total = int(storage_bytes["group_kind"] + storage_bytes["group_offsets"] + storage_bytes["quant_values"] + storage_bytes["keep_values"] + storage_bytes["weight_scale"])
        allkeep_tile_total = int(base_total + storage_bytes["n8_allkeep_tile_offsets"] + storage_bytes["n8_allkeep_tile_values"])
        hot_tile_total = int(base_total + storage_bytes["n8_hot_tile_offsets"] + storage_bytes["n8_hot_tile_values"])
        hot_replace_total = int(
            storage_bytes["n8_kind_masks"]
            + storage_bytes["group_offsets_hot_replace"]
            + storage_bytes["quant_values_hot_replace"]
            + storage_bytes["keep_values_hot_replace"]
            + storage_bytes["weight_scale"]
            + storage_bytes["n8_hot_tile_offsets"]
            + storage_bytes["n8_hot_tile_values"]
        )
        n8mask_total = int(base_total - storage_bytes["group_kind"] + storage_bytes["n8_kind_masks"])
        schedule_total = int(base_total - storage_bytes["group_kind"] + storage_bytes["n8_schedule_groups"] + storage_bytes["n8_schedule_offsets"])
        dense_bytes = int(out_features * in_features * 4)
        meta = {
            "shape": [out_features, in_features],
            "group_size": group,
            "group_count": group_count,
            "keep_groups": int(keep_groups),
            "quant_groups": int(quant_groups),
            "keep_ratio": float(keep_groups / max(keep_groups + quant_groups, 1)),
            "storage_bytes": storage_bytes,
            "storage_total_base_layout": base_total,
            "storage_total_if_n8_mask_replaces_group_kind": n8mask_total,
            "storage_total_if_schedule_replaces_group_kind": schedule_total,
            "storage_total_with_allkeep_tiles": allkeep_tile_total,
            "storage_total_with_hot_tiles": hot_tile_total,
            "storage_total_hot_replace": hot_replace_total,
            "dense_float_weight_bytes": dense_bytes,
            "storage_ratio_base_layout": float(base_total / dense_bytes),
            "storage_ratio_if_n8_mask_replaces_group_kind": float(n8mask_total / dense_bytes),
            "storage_ratio_if_schedule_replaces_group_kind": float(schedule_total / dense_bytes),
            "storage_ratio_with_allkeep_tiles": float(allkeep_tile_total / dense_bytes),
            "storage_ratio_with_hot_tiles": float(hot_tile_total / dense_bytes),
            "storage_ratio_hot_replace": float(hot_replace_total / dense_bytes),
            "keep_groups_hot_replace": int(keep_groups_hot_replace),
            "quant_groups_hot_replace": int(quant_groups_hot_replace),
            "residual_correction": residual_meta,
        }
        return packed, meta

    patched: list[str] = []
    skipped: list[str] = []
    selected_dims: dict[str, str] = {}
    per_module: dict[str, dict[str, Any]] = {}
    runtime_stats: dict[str, Any] = {}

    def make_forward(module_name: str, packed: dict[str, Any]):
        stats = {
            "call_count": 0,
            "fallback_count": 0,
            "step_fallback_count": 0,
            "rows_total": 0,
            "last_shape": None,
            "last_rows": 0,
            "contiguous_sec": 0.0,
            "alloc_sec": 0.0,
            "kernel_sec": 0.0,
            "view_sec": 0.0,
            "total_sec": 0.0,
            "last_call_sec": 0.0,
            "last_kernel_sec": 0.0,
            "by_step": {},
        }
        runtime_stats[module_name] = stats

        def step_stats(step: int) -> dict[str, Any]:
            key = str(int(step))
            by_step = stats["by_step"]
            target = by_step.get(key)
            if target is None:
                target = {"call_count": 0, "fallback_count": 0, "rows_total": 0, "kernel_sec": 0.0, "total_sec": 0.0}
                by_step[key] = target
            return target

        def patched_forward(self, x):
            if torch.is_grad_enabled() or x.device.type != "cpu" or x.dtype != torch.float32:
                stats["fallback_count"] += 1
                return self._original_forward_mixed_pc8_linear(x)
            if int(x.shape[-1]) != int(self.in_features):
                stats["fallback_count"] += 1
                return self._original_forward_mixed_pc8_linear(x)
            rows = int(x.numel() // int(self.in_features))
            if rows < min_m:
                stats["fallback_count"] += 1
                return self._original_forward_mixed_pc8_linear(x)
            step = current_step()
            if step in fallback_step_set:
                stats["fallback_count"] += 1
                stats["step_fallback_count"] += 1
                target = step_stats(step)
                target["fallback_count"] += 1
                target["rows_total"] += rows
                return self._original_forward_mixed_pc8_linear(x)
            t0 = time.perf_counter()
            x2 = x.reshape(rows, int(self.in_features)).contiguous()
            t1 = time.perf_counter()
            out = torch.empty((rows, int(self.out_features)), dtype=torch.float32, device=x.device)
            t2 = time.perf_counter()
            t_kernel0 = time.perf_counter()
            if kernel_variant == "n8_oneq_schedule":
                status = int(kernel(
                    int(x2.data_ptr()),
                    int(packed["n8_kind_masks"].data_ptr()),
                    int(packed["n8_schedule_groups"].data_ptr()),
                    int(packed["n8_schedule_offsets"].data_ptr()),
                    int(packed["group_offsets"].data_ptr()),
                    int(packed["quant_values"].data_ptr()),
                    int(packed["keep_values"].data_ptr()),
                    int(packed["weight_scale"].data_ptr()),
                    int(packed["bias"].data_ptr()),
                    int(out.data_ptr()),
                    rows,
                    int(self.in_features),
                    int(self.out_features),
                    group,
                    int(packed["group_count"]),
                    int(self.in_features),
                    int(self.out_features),
                    thread_count,
                ))
            elif kernel_variant in {"n8_tile_allkeep_mr2", "n8_tile_hot_mr2", "n8_tile_hot_replace_mr2", "n8_tile_hot_replace_mr4", "n8_tile_hot_replace_mr4_mblock", "n8_tile_hot_replace_mr4_cold_mr4", "n8_tile_hot_replace_mr8", "n8_tile_hot_replace_mr8_runs"}:
                use_hot = kernel_variant in {"n8_tile_hot_mr2", "n8_tile_hot_replace_mr2", "n8_tile_hot_replace_mr4", "n8_tile_hot_replace_mr4_mblock", "n8_tile_hot_replace_mr4_cold_mr4", "n8_tile_hot_replace_mr8", "n8_tile_hot_replace_mr8_runs"}
                tile_offsets_key = "n8_hot_tile_offsets" if use_hot else "n8_allkeep_tile_offsets"
                tile_values_key = "n8_hot_tile_values" if use_hot else "n8_allkeep_tile_values"
                group_offsets_key = "group_offsets_hot_replace" if kernel_variant in {"n8_tile_hot_replace_mr2", "n8_tile_hot_replace_mr4", "n8_tile_hot_replace_mr4_mblock", "n8_tile_hot_replace_mr4_cold_mr4", "n8_tile_hot_replace_mr8", "n8_tile_hot_replace_mr8_runs"} else "group_offsets"
                quant_values_key = "quant_values_hot_replace" if kernel_variant in {"n8_tile_hot_replace_mr2", "n8_tile_hot_replace_mr4", "n8_tile_hot_replace_mr4_mblock", "n8_tile_hot_replace_mr4_cold_mr4", "n8_tile_hot_replace_mr8", "n8_tile_hot_replace_mr8_runs"} else "quant_values"
                keep_values_key = "keep_values_hot_replace" if kernel_variant in {"n8_tile_hot_replace_mr2", "n8_tile_hot_replace_mr4", "n8_tile_hot_replace_mr4_mblock", "n8_tile_hot_replace_mr4_cold_mr4", "n8_tile_hot_replace_mr8", "n8_tile_hot_replace_mr8_runs"} else "keep_values"
                status = int(kernel(
                    int(x2.data_ptr()),
                    int(packed["n8_kind_masks"].data_ptr()),
                    int(packed[tile_offsets_key].data_ptr()),
                    int(packed[tile_values_key].data_ptr()),
                    int(packed[group_offsets_key].data_ptr()),
                    int(packed[quant_values_key].data_ptr()),
                    int(packed[keep_values_key].data_ptr()),
                    int(packed["weight_scale"].data_ptr()),
                    int(packed["bias"].data_ptr()),
                    int(out.data_ptr()),
                    rows,
                    int(self.in_features),
                    int(self.out_features),
                    group,
                    int(packed["group_count"]),
                    int(self.in_features),
                    int(self.out_features),
                    thread_count,
                ))
            else:
                kind_key = "n8_kind_masks" if kernel_variant in {"n8_blocked", "n8_oneq_mask", "n8_oneq_mask_mr2", "n8_twq_mask"} else "group_kind"
                status = int(kernel(
                    int(x2.data_ptr()),
                    int(packed[kind_key].data_ptr()),
                    int(packed["group_offsets"].data_ptr()),
                    int(packed["quant_values"].data_ptr()),
                    int(packed["keep_values"].data_ptr()),
                    int(packed["weight_scale"].data_ptr()),
                    int(packed["bias"].data_ptr()),
                    int(out.data_ptr()),
                    rows,
                    int(self.in_features),
                    int(self.out_features),
                    group,
                    int(packed["group_count"]),
                    int(self.in_features),
                    int(self.out_features),
                    thread_count,
                ))
            t_kernel1 = time.perf_counter()
            if status != 0:
                raise RuntimeError(f"{symbol} returned status {status} for {module_name}")
            viewed = out.view(*x.shape[:-1], int(self.out_features))
            t3 = time.perf_counter()
            contiguous_sec = t1 - t0
            alloc_sec = t2 - t1
            kernel_sec = t_kernel1 - t_kernel0
            view_sec = t3 - t_kernel1
            total_sec = t3 - t0
            stats["call_count"] += 1
            stats["rows_total"] += rows
            stats["last_shape"] = [int(v) for v in x.shape]
            stats["last_rows"] = rows
            stats["contiguous_sec"] += contiguous_sec
            stats["alloc_sec"] += alloc_sec
            stats["kernel_sec"] += kernel_sec
            stats["view_sec"] += view_sec
            stats["total_sec"] += total_sec
            stats["last_call_sec"] = total_sec
            stats["last_kernel_sec"] = kernel_sec
            target = step_stats(step)
            target["call_count"] += 1
            target["rows_total"] += rows
            target["kernel_sec"] += kernel_sec
            residual = packed.get("residual_correction")
            if residual is not None:
                t_res0 = time.perf_counter()
                if residual_fused is not None and "right_t" in residual and "left_t" in residual:
                    rank = int(residual["right_t"].shape[1])
                    corr_mid = torch.empty((rows, rank), dtype=torch.float32, device=x.device)
                    run_residual_fused(x2, residual["right_t"], residual["left_t"], out, corr_mid)
                    stats["residual_correction_backend"] = "ctypes_fused_add"
                elif residual_gemm is not None and "right_t" in residual and "left_t" in residual:
                    rank = int(residual["right_t"].shape[1])
                    corr_mid = torch.empty((rows, rank), dtype=torch.float32, device=x.device)
                    corr_out = torch.empty((rows, int(self.out_features)), dtype=torch.float32, device=x.device)
                    run_residual_gemm(x2, residual["right_t"], residual["bias_rank"], corr_mid)
                    run_residual_gemm(corr_mid, residual["left_t"], residual["bias_out"], corr_out)
                    out.add_(corr_out)
                    stats["residual_correction_backend"] = "ctypes_f32"
                else:
                    corr_mid = x2.matmul(residual["right"].transpose(0, 1))
                    out.add_(corr_mid.matmul(residual["left"].transpose(0, 1)))
                    stats["residual_correction_backend"] = "torch_matmul"
                residual_sec = time.perf_counter() - t_res0
                stats["residual_correction_sec"] = stats.get("residual_correction_sec", 0.0) + residual_sec
                stats["last_residual_correction_sec"] = residual_sec
                target["residual_correction_sec"] = target.get("residual_correction_sec", 0.0) + residual_sec
                target["residual_correction_backend"] = stats["residual_correction_backend"]
                total_sec += residual_sec
                stats["total_sec"] += residual_sec
            target["total_sec"] += total_sec
            return viewed

        patched_forward.__name__ = f"mixed_pc8_linear_forward_{module_name.replace(chr(46), chr(95))}"
        return patched_forward

    for name, module in flow_model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not matches(name) or not supported_dims(module):
            skipped.append(name)
            continue
        packed, pack_meta = pack_weight(name, module)
        if not hasattr(module, "_original_forward_mixed_pc8_linear"):
            module._original_forward_mixed_pc8_linear = module.forward
        module._mixed_pc8_packed = packed
        module.forward = types.MethodType(make_forward(name, packed), module)
        patched.append(name)
        selected_dims[name] = f"{int(module.in_features)}x{int(module.out_features)}"
        per_module[name] = pack_meta

    residual_factor_save_meta: dict[str, Any] = {"enabled": False, "path": residual_correction_save_factors_npz}
    if residual_correction_save_factors_npz:
        save_path = Path(residual_correction_save_factors_npz)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        names_to_save = [name for name in patched if name in residual_factors_to_save]
        arrays: dict[str, Any] = {"module_names": np.asarray(names_to_save)}
        modules_meta: dict[str, Any] = {}
        total_bytes = 0
        for idx, name in enumerate(names_to_save):
            left = np.ascontiguousarray(residual_factors_to_save[name]["left"], dtype=np.float32)
            right = np.ascontiguousarray(residual_factors_to_save[name]["right"], dtype=np.float32)
            arrays[f"left_{idx}"] = left
            arrays[f"right_{idx}"] = right
            total_bytes += int(left.nbytes + right.nbytes)
            modules_meta[name] = residual_factors_to_save[name]["meta"]
        arrays["metadata_json"] = np.asarray([
            json.dumps({
                "kind": "mixed_pc8_residual_correction_factors",
                "mode": residual_correction_mode,
                "rank_requested": int(residual_correction_rank),
                "group_mask_npz": group_mask_npz,
                "group_size": group,
                "modules": modules_meta,
            }, sort_keys=True)
        ])
        np.savez(save_path, **arrays)
        residual_factor_save_meta = {
            "enabled": True,
            "path": save_path.as_posix(),
            "module_count": len(names_to_save),
            "module_names": names_to_save[:32],
            "factor_bytes": int(total_bytes),
        }

    return {
        "enabled": bool(patched),
        "kind": "mixed_pc8_avx2_ctypes_linear_patch",
        "library_path": lib_path.as_posix(),
        "symbol": symbol,
        "kernel_variant": kernel_variant,
        "include_regex": include_regex,
        "exclude_regex": exclude_regex,
        "min_rows": min_m,
        "threads": thread_count,
        "fallback_steps": sorted(int(v) for v in fallback_step_set),
        "step_context_enabled": bool(step_context is not None and fallback_step_set),
        "group_size": group,
        "mask": group_mask_meta,
        "selected_count": len(patched),
        "selected": patched[:100],
        "selected_dims": selected_dims,
        "per_module": per_module,
        "residual_correction": {
            "enabled": bool(int(residual_correction_rank) > 0 and residual_correction_mode != "none"),
            "mode": residual_correction_mode,
            "rank": int(residual_correction_rank),
            "calibration": residual_activation_cov_meta,
            "factors_npz": residual_factor_npz_meta,
            "saved_factors_npz": residual_factor_save_meta,
            "runtime_gemm": residual_gemm_meta,
            "runtime_fused": residual_fused_meta,
        },
        "runtime_stats": runtime_stats,
        "skipped_count": len(skipped),
        "skipped_sample": skipped[:50],
        "semantics": "Runtime packed-weight probe: selected nn.Linear calls read pc8 quantized groups plus float kept groups directly through an AVX2 ctypes GEMM. Intended to match the dequantized-weight group_keep_mask probe before any quality promotion.",
    }



def apply_triposplat_dequantized_weight_linear_patch(
    flow_model,
    *,
    enabled: bool = False,
    include_regex: str | None = None,
    exclude_regex: str | None = None,
    bits: int = 8,
    mode: str = "linear_per_channel_symmetric",
    percentile: float = 1.0,
    mu: float = 255.0,
    patch_mlp: bool = True,
    group_size: int = 128,
    external_weight_npz: str | None = None,
    mixed_keep_ratio: float = 0.0,
    mixed_keep_count: int = 0,
    mixed_rank: str = "row_rmse_error",
    mixed_group_size: int = 16,
    mixed_group_keep_ratio: float = 0.0,
    mixed_group_keep_count: int = 0,
    mixed_group_rank: str = "group_rmse_error",
    mixed_group_calibration_npz: str | None = None,
    mixed_group_mask_npz: str | None = None,
    mixed_output_residual_npz: str | None = None,
    mixed_output_residual_step_weights: str | None = None,
    residual_correction_rank: int = 0,
    residual_correction_mode: str = "none",
    residual_correction_calibration_npz: str | None = None,
) -> dict[str, Any]:
    """Quantize selected Linear weights once, then dequantize back to float tensors.

    This is a quality decomposition/calibration probe, not a speed path. The
    normal torch Linear kernels still run, but with weights carrying the chosen
    quantization error. Because weights are modified in-place, custom F.linear
    paths that read the same module weights also see the dequantized values.
    """
    if not enabled:
        return {"enabled": False}

    import torch
    import torch.nn as nn

    if bits < 2 or bits > 8:
        raise ValueError(f"dequantized weight bits must be in [2, 8], got {bits}")
    if mode not in {"linear_per_channel_symmetric", "linear_per_tensor_symmetric", "linear_per_channel_group_symmetric", "mulaw_per_channel_symmetric"}:
        raise ValueError(f"unsupported dequantized weight mode: {mode}")
    if not (0.0 < float(percentile) <= 1.0):
        raise ValueError(f"percentile must be in (0, 1], got {percentile}")
    if float(mu) <= 0.0:
        raise ValueError(f"mu must be > 0, got {mu}")
    if int(group_size) <= 0:
        raise ValueError(f"group_size must be > 0, got {group_size}")
    if not (0.0 <= float(mixed_keep_ratio) <= 1.0):
        raise ValueError(f"mixed_keep_ratio must be in [0, 1], got {mixed_keep_ratio}")
    if int(mixed_keep_count) < 0:
        raise ValueError(f"mixed_keep_count must be >= 0, got {mixed_keep_count}")
    if mixed_rank not in {"row_rmse_error", "row_max_abs_error", "row_weight_norm", "row_output_residual_error"}:
        raise ValueError(f"unsupported mixed_rank: {mixed_rank}")
    if mixed_rank == "row_output_residual_error" and not mixed_output_residual_npz:
        raise ValueError("row_output_residual_error requires mixed_output_residual_npz")
    if int(mixed_group_size) <= 0:
        raise ValueError(f"mixed_group_size must be > 0, got {mixed_group_size}")
    if not (0.0 <= float(mixed_group_keep_ratio) <= 1.0):
        raise ValueError(f"mixed_group_keep_ratio must be in [0, 1], got {mixed_group_keep_ratio}")
    if int(mixed_group_keep_count) < 0:
        raise ValueError(f"mixed_group_keep_count must be >= 0, got {mixed_group_keep_count}")
    if mixed_group_rank not in {"group_rmse_error", "group_max_abs_error", "group_weight_norm", "group_activation_error", "group_keep_mask"}:
        raise ValueError(f"unsupported mixed_group_rank: {mixed_group_rank}")
    if mixed_group_rank == "group_activation_error" and not mixed_group_calibration_npz:
        raise ValueError("group_activation_error requires mixed_group_calibration_npz")
    if mixed_group_rank == "group_keep_mask" and not mixed_group_mask_npz:
        raise ValueError("group_keep_mask requires mixed_group_mask_npz")
    if int(residual_correction_rank) < 0:
        raise ValueError(f"residual_correction_rank must be >= 0, got {residual_correction_rank}")
    if residual_correction_mode not in {"none", "svd", "activation_svd"}:
        raise ValueError(f"unsupported residual_correction_mode: {residual_correction_mode}")
    if int(residual_correction_rank) > 0 and residual_correction_mode == "none":
        raise ValueError("residual_correction_rank > 0 requires residual_correction_mode=svd or activation_svd")
    if residual_correction_mode == "activation_svd" and not (residual_correction_calibration_npz or mixed_group_calibration_npz):
        raise ValueError("activation_svd requires residual_correction_calibration_npz or mixed_group_calibration_npz")

    include = re.compile(include_regex) if include_regex else None
    exclude = re.compile(exclude_regex) if exclude_regex else None
    levels = float((1 << (int(bits) - 1)) - 1)
    selected: list[str] = []
    skipped: list[str] = []
    selected_dims: dict[str, str] = {}
    per_module: dict[str, dict[str, float | str | list[int]]] = {}
    mixed_total_keep_rows = 0
    mixed_total_keep_groups = 0
    mixed_total_keep_group_elements = 0
    residual_correction_total_rank = 0
    residual_correction_total_params = 0
    activation_cov_by_name: dict[str, Any] = {}
    activation_count_by_name: dict[str, int] = {}
    activation_cov_meta: dict[str, Any] = {"enabled": False, "path": mixed_group_calibration_npz}
    mixed_group_mask_by_name: dict[str, Any] = {}
    mixed_group_mask_meta: dict[str, Any] = {"enabled": False, "path": mixed_group_mask_npz}
    external_weight_by_name: dict[str, Any] = {}
    external_weight_meta: dict[str, Any] = {"enabled": False, "path": external_weight_npz}
    output_residual_score_by_name: dict[str, Any] = {}
    output_residual_meta: dict[str, Any] = {
        "enabled": False,
        "path": mixed_output_residual_npz,
        "step_weights": mixed_output_residual_step_weights,
    }
    residual_activation_cov_by_name: dict[str, Any] = {}
    residual_activation_count_by_name: dict[str, int] = {}
    residual_activation_cov_meta: dict[str, Any] = {"enabled": False, "path": residual_correction_calibration_npz}

    def load_output_residual_scores_npz(path: str, *, step_weights_raw: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        import numpy as np

        calibration = np.load(path, allow_pickle=False)
        names = [str(value) for value in calibration["module_names"].tolist()]
        steps = [int(value) for value in calibration["steps"].tolist()]
        rmse = calibration["output_rmse"]
        out_features = calibration["out_features"]
        if rmse.ndim != 3:
            raise ValueError(f"output residual rmse must be [module, step, row], got shape={rmse.shape}")
        weights = np.ones((len(steps),), dtype=np.float64)
        raw = (step_weights_raw or "").strip()
        if raw:
            if ":" in raw:
                mapping: dict[int, float] = {}
                for part in raw.split(","):
                    if not part.strip():
                        continue
                    step_text, weight_text = part.split(":", 1)
                    mapping[int(step_text.strip())] = float(weight_text.strip())
                weights = np.array([float(mapping.get(step, 0.0)) for step in steps], dtype=np.float64)
            else:
                values = [float(part.strip()) for part in raw.split(",") if part.strip()]
                if len(values) != len(steps):
                    raise ValueError(f"step weight count mismatch: weights={len(values)} steps={len(steps)}")
                weights = np.array(values, dtype=np.float64)
        if float(weights.sum()) <= 0.0:
            raise ValueError("output residual step weights must sum to a positive value")
        weights = weights / weights.sum()
        by_name: dict[str, Any] = {}
        score_preview: dict[str, list[float]] = {}
        top_rows: dict[str, list[int]] = {}
        for idx, name in enumerate(names):
            width = int(out_features[idx])
            row_rmse = rmse[idx, :, :width].astype(np.float64)
            score = np.sqrt(np.sum((row_rmse * row_rmse) * weights[:, None], axis=0)).astype(np.float32)
            by_name[name] = score
            order = np.argsort(-score)[:16]
            top_rows[name] = [int(v) for v in order.tolist()]
            score_preview[name] = [float(score[v]) for v in order[:8].tolist()]
        meta = {
            "enabled": True,
            "path": str(path),
            "module_count": len(names),
            "module_names": names[:32],
            "steps": steps,
            "step_weights": [float(v) for v in weights.tolist()],
            "step_weights_raw": step_weights_raw,
            "top_rows_by_score": top_rows,
            "top_scores_preview": score_preview,
        }
        return by_name, meta

    def load_activation_covariance_npz(path: str, *, expected_group_size: int | None = None) -> tuple[dict[str, Any], dict[str, int], dict[str, Any]]:
        import numpy as np

        calibration = np.load(path, allow_pickle=False)
        calibration_group_size = int(calibration["group_size"].reshape(-1)[0])
        if expected_group_size is not None and calibration_group_size != int(expected_group_size):
            raise ValueError(
                f"calibration group_size mismatch: file={calibration_group_size} requested={expected_group_size}"
            )
        names = [str(value) for value in calibration["module_names"].tolist()]
        counts = calibration["counts"]
        covariance_keys = [str(value) for value in calibration["covariance_keys"].tolist()] if "covariance_keys" in calibration.files else []
        if covariance_keys:
            if len(covariance_keys) != len(names):
                raise ValueError(f"covariance_keys length mismatch: names={len(names)} keys={len(covariance_keys)}")
            covariances_by_module = []
            for name, covariance_key in zip(names, covariance_keys):
                if covariance_key not in calibration.files:
                    raise ValueError(f"covariance key missing for {name}: {covariance_key}")
                covariances_by_module.append(calibration[covariance_key])
            covariance_format = "per_module_arrays"
        else:
            covariances = calibration["covariances"]
            covariances_by_module = [covariances[idx] for idx in range(len(names))]
            covariance_format = "stacked"
        by_name: dict[str, Any] = {}
        count_by_name: dict[str, int] = {}
        count_meta: dict[str, int] = {}
        shapes: dict[str, list[int]] = {}
        for idx, name in enumerate(names):
            by_name[name] = covariances_by_module[idx]
            count_value = int(np.sum(counts[idx])) if getattr(counts[idx], "ndim", 0) else int(counts[idx])
            count_by_name[name] = count_value
            count_meta[name] = count_value
            shapes[name] = [int(v) for v in covariances_by_module[idx].shape]
        meta = {
            "enabled": True,
            "path": str(path),
            "module_count": len(names),
            "module_names": names[:32],
            "group_size": calibration_group_size,
            "counts": count_meta,
            "format": covariance_format,
            "covariance_keys": covariance_keys[:32],
            "shapes": shapes,
        }
        return by_name, count_by_name, meta

    def load_external_dequant_weight_npz(path: str) -> tuple[dict[str, Any], dict[str, Any]]:
        import numpy as np

        data = np.load(path, allow_pickle=False)
        names = [str(value) for value in data["module_names"].tolist()]
        by_name: dict[str, Any] = {}
        shapes: dict[str, list[int]] = {}
        dtypes: dict[str, str] = {}
        keys: dict[str, str] = {}
        if "dequant_weight_keys" in data.files or "weight_keys" in data.files:
            key_name = "dequant_weight_keys" if "dequant_weight_keys" in data.files else "weight_keys"
            weight_keys = [str(value) for value in data[key_name].tolist()]
            if len(weight_keys) != len(names):
                raise ValueError(f"external weight key count mismatch: names={len(names)} keys={len(weight_keys)}")
            for name, weight_key in zip(names, weight_keys):
                if weight_key not in data.files:
                    raise ValueError(f"external dequant weight key missing for {name}: {weight_key}")
                weight = data[weight_key].astype(np.float32, copy=False)
                if weight.ndim != 2:
                    raise ValueError(f"external dequant weight for {name} must be 2D, got shape={weight.shape}")
                by_name[name] = weight
                shapes[name] = [int(v) for v in weight.shape]
                dtypes[name] = str(data[weight_key].dtype)
                keys[name] = weight_key
            weight_format = "per_module_arrays"
        elif "dequant_weights" in data.files or "weights" in data.files:
            key_name = "dequant_weights" if "dequant_weights" in data.files else "weights"
            weights = data[key_name]
            if weights.ndim != 3:
                raise ValueError(f"stacked external dequant weights must be [module,out,in], got shape={weights.shape}")
            if int(weights.shape[0]) != len(names):
                raise ValueError(f"stacked external dequant weights module count mismatch: weights={weights.shape[0]} names={len(names)}")
            for idx, name in enumerate(names):
                weight = weights[idx].astype(np.float32, copy=False)
                by_name[name] = weight
                shapes[name] = [int(v) for v in weight.shape]
                dtypes[name] = str(weights.dtype)
                keys[name] = f"{key_name}[{idx}]"
            weight_format = "stacked"
        else:
            raise ValueError("external dequant weight NPZ requires dequant_weight_keys/weight_keys or dequant_weights/weights")
        meta: dict[str, Any] = {
            "enabled": True,
            "path": str(path),
            "module_count": len(names),
            "module_names": names[:32],
            "format": weight_format,
            "keys": keys,
            "shapes": shapes,
            "dtypes": dtypes,
        }
        if "metadata_json" in data.files:
            try:
                meta["metadata"] = json.loads(str(data["metadata_json"].reshape(-1)[0]))
            except Exception as exc:
                meta["metadata_parse_error"] = repr(exc)
        return by_name, meta


    def load_group_keep_mask_npz(path: str, *, expected_group_size: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        import numpy as np

        calibration = np.load(path, allow_pickle=False)
        calibration_group_size = int(calibration["group_size"].reshape(-1)[0])
        if expected_group_size is not None and calibration_group_size != int(expected_group_size):
            raise ValueError(
                f"group keep mask group_size mismatch: file={calibration_group_size} requested={expected_group_size}"
            )
        names = [str(value) for value in calibration["module_names"].tolist()]
        by_name: dict[str, Any] = {}
        counts: dict[str, int] = {}
        shapes: dict[str, list[int]] = {}
        if "keep_masks" in calibration.files or "masks" in calibration.files:
            key = "keep_masks" if "keep_masks" in calibration.files else "masks"
            masks = calibration[key]
            if masks.ndim != 3:
                raise ValueError(f"group keep masks must be [module, out, group], got shape={masks.shape}")
            for idx, name in enumerate(names):
                mask = masks[idx].astype(np.bool_)
                by_name[name] = mask
                counts[name] = int(mask.sum())
                shapes[name] = [int(v) for v in mask.shape]
            mask_format = "stacked"
        elif "mask_keys" in calibration.files:
            key = "mask_keys"
            mask_keys = [str(value) for value in calibration["mask_keys"].tolist()]
            if len(mask_keys) != len(names):
                raise ValueError(f"mask_keys length mismatch: names={len(names)} mask_keys={len(mask_keys)}")
            for name, mask_key in zip(names, mask_keys):
                if mask_key not in calibration.files:
                    raise ValueError(f"group keep mask key missing for {name}: {mask_key}")
                mask = calibration[mask_key]
                if mask.ndim != 2:
                    raise ValueError(f"group keep mask for {name} must be [out, group], got shape={mask.shape}")
                mask = mask.astype(np.bool_)
                by_name[name] = mask
                counts[name] = int(mask.sum())
                shapes[name] = [int(v) for v in mask.shape]
            mask_format = "per_module_arrays"
        else:
            raise ValueError("group keep mask NPZ requires keep_masks/masks or mask_keys")
        meta = {
            "enabled": True,
            "path": str(path),
            "module_count": len(names),
            "module_names": names[:32],
            "group_size": calibration_group_size,
            "mask_key": key,
            "format": mask_format,
            "counts": counts,
            "shapes": shapes,
        }
        if "metadata_json" in calibration.files:
            try:
                meta["metadata"] = json.loads(str(calibration["metadata_json"].reshape(-1)[0]))
            except Exception as exc:
                meta["metadata_parse_error"] = repr(exc)
        return by_name, meta

    if external_weight_npz:
        external_weight_by_name, external_weight_meta = load_external_dequant_weight_npz(external_weight_npz)
    if mixed_group_calibration_npz:
        activation_cov_by_name, activation_count_by_name, activation_cov_meta = load_activation_covariance_npz(
            mixed_group_calibration_npz,
            expected_group_size=int(mixed_group_size),
        )
    if mixed_group_mask_npz:
        mixed_group_mask_by_name, mixed_group_mask_meta = load_group_keep_mask_npz(
            mixed_group_mask_npz,
            expected_group_size=int(mixed_group_size),
        )
    if mixed_output_residual_npz:
        output_residual_score_by_name, output_residual_meta = load_output_residual_scores_npz(
            mixed_output_residual_npz,
            step_weights_raw=mixed_output_residual_step_weights,
        )
    if residual_correction_calibration_npz:
        residual_activation_cov_by_name, residual_activation_count_by_name, residual_activation_cov_meta = load_activation_covariance_npz(
            residual_correction_calibration_npz,
            expected_group_size=None,
        )
    elif mixed_group_calibration_npz:
        residual_activation_cov_by_name = activation_cov_by_name
        residual_activation_count_by_name = activation_count_by_name
        residual_activation_cov_meta = activation_cov_meta

    def matches(name: str) -> bool:
        if include is not None and include.search(name) is None:
            return False
        if exclude is not None and exclude.search(name) is not None:
            return False
        return True

    def supported_dims(module: nn.Linear) -> bool:
        dims = (int(module.in_features), int(module.out_features))
        if dims in {(1024, 3072), (1024, 1024)}:
            return True
        if patch_mlp and dims in {(1024, 4096), (4096, 1024)}:
            return True
        return False

    def row_amax(w: torch.Tensor) -> torch.Tensor:
        abs_w = w.abs()
        if float(percentile) >= 1.0:
            return abs_w.amax(dim=1)
        return torch.quantile(abs_w.float(), float(percentile), dim=1)

    def group_quant_dequant(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        group = int(group_size)
        chunks = []
        scales = []
        for start in range(0, int(w.shape[1]), group):
            part = w[:, start:start + group]
            abs_part = part.abs()
            if float(percentile) >= 1.0:
                scale_abs = abs_part.amax(dim=1)
            else:
                scale_abs = torch.quantile(abs_part.float(), float(percentile), dim=1)
            scale = torch.clamp(scale_abs / levels, min=1.0e-30)
            q = torch.round(part / scale[:, None]).clamp(-levels, levels)
            chunks.append(q * scale[:, None])
            scales.append(scale)
        return torch.cat(chunks, dim=1), torch.stack(scales, dim=1)

    def quant_dequant(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        wf = w.detach().to(dtype=torch.float32)
        if mode == "linear_per_tensor_symmetric":
            scale = torch.clamp(wf.abs().amax() / levels, min=1.0e-30)
            q = torch.round(wf / scale).clamp(-levels, levels)
            return q * scale, scale.reshape(1)
        scale_abs = torch.clamp(row_amax(wf), min=1.0e-30)
        if mode == "linear_per_channel_symmetric":
            scale = scale_abs / levels
            q = torch.round(wf / scale[:, None]).clamp(-levels, levels)
            return q * scale[:, None], scale
        if mode == "linear_per_channel_group_symmetric":
            return group_quant_dequant(wf)
        norm = torch.clamp(wf.abs() / scale_abs[:, None], max=1.0)
        log_mu = math.log1p(float(mu))
        encoded = torch.sign(wf) * (torch.log1p(float(mu) * norm) / log_mu)
        q = torch.round(encoded * levels).clamp(-levels, levels)
        decoded_norm = torch.expm1((q.abs() / levels) * log_mu) / float(mu)
        return torch.sign(q) * decoded_norm * scale_abs[:, None], scale_abs

    def keep_float_rows(module_name: str, original: torch.Tensor, dequant: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        keep_count = int(mixed_keep_count)
        keep_ratio = float(mixed_keep_ratio)
        out_features = int(original.shape[0])
        if keep_ratio > 0.0:
            keep_count = max(keep_count, int(math.ceil(out_features * keep_ratio)))
        keep_count = min(max(keep_count, 0), out_features)
        if keep_count <= 0:
            return dequant, {
                "enabled": False,
                "keep_count": 0,
                "keep_ratio_effective": 0.0,
                "rank": mixed_rank,
            }
        diff = dequant - original
        if mixed_rank == "row_max_abs_error":
            scores = diff.abs().amax(dim=1)
        elif mixed_rank == "row_weight_norm":
            scores = torch.sqrt(torch.mean(original * original, dim=1))
        elif mixed_rank == "row_output_residual_error":
            score_np = output_residual_score_by_name.get(module_name)
            if score_np is None:
                raise ValueError(f"output residual calibration missing for module: {module_name}")
            scores = torch.from_numpy(score_np).to(device=original.device, dtype=original.dtype)
            if int(scores.numel()) != out_features:
                raise ValueError(f"output residual row count mismatch for {module_name}: scores={int(scores.numel())} out={out_features}")
        else:
            scores = torch.sqrt(torch.mean(diff * diff, dim=1))
        keep_idx = torch.topk(scores, k=keep_count, largest=True, sorted=False).indices.sort().values
        mixed = dequant.clone()
        mixed[keep_idx] = original[keep_idx]
        kept_scores = scores.index_select(0, keep_idx)
        return mixed, {
            "enabled": True,
            "keep_count": int(keep_count),
            "keep_ratio_effective": float(keep_count / out_features),
            "rank": mixed_rank,
            "score_min_kept": float(kept_scores.min().item()),
            "score_max_kept": float(kept_scores.max().item()),
            "kept_rows_sample": [int(v) for v in keep_idx[:64].detach().cpu().tolist()],
        }

    def keep_float_input_groups(module_name: str, original: torch.Tensor, dequant: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        group = int(mixed_group_size)
        out_features = int(original.shape[0])
        in_features = int(original.shape[1])
        group_count_per_row = int(math.ceil(in_features / group))
        total_groups = out_features * group_count_per_row
        keep_count = int(mixed_group_keep_count)
        keep_ratio = float(mixed_group_keep_ratio)
        if keep_ratio > 0.0:
            keep_count = max(keep_count, int(math.ceil(total_groups * keep_ratio)))
        keep_count = min(max(keep_count, 0), total_groups)
        explicit_keep_idx = None
        if mixed_group_rank == "group_keep_mask":
            mask_np = mixed_group_mask_by_name.get(module_name)
            if mask_np is None:
                raise ValueError(f"group keep mask missing for module: {module_name}")
            if tuple(mask_np.shape) != (out_features, group_count_per_row):
                raise ValueError(
                    f"group keep mask shape mismatch for {module_name}: "
                    f"mask={tuple(mask_np.shape)} expected={(out_features, group_count_per_row)}"
                )
            mask_tensor = torch.from_numpy(mask_np.reshape(-1)).to(device=original.device, dtype=torch.bool)
            explicit_keep_idx = torch.nonzero(mask_tensor, as_tuple=False).reshape(-1).to(dtype=torch.long)
            keep_count = int(explicit_keep_idx.numel())
        if keep_count <= 0:
            return dequant, {
                "enabled": False,
                "keep_count": 0,
                "keep_ratio_effective": 0.0,
                "group_size": group,
                "rank": mixed_group_rank,
                "kept_elements": 0,
                "kept_element_ratio": 0.0,
                "mask": mixed_group_mask_meta,
            }

        diff = dequant - original
        if explicit_keep_idx is None:
            activation_cov = None
            if mixed_group_rank == "group_activation_error":
                calibration_cov = activation_cov_by_name.get(module_name)
                if calibration_cov is None:
                    raise ValueError(f"activation calibration missing for module: {module_name}")
                activation_cov = torch.from_numpy(calibration_cov).to(device=original.device, dtype=original.dtype)
            score_chunks = []
            for group_idx, start in enumerate(range(0, in_features, group)):
                end = min(start + group, in_features)
                if mixed_group_rank == "group_max_abs_error":
                    scores = diff[:, start:end].abs().amax(dim=1)
                elif mixed_group_rank == "group_weight_norm":
                    part = original[:, start:end]
                    scores = torch.sqrt(torch.mean(part * part, dim=1))
                elif mixed_group_rank == "group_activation_error":
                    part = diff[:, start:end]
                    cov = activation_cov[group_idx, : end - start, : end - start]
                    scores = torch.sqrt(torch.clamp(torch.sum((part @ cov) * part, dim=1), min=0.0))
                else:
                    part = diff[:, start:end]
                    scores = torch.sqrt(torch.mean(part * part, dim=1))
                score_chunks.append(scores)
            score_matrix = torch.stack(score_chunks, dim=1)
            flat_scores = score_matrix.reshape(-1)
            keep_idx = torch.topk(flat_scores, k=keep_count, largest=True, sorted=False).indices
            keep_idx = keep_idx.sort().values
        else:
            keep_idx = explicit_keep_idx.sort().values
            flat_scores = torch.ones((total_groups,), device=original.device, dtype=original.dtype)
        rows = torch.div(keep_idx, group_count_per_row, rounding_mode="floor")
        groups = keep_idx.remainder(group_count_per_row)

        mixed = dequant.clone()
        kept_elements = 0
        kept_sample = []
        for row_tensor, group_tensor in zip(rows, groups):
            row = int(row_tensor.item())
            group_idx = int(group_tensor.item())
            start = group_idx * group
            end = min(start + group, in_features)
            mixed[row, start:end] = original[row, start:end]
            kept_elements += int(end - start)
            if len(kept_sample) < 64:
                kept_sample.append([row, group_idx, start, end])
        kept_scores = flat_scores.index_select(0, keep_idx)
        return mixed, {
            "enabled": True,
            "keep_count": int(keep_count),
            "keep_ratio_effective": float(keep_count / total_groups),
            "group_size": group,
            "rank": mixed_group_rank,
            "group_count_per_row": int(group_count_per_row),
            "total_groups": int(total_groups),
            "kept_elements": int(kept_elements),
            "kept_element_ratio": float(kept_elements / original.numel()),
            "score_min_kept": float(kept_scores.min().item()),
            "score_max_kept": float(kept_scores.max().item()),
            "kept_groups_sample": kept_sample,
            "mask": mixed_group_mask_meta,
        }

    def apply_residual_correction(module_name: str, original: torch.Tensor, dequant: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        rank_requested = int(residual_correction_rank)
        if rank_requested <= 0 or residual_correction_mode == "none":
            return dequant, {
                "enabled": False,
                "mode": residual_correction_mode,
                "rank_requested": rank_requested,
                "rank_effective": 0,
            }
        residual = original - dequant
        max_rank = min(int(residual.shape[0]), int(residual.shape[1]))
        rank = min(rank_requested, max_rank)
        pre_rmse = torch.sqrt(torch.mean((dequant - original) * (dequant - original)))
        weighted_pre_rmse = None
        weighted_post_rmse = None
        covariance_group_size = None
        covariance_count = None

        if residual_correction_mode == "activation_svd":
            calibration_cov = residual_activation_cov_by_name.get(module_name)
            if calibration_cov is None:
                raise ValueError(f"activation residual calibration missing for module: {module_name}")
            activation_cov = torch.from_numpy(calibration_cov).to(device=original.device, dtype=original.dtype)
            covariance_count = int(residual_activation_count_by_name.get(module_name, 0))
            normalize = float(max(covariance_count, 1))
            covariance_group_size = int(activation_cov.shape[-1])
            weighted_chunks = []
            invsqrt_chunks = []
            widths = []
            for group_idx, start in enumerate(range(0, int(residual.shape[1]), covariance_group_size)):
                end = min(start + covariance_group_size, int(residual.shape[1]))
                width = end - start
                cov = activation_cov[group_idx, :width, :width] / normalize
                cov = (cov + cov.transpose(0, 1)) * 0.5
                evals, evecs = torch.linalg.eigh(cov)
                max_eval = torch.clamp(evals.max(), min=1.0e-30)
                floor = max_eval * 1.0e-6
                evals = torch.clamp(evals, min=floor)
                sqrt_cov = (evecs * torch.sqrt(evals)[None, :]) @ evecs.transpose(0, 1)
                invsqrt_cov = (evecs * torch.rsqrt(evals)[None, :]) @ evecs.transpose(0, 1)
                weighted_chunks.append(residual[:, start:end] @ sqrt_cov)
                invsqrt_chunks.append(invsqrt_cov)
                widths.append(width)
            weighted_residual = torch.cat(weighted_chunks, dim=1)
            u, s, vh = torch.linalg.svd(weighted_residual, full_matrices=False)
            weighted_correction = (u[:, :rank] * s[:rank]) @ vh[:rank]
            correction_chunks = []
            offset = 0
            for width, invsqrt_cov in zip(widths, invsqrt_chunks):
                correction_chunks.append(weighted_correction[:, offset:offset + width] @ invsqrt_cov)
                offset += width
            correction = torch.cat(correction_chunks, dim=1)
            weighted_pre_rmse = torch.sqrt(torch.mean(weighted_residual * weighted_residual))
            weighted_post = weighted_residual - weighted_correction
            weighted_post_rmse = torch.sqrt(torch.mean(weighted_post * weighted_post))
        else:
            u, s, vh = torch.linalg.svd(residual, full_matrices=False)
            correction = (u[:, :rank] * s[:rank]) @ vh[:rank]

        corrected = dequant + correction
        post_diff = corrected - original
        post_rmse = torch.sqrt(torch.mean(post_diff * post_diff))
        denom = torch.clamp(torch.sum(s * s), min=1.0e-30)
        energy_fraction = torch.sum(s[:rank] * s[:rank]) / denom
        # Runtime equivalent is Linear(x, W_pc8) + (x @ V_r.T) @ (U_r * S_r).T.
        params = int(rank * (int(original.shape[0]) + int(original.shape[1]) + 1))
        meta = {
            "enabled": True,
            "mode": residual_correction_mode,
            "rank_requested": int(rank_requested),
            "rank_effective": int(rank),
            "pre_rmse": float(pre_rmse.item()),
            "post_rmse": float(post_rmse.item()),
            "energy_fraction": float(energy_fraction.item()),
            "approx_params": int(params),
            "dense_weight_params": int(original.numel()),
            "param_ratio_vs_dense": float(params / original.numel()),
            "semantics": "Quality probe for additive low-rank residual correction: W ~= W_pc8 + U_r diag(S_r) V_r. Runtime equivalent is two small GEMMs plus the quantized GEMM.",
        }
        if residual_correction_mode == "activation_svd":
            meta.update({
                "weighted_pre_rmse": float(weighted_pre_rmse.item()) if weighted_pre_rmse is not None else None,
                "weighted_post_rmse": float(weighted_post_rmse.item()) if weighted_post_rmse is not None else None,
                "covariance_group_size": int(covariance_group_size) if covariance_group_size is not None else None,
                "covariance_count": int(covariance_count) if covariance_count is not None else None,
                "calibration": residual_activation_cov_meta,
            })
        return corrected, meta

    with torch.no_grad():
        for name, module in flow_model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not matches(name) or not supported_dims(module):
                skipped.append(name)
                continue
            original = module.weight.detach().to(dtype=torch.float32)
            external_weight_module_meta: dict[str, Any] = {"enabled": False}
            if external_weight_npz:
                external_weight_np = external_weight_by_name.get(name)
                if external_weight_np is None:
                    raise ValueError(f"external dequant weight missing for selected module: {name}")
                if tuple(external_weight_np.shape) != tuple(original.shape):
                    raise ValueError(
                        f"external dequant weight shape mismatch for {name}: "
                        f"weight={tuple(external_weight_np.shape)} expected={tuple(original.shape)}"
                    )
                dequant = torch.from_numpy(external_weight_np).to(device=original.device, dtype=original.dtype)
                scale = torch.ones((1,), device=original.device, dtype=original.dtype)
                external_weight_module_meta = {
                    "enabled": True,
                    "path": external_weight_npz,
                    "shape": [int(v) for v in external_weight_np.shape],
                    "dtype": str(external_weight_np.dtype),
                }
            else:
                dequant, scale = quant_dequant(original)
            dequant, mixed_meta = keep_float_rows(name, original, dequant)
            dequant, mixed_group_meta = keep_float_input_groups(name, original, dequant)
            dequant, residual_correction_meta = apply_residual_correction(name, original, dequant)
            diff = dequant - original
            rmse_tensor = torch.sqrt(torch.mean(diff * diff))
            denom = torch.clamp(torch.sqrt(torch.mean(original * original)), min=1.0e-30)
            module.weight.copy_(dequant.to(dtype=module.weight.dtype))
            selected.append(name)
            selected_dims[name] = f"{int(module.in_features)}x{int(module.out_features)}"
            mixed_total_keep_rows += int(mixed_meta.get("keep_count", 0))
            mixed_total_keep_groups += int(mixed_group_meta.get("keep_count", 0))
            mixed_total_keep_group_elements += int(mixed_group_meta.get("kept_elements", 0))
            residual_correction_total_rank += int(residual_correction_meta.get("rank_effective", 0))
            residual_correction_total_params += int(residual_correction_meta.get("approx_params", 0))
            per_module[name] = {
                "shape": [int(v) for v in original.shape],
                "rmse": float(rmse_tensor.item()),
                "rel_rmse": float((rmse_tensor / denom).item()),
                "max_abs": float(diff.abs().amax().item()),
                "scale_min": float(scale.min().item()),
                "scale_max": float(scale.max().item()),
                "external_dequant_weight": external_weight_module_meta,
                "mixed_precision": mixed_meta,
                "mixed_group_precision": mixed_group_meta,
                "residual_correction": residual_correction_meta,
            }

    return {
        "enabled": bool(selected),
        "kind": "dequantized_weight_linear_patch",
        "mode": mode,
        "bits": int(bits),
        "levels": int(levels),
        "percentile": float(percentile),
        "mu": float(mu),
        "patch_mlp": bool(patch_mlp),
        "group_size": int(group_size),
        "external_dequant_weight": external_weight_meta,
        "mixed_precision": {
            "enabled": bool(float(mixed_keep_ratio) > 0.0 or int(mixed_keep_count) > 0),
            "keep_ratio": float(mixed_keep_ratio),
            "keep_count": int(mixed_keep_count),
            "rank": mixed_rank,
            "total_keep_rows": int(mixed_total_keep_rows),
            "output_residual_calibration": output_residual_meta,
        },
        "mixed_group_precision": {
            "enabled": bool(float(mixed_group_keep_ratio) > 0.0 or int(mixed_group_keep_count) > 0 or mixed_group_mask_npz),
            "group_size": int(mixed_group_size),
            "keep_ratio": float(mixed_group_keep_ratio),
            "keep_count": int(mixed_group_keep_count),
            "rank": mixed_group_rank,
            "total_keep_groups": int(mixed_total_keep_groups),
            "total_keep_elements": int(mixed_total_keep_group_elements),
            "calibration": activation_cov_meta,
            "mask": mixed_group_mask_meta,
        },
        "residual_correction": {
            "enabled": bool(int(residual_correction_rank) > 0 and residual_correction_mode != "none"),
            "mode": residual_correction_mode,
            "rank": int(residual_correction_rank),
            "total_rank_effective": int(residual_correction_total_rank),
            "total_approx_params": int(residual_correction_total_params),
            "calibration": residual_activation_cov_meta,
        },
        "include_regex": include_regex,
        "exclude_regex": exclude_regex,
        "selected_count": len(selected),
        "selected": selected[:100],
        "selected_dims": selected_dims,
        "per_module": per_module,
        "skipped_count": len(skipped),
        "skipped_sample": skipped[:50],
        "semantics": "Quality probe: selected nn.Linear weights are quantized and immediately dequantized in-place, or loaded from an external reconstructed-weight NPZ, before normal float32 GEMM. Shared custom F.linear paths see the modified weights too.",
    }





def _rope_self_attention_with_key_bias(attn, x, rope_emb, key_bias, *, backend: str = "default", timing_callback=None):
    import time
    import torch.nn.functional as F
    try:
        import model as triposplat_model
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("could not import TripoSplat model module") from exc

    def _stage(name: str, fn, *, shape=None):
        if timing_callback is None:
            return fn()
        started = time.perf_counter()
        try:
            return fn()
        finally:
            timing_callback(name, time.perf_counter() - started, shape=shape)

    B, L, C = x.shape
    H = attn.num_heads
    D = attn.head_dim
    qkv = _stage(
        "attention.qkv_projection",
        lambda: attn.qkv(x),
        shape=[int(B), int(L), int(3 * C)],
    ).reshape(B, L, 3, H, D)
    q, k, v = qkv.unbind(2)
    if attn.use_rope:
        started = time.perf_counter() if timing_callback is not None else None
        q = triposplat_model.apply_rotary_emb(q, rope_emb)
        k = triposplat_model.apply_rotary_emb(k, rope_emb)
        if started is not None:
            timing_callback("attention.rope", time.perf_counter() - started, shape=[int(B), int(L), int(H), int(D)])
    if attn.qk_rms_norm:
        started = time.perf_counter() if timing_callback is not None else None
        q = attn.q_norm(q)
        k = attn.k_norm(k)
        if started is not None:
            timing_callback("attention.qk_norm", time.perf_counter() - started, shape=[int(B), int(L), int(H), int(D)])
    started = time.perf_counter() if timing_callback is not None else None
    q = q.permute(0, 2, 1, 3).contiguous()
    k = k.permute(0, 2, 1, 3).contiguous()
    v = v.permute(0, 2, 1, 3).contiguous()
    if started is not None:
        timing_callback("attention.layout", time.perf_counter() - started, shape=[int(B), int(H), int(L), int(D)])
    if _is_native_sdpa_backend(backend):
        out = _stage(
            "attention.sdpa",
            lambda: native_exact_scaled_dot_product_attention_bhld(q, k, v, key_bias=key_bias),
            shape=[int(B), int(H), int(L), int(D)],
        )
    elif backend in {"aten_flash_direct", "aten_flash_direct_scale", "aten_flash_direct_auto"}:
        import torch
        direct_scale = 1.0 / math.sqrt(float(q.shape[-1])) if backend == "aten_flash_direct_scale" else None
        out = _stage(
            "attention.sdpa",
            lambda: _direct_flash_attention_with_key_bias(q, k, v, key_bias, direct_scale=direct_scale),
            shape=[int(B), int(H), int(L), int(D)],
        )
    elif backend == "f_explicit_scale":
        out = _stage(
            "attention.sdpa",
            lambda: F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=key_bias,
                scale=1.0 / math.sqrt(float(q.shape[-1])),
            ),
            shape=[int(B), int(H), int(L), int(D)],
        )
    else:
        out = _stage(
            "attention.sdpa",
            lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=key_bias),
            shape=[int(B), int(H), int(L), int(D)],
        )
    out = out.permute(0, 2, 1, 3).reshape(B, L, C)
    return _stage("attention.out_projection", lambda: attn.out(out), shape=[int(B), int(L), int(C)])



def _rope_self_attention_standard(
    attn,
    x,
    rope_emb,
    *,
    backend: str = "default",
    compute_dtype: str = "model",
    query_chunk_size: int = 128,
    contiguous_qkv: bool = True,
    timing_callback=None,
):
    import time
    import torch
    import torch.nn.functional as F
    try:
        import model as triposplat_model
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("could not import TripoSplat model module") from exc

    def _stage(name: str, fn, *, shape=None):
        if timing_callback is None:
            return fn()
        started = time.perf_counter()
        try:
            return fn()
        finally:
            timing_callback(name, time.perf_counter() - started, shape=shape)

    if compute_dtype not in {"model", "float32"}:
        raise ValueError(f"unsupported attention compute dtype: {compute_dtype}")
    if attn._type != "self":
        raise ValueError("standard timed attention path only supports self-attention")

    B, L, C = x.shape
    H = attn.num_heads
    D = attn.head_dim
    qkv = _stage(
        "attention.qkv_projection",
        lambda: attn.qkv(x),
        shape=[int(B), int(L), int(3 * C)],
    ).reshape(B, L, 3, H, D)
    q, k, v = qkv.unbind(2)
    if attn.use_rope:
        started = time.perf_counter() if timing_callback is not None else None
        q = triposplat_model.apply_rotary_emb(q, rope_emb)
        k = triposplat_model.apply_rotary_emb(k, rope_emb)
        if started is not None:
            timing_callback("attention.rope", time.perf_counter() - started, shape=[int(B), int(L), int(H), int(D)])
    if attn.qk_rms_norm:
        started = time.perf_counter() if timing_callback is not None else None
        q = attn.q_norm(q)
        k = attn.k_norm(k)
        if started is not None:
            timing_callback("attention.qk_norm", time.perf_counter() - started, shape=[int(B), int(L), int(H), int(D)])
    orig_dtype = q.dtype
    started = time.perf_counter() if timing_callback is not None else None
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    if contiguous_qkv:
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
    if started is not None:
        timing_callback("attention.layout", time.perf_counter() - started, shape=[int(B), int(H), int(L), int(D)])
    if compute_dtype == "float32":
        q = q.float()
        k = k.float()
        v = v.float()
    if _is_native_sdpa_backend(backend):
        out = _stage(
            "attention.sdpa",
            lambda: native_exact_scaled_dot_product_attention_bhld(q, k, v),
            shape=[int(B), int(H), int(L), int(D)],
        )
    elif backend == "chunked":
        out = _stage(
            "attention.sdpa",
            lambda: chunked_exact_scaled_dot_product_attention_bhld(
                q, k, v, query_chunk_size=int(query_chunk_size), compute_dtype=compute_dtype
            ),
            shape=[int(B), int(H), int(L), int(D)],
        )
    elif _is_streaming_backend(backend):
        out = _stage(
            "attention.sdpa",
            lambda: streaming_exact_scaled_dot_product_attention_bhld(q, k, v, backend=backend),
            shape=[int(B), int(H), int(L), int(D)],
        )
    elif backend in {"aten_flash_direct", "aten_flash_direct_scale", "aten_flash_direct_auto"}:
        direct_scale = None
        if backend == "aten_flash_direct_scale" or (backend == "aten_flash_direct_auto" and q.shape[-2] == k.shape[-2]):
            direct_scale = 1.0 / math.sqrt(float(q.shape[-1]))
        out = _stage(
            "attention.sdpa",
            lambda: torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(
                q, k, v, 0.0, False, attn_mask=None, scale=direct_scale
            )[0],
            shape=[int(B), int(H), int(L), int(D)],
        )
    elif backend == "f_explicit_scale":
        out = _stage(
            "attention.sdpa",
            lambda: F.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(float(q.shape[-1]))),
            shape=[int(B), int(H), int(L), int(D)],
        )
    else:
        with _backend_context(backend):
            out = _stage(
                "attention.sdpa",
                lambda: F.scaled_dot_product_attention(q, k, v),
                shape=[int(B), int(H), int(L), int(D)],
            )
    out = out.permute(0, 2, 1, 3).to(dtype=orig_dtype)
    return _stage("attention.out_projection", lambda: attn.out(out.reshape(B, L, C)), shape=[int(B), int(L), int(C)])



def _feed_forward_mlp_timed(mlp, x, *, timing_callback=None):
    import time

    def _shape_with_last(tensor, last_dim):
        shape = getattr(tensor, "shape", None)
        if shape is None:
            return None
        values = [int(part) for part in shape]
        if values:
            values[-1] = int(last_dim)
        return values

    def _stage(name: str, fn, *, shape=None):
        if timing_callback is None:
            return fn()
        started = time.perf_counter()
        try:
            return fn()
        finally:
            timing_callback(name, time.perf_counter() - started, shape=shape)

    if timing_callback is None:
        return mlp(x)
    patched_attrs = (
        "_original_forward_mkldnn_fused_mlp",
        "_original_forward_gelu_out_buffer_mlp",
        "_original_forward_chunked_mlp",
    )
    if any(hasattr(mlp, attr) for attr in patched_attrs):
        return mlp(x)
    layers = list(getattr(mlp, "mlp", []))
    if len(layers) != 3:
        return mlp(x)
    l0, act, l2 = layers
    if not hasattr(l0, "weight") or not hasattr(l2, "weight"):
        return mlp(x)
    h = _stage("mlp.fc1", lambda: l0(x), shape=_shape_with_last(x, getattr(l0, "out_features", x.shape[-1])))
    h = _stage("mlp.gelu", lambda: act(h), shape=[int(v) for v in h.shape])
    return _stage("mlp.fc2", lambda: l2(h), shape=_shape_with_last(h, getattr(l2, "out_features", h.shape[-1])))


_COMPILED_REALROPE_ATTENTION_CACHE: dict[tuple[int, int, int, int, int, str], object] = {}
_COMPILED_REALROPE_FULLBLOCK_CACHE: dict[tuple[int, int, int, int, int, float, str], object] = {}
_COMPILED_REALROPE_SELECTED_CACHE: dict[tuple[int, int, int, int, int, int, float, str, bool], object] = {}
_COMPILED_REALROPE_NEG_FULLBLOCK_CACHE: dict[tuple[int, int, int, int, int, float, str, bool], object] = {}


def _repo_angles(repo, hidden_states):
    import torch
    try:
        import model as triposplat_model
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("could not import TripoSplat model module") from exc

    h = repo.norm(hidden_states)
    feat = repo.act(repo.gate_map(h)) * repo.content_map(h)
    out = repo.final_map(feat)
    B, L, _ = out.shape
    delta_pos = out.reshape(B, L, repo.num_heads, 3)
    ang_0 = triposplat_model.clamp_mul(delta_pos[..., 0].unsqueeze(-1), repo.freqs_0) * torch.pi
    ang_1 = triposplat_model.clamp_mul(delta_pos[..., 1].unsqueeze(-1), repo.freqs_1) * torch.pi
    ang_2 = triposplat_model.clamp_mul(delta_pos[..., 2].unsqueeze(-1), repo.freqs_2) * torch.pi
    return torch.cat([ang_0, ang_1, ang_2], dim=-1).float()


def _apply_rotary_real_from_angles(hidden_states, angles):
    import torch

    x = hidden_states.float().reshape(*hidden_states.shape[:-1], -1, 2)
    x0 = x[..., 0]
    x1 = x[..., 1]
    c = torch.cos(angles)
    s = torch.sin(angles)
    y0 = x0 * c - x1 * s
    y1 = x0 * s + x1 * c
    y = torch.stack((y0, y1), dim=-1).reshape_as(hidden_states.float())
    return y.type_as(hidden_states)


def _rms_norm_official(x, weight):
    import torch.nn.functional as F

    return (F.normalize(x.float(), dim=-1) * weight.float() * math.sqrt(float(x.shape[-1]))).to(x.dtype)


def _compiled_realrope_attention_path(B: int, L: int, C: int, H: int, D: int):
    import torch
    import torch.nn.functional as F
    from pathlib import Path

    mode = os.environ.get("TRIPOSPLAT_POS_COMPILED_REALROPE_MODE", "reduce-overhead")
    key = (int(B), int(L), int(C), int(H), int(D), mode)
    cached = _COMPILED_REALROPE_ATTENTION_CACHE.get(key)
    if cached is not None:
        return cached

    wrapper = Path.cwd() / "scripts/cxx20_to_cxx2a_wrapper.sh"
    if wrapper.exists():
        os.environ.setdefault("CXX", wrapper.as_posix())
    scale = 1.0 / math.sqrt(float(D))

    def path_fn(x_arg, wqkv_arg, bqkv_arg, wout_arg, bout_arg, angles_arg, qnw_arg, knw_arg):
        qkv = F.linear(x_arg, wqkv_arg, bqkv_arg)
        q, k, v = qkv.reshape(B, L, 3, H, D).unbind(2)
        q = _apply_rotary_real_from_angles(q, angles_arg)
        k = _apply_rotary_real_from_angles(k, angles_arg)
        q = _rms_norm_official(q, qnw_arg)
        k = _rms_norm_official(k, knw_arg)
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
        y = F.scaled_dot_product_attention(q, k, v, scale=scale)
        y = y.permute(0, 2, 1, 3).reshape(B, L, C)
        return F.linear(y, wout_arg, bout_arg)

    compiled = torch.compile(
        path_fn,
        backend="inductor",
        mode=None if mode == "none" else mode,
        fullgraph=False,
        dynamic=False,
    )
    _COMPILED_REALROPE_ATTENTION_CACHE[key] = compiled
    return compiled


def _rope_self_attention_compiled_realrope(attn, x, angles):
    B, L, C = x.shape
    H = int(attn.num_heads)
    D = int(attn.head_dim)
    if not bool(getattr(attn, "use_rope", False)) or not bool(getattr(attn, "qk_rms_norm", False)):
        raise ValueError("compiled real-RoPE positive attention requires use_rope and qk_rms_norm")
    compiled = _compiled_realrope_attention_path(B, L, C, H, D)
    return compiled(
        x,
        attn.qkv.weight,
        attn.qkv.bias,
        attn.out.weight,
        attn.out.bias,
        angles,
        attn.q_norm.gamma,
        attn.k_norm.gamma,
    )


def _compiled_realrope_fullblock_path(B: int, L: int, C: int, H: int, D: int, eps: float):
    import torch
    import torch.nn.functional as F
    from pathlib import Path

    mode = os.environ.get(
        "TRIPOSPLAT_POS_FULLBLOCK_COMPILED_REALROPE_MODE",
        os.environ.get("TRIPOSPLAT_POS_COMPILED_REALROPE_MODE", "reduce-overhead"),
    )
    fullgraph = os.environ.get("TRIPOSPLAT_POS_FULLBLOCK_COMPILED_REALROPE_FULLGRAPH", "").strip().lower() in {"1", "true", "yes", "on"}
    key = (int(B), int(L), int(C), int(H), int(D), float(eps), mode, bool(fullgraph))
    cached = _COMPILED_REALROPE_FULLBLOCK_CACHE.get(key)
    if cached is not None:
        return cached

    wrapper = Path.cwd() / "scripts/cxx20_to_cxx2a_wrapper.sh"
    if wrapper.exists():
        os.environ.setdefault("CXX", wrapper.as_posix())
    scale = 1.0 / math.sqrt(float(D))

    def path_fn(
        x_arg,
        mod_arg,
        shift_table_arg,
        angles_arg,
        wqkv_arg,
        bqkv_arg,
        wout_arg,
        bout_arg,
        qnw_arg,
        knw_arg,
        wmlp0_arg,
        bmlp0_arg,
        wmlp2_arg,
        bmlp2_arg,
    ):
        mod_work = mod_arg + shift_table_arg.type(mod_arg.dtype)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod_work.chunk(6, dim=1)

        h = F.layer_norm(x_arg.float(), (C,), None, None, eps).to(x_arg.dtype)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        qkv = F.linear(h, wqkv_arg, bqkv_arg)
        q, k, v = qkv.reshape(B, L, 3, H, D).unbind(2)
        q = _apply_rotary_real_from_angles(q, angles_arg)
        k = _apply_rotary_real_from_angles(k, angles_arg)
        q = _rms_norm_official(q, qnw_arg)
        k = _rms_norm_official(k, knw_arg)
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
        h = F.scaled_dot_product_attention(q, k, v, scale=scale)
        h = h.permute(0, 2, 1, 3).reshape(B, L, C)
        h = F.linear(h, wout_arg, bout_arg)

        x1 = x_arg + h * gate_msa.unsqueeze(1)
        h2 = F.layer_norm(x1.float(), (C,), None, None, eps).to(x1.dtype)
        h2 = h2 * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp = F.linear(h2, wmlp0_arg, bmlp0_arg)
        mlp = F.gelu(mlp, approximate="tanh")
        mlp = F.linear(mlp, wmlp2_arg, bmlp2_arg)
        return x1 + mlp * gate_mlp.unsqueeze(1)

    compiled = torch.compile(
        path_fn,
        backend="inductor",
        mode=None if mode == "none" else mode,
        fullgraph=bool(fullgraph),
        dynamic=False,
    )
    _COMPILED_REALROPE_FULLBLOCK_CACHE[key] = compiled
    return compiled


def _unified_block_forward_standard_inplace_fullblock_compiled_realrope(block, x, mod, repo_layer):
    if not bool(getattr(block, "modulation", False)):
        raise ValueError("positive full-block compiled real-RoPE currently expects modulated TripoSplat main blocks")
    if not bool(getattr(block, "share_mod", False)) or getattr(block, "shift_table", None) is None:
        raise ValueError("positive full-block compiled real-RoPE currently expects share_mod=True with shift_table")
    B, L, C = x.shape
    H = int(block.attn.num_heads)
    D = int(block.attn.head_dim)
    if not bool(getattr(block.attn, "use_rope", False)) or not bool(getattr(block.attn, "qk_rms_norm", False)):
        raise ValueError("positive full-block compiled real-RoPE requires use_rope and qk_rms_norm")
    mlp0 = block.mlp.mlp[0]
    mlp2 = block.mlp.mlp[2]
    compiled = _compiled_realrope_fullblock_path(B, L, C, H, D, float(block.norm1.eps))
    angles = _repo_angles(repo_layer, x)
    return compiled(
        x,
        mod,
        block.shift_table,
        angles,
        block.attn.qkv.weight,
        block.attn.qkv.bias,
        block.attn.out.weight,
        block.attn.out.bias,
        block.attn.q_norm.gamma,
        block.attn.k_norm.gamma,
        mlp0.weight,
        mlp0.bias,
        mlp2.weight,
        mlp2.bias,
    )


def warmup_triposplat_positive_fullblock_compiled_realrope(
    flow_model,
    torch_module,
    *,
    sequence_length: int,
    dtype=None,
    device=None,
) -> dict[str, Any]:
    """Compile/load the positive full-block real-RoPE graph before sampler timing."""
    meta: dict[str, Any] = {
        "enabled": True,
        "kind": "positive_fullblock_compiled_realrope_warmup",
        "sequence_length": int(sequence_length),
    }
    if not hasattr(flow_model, "blocks") or not hasattr(flow_model, "repo_layers") or not flow_model.blocks:
        raise ValueError("flow_model has no main blocks/repo_layers for positive full-block compile warmup")
    block = flow_model.blocks[0]
    repo_layer = flow_model.repo_layers[0]
    C = int(getattr(block.attn, "channels", 1024))
    d = dtype if dtype is not None else next(flow_model.parameters()).dtype
    dev = device if device is not None else next(flow_model.parameters()).device
    x = torch_module.zeros((1, int(sequence_length), C), device=dev, dtype=d)
    mod = torch_module.zeros((1, 6 * C), device=dev, dtype=d)
    import time

    started = time.time()
    with torch_module.inference_mode():
        out = _unified_block_forward_standard_inplace_fullblock_compiled_realrope(block, x, mod, repo_layer)
        meta["materialized_scalar"] = float(out.reshape(-1)[0].detach().float().cpu())
    meta["warmup_elapsed_sec"] = time.time() - started
    meta["block_index"] = 0
    meta["channels"] = C
    meta["dtype"] = str(d).replace("torch.", "")
    meta["device"] = str(dev)
    meta["mode"] = os.environ.get(
        "TRIPOSPLAT_POS_FULLBLOCK_COMPILED_REALROPE_MODE",
        os.environ.get("TRIPOSPLAT_POS_COMPILED_REALROPE_MODE", "reduce-overhead"),
    )
    meta["fullgraph"] = os.environ.get("TRIPOSPLAT_POS_FULLBLOCK_COMPILED_REALROPE_FULLGRAPH", "").strip().lower() in {"1", "true", "yes", "on"}
    return meta


def _compiled_realrope_selected_path(B: int, L: int, S: int, C: int, H: int, D: int, eps: float):
    import torch
    import torch.nn.functional as F
    from pathlib import Path

    mode = os.environ.get("TRIPOSPLAT_POS_FINAL_SELECTED_COMPILED_REALROPE_MODE", "reduce-overhead")
    fullgraph = os.environ.get("TRIPOSPLAT_POS_FINAL_SELECTED_COMPILED_REALROPE_FULLGRAPH", "").strip().lower() in {"1", "true", "yes", "on"}
    key = (int(B), int(L), int(S), int(C), int(H), int(D), float(eps), mode, bool(fullgraph))
    cached = _COMPILED_REALROPE_SELECTED_CACHE.get(key)
    if cached is not None:
        return cached

    wrapper = Path.cwd() / "scripts/cxx20_to_cxx2a_wrapper.sh"
    if wrapper.exists():
        os.environ.setdefault("CXX", wrapper.as_posix())
    scale = 1.0 / math.sqrt(float(D))

    def path_fn(
        x_arg, mod_arg, shift_table_arg, selected_idx_arg, angles_arg,
        wq_arg, bq_arg, wkv_arg, bkv_arg, wout_arg, bout_arg,
        qnw_arg, knw_arg, wmlp0_arg, bmlp0_arg, wmlp2_arg, bmlp2_arg,
    ):
        mod_work = mod_arg + shift_table_arg.type(mod_arg.dtype)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod_work.chunk(6, dim=1)
        h = F.layer_norm(x_arg.float(), (C,), None, None, eps).to(x_arg.dtype)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h_sel = h.index_select(1, selected_idx_arg)
        q = F.linear(h_sel, wq_arg, bq_arg).reshape(B, S, H, D)
        kv = F.linear(h, wkv_arg, bkv_arg).reshape(B, L, 2, H, D)
        k, v = kv.unbind(2)
        q = _apply_rotary_real_from_angles(q, angles_arg.index_select(1, selected_idx_arg))
        k = _apply_rotary_real_from_angles(k, angles_arg)
        q = _rms_norm_official(q, qnw_arg)
        k = _rms_norm_official(k, knw_arg)
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
        h_attn = F.scaled_dot_product_attention(q, k, v, scale=scale)
        h_attn = h_attn.permute(0, 2, 1, 3).reshape(B, S, C)
        h_attn = F.linear(h_attn, wout_arg, bout_arg)
        x_sel = x_arg.index_select(1, selected_idx_arg) + h_attn * gate_msa.unsqueeze(1)
        h2 = F.layer_norm(x_sel.float(), (C,), None, None, eps).to(x_arg.dtype)
        h2 = h2 * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp = F.linear(h2, wmlp0_arg, bmlp0_arg)
        mlp = F.gelu(mlp, approximate="tanh")
        mlp = F.linear(mlp, wmlp2_arg, bmlp2_arg)
        return x_sel + mlp * gate_mlp.unsqueeze(1)

    compiled = torch.compile(
        path_fn,
        backend="inductor",
        mode=None if mode == "none" else mode,
        fullgraph=bool(fullgraph),
        dynamic=False,
    )
    _COMPILED_REALROPE_SELECTED_CACHE[key] = compiled
    return compiled


def _unified_block_forward_selected_rows_only_compiled_realrope(block, x, mod, repo_layer, selected_idx):
    if not bool(getattr(block, "modulation", False)):
        raise ValueError("positive final selected compiled real-RoPE currently expects modulated TripoSplat main blocks")
    if not bool(getattr(block, "share_mod", False)) or getattr(block, "shift_table", None) is None:
        raise ValueError("positive final selected compiled real-RoPE currently expects share_mod=True with shift_table")
    if not bool(getattr(block.attn, "use_rope", False)) or not bool(getattr(block.attn, "qk_rms_norm", False)):
        raise ValueError("positive final selected compiled real-RoPE requires use_rope and qk_rms_norm")
    B, L, C = x.shape
    S = int(selected_idx.numel())
    H = int(block.attn.num_heads)
    D = int(block.attn.head_dim)
    qkv_w = block.attn.qkv.weight
    qkv_b = block.attn.qkv.bias
    mlp0 = block.mlp.mlp[0]
    mlp2 = block.mlp.mlp[2]
    compiled = _compiled_realrope_selected_path(B, L, S, C, H, D, float(block.norm1.eps))
    angles = _repo_angles(repo_layer, x)
    return compiled(
        x, mod, block.shift_table, selected_idx, angles,
        qkv_w[:C], None if qkv_b is None else qkv_b[:C],
        qkv_w[C:], None if qkv_b is None else qkv_b[C:],
        block.attn.out.weight, block.attn.out.bias,
        block.attn.q_norm.gamma, block.attn.k_norm.gamma,
        mlp0.weight, mlp0.bias, mlp2.weight, mlp2.bias,
    )


def warmup_triposplat_positive_final_selected_compiled_realrope(
    flow_model, torch_module, *, sequence_length: int, latent_tokens: int,
    camera_tokens: int, dtype=None, device=None,
) -> dict[str, Any]:
    """Compile/load the positive final selected-row real-RoPE graph before sampler timing."""
    meta: dict[str, Any] = {
        "enabled": True,
        "kind": "positive_final_selected_compiled_realrope_warmup",
        "sequence_length": int(sequence_length),
        "latent_tokens": int(latent_tokens),
        "camera_tokens": int(camera_tokens),
    }
    if not hasattr(flow_model, "blocks") or not hasattr(flow_model, "repo_layers") or not flow_model.blocks:
        raise ValueError("flow_model has no main blocks/repo_layers for positive final selected compile warmup")
    block = flow_model.blocks[-1]
    repo_layer = flow_model.repo_layers[-1]
    C = int(getattr(block.attn, "channels", 1024))
    d = dtype if dtype is not None else next(flow_model.parameters()).dtype
    dev = device if device is not None else next(flow_model.parameters()).device
    x = torch_module.zeros((1, int(sequence_length), C), device=dev, dtype=d)
    mod = torch_module.zeros((1, 6 * C), device=dev, dtype=d)
    selected_idx = _selected_latent_camera_idx(int(sequence_length), int(latent_tokens), int(camera_tokens), dev)
    import time

    started = time.time()
    with torch_module.inference_mode():
        out = _unified_block_forward_selected_rows_only_compiled_realrope(block, x, mod, repo_layer, selected_idx)
        meta["materialized_scalar"] = float(out.reshape(-1)[0].detach().float().cpu())
    meta["warmup_elapsed_sec"] = time.time() - started
    meta["block_index"] = len(flow_model.blocks) - 1
    meta["selected_rows"] = int(selected_idx.numel())
    meta["channels"] = C
    meta["dtype"] = str(d).replace("torch.", "")
    meta["device"] = str(dev)
    meta["mode"] = os.environ.get("TRIPOSPLAT_POS_FINAL_SELECTED_COMPILED_REALROPE_MODE", "reduce-overhead")
    meta["fullgraph"] = os.environ.get("TRIPOSPLAT_POS_FINAL_SELECTED_COMPILED_REALROPE_FULLGRAPH", "").strip().lower() in {"1", "true", "yes", "on"}
    return meta


def _compiled_realrope_neg_fullblock_path(B: int, L: int, C: int, H: int, D: int, eps: float):
    import torch
    import torch.nn.functional as F
    from pathlib import Path

    mode = os.environ.get("TRIPOSPLAT_NEG_FULLBLOCK_COMPILED_REALROPE_MODE", "none")
    fullgraph = os.environ.get("TRIPOSPLAT_NEG_FULLBLOCK_COMPILED_REALROPE_FULLGRAPH", "").strip().lower() in {"1", "true", "yes", "on"}
    key = (int(B), int(L), int(C), int(H), int(D), float(eps), mode, bool(fullgraph))
    cached = _COMPILED_REALROPE_NEG_FULLBLOCK_CACHE.get(key)
    if cached is not None:
        return cached

    wrapper = Path.cwd() / "scripts/cxx20_to_cxx2a_wrapper.sh"
    if wrapper.exists():
        os.environ.setdefault("CXX", wrapper.as_posix())

    def path_fn(
        x_arg, mod_arg, shift_table_arg, angles_arg, key_bias_arg,
        wqkv_arg, bqkv_arg, wout_arg, bout_arg, qnw_arg, knw_arg,
        wmlp0_arg, bmlp0_arg, wmlp2_arg, bmlp2_arg,
    ):
        mod_work = mod_arg + shift_table_arg.type(mod_arg.dtype)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod_work.chunk(6, dim=1)
        h = F.layer_norm(x_arg.float(), (C,), None, None, eps).to(x_arg.dtype)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        qkv = F.linear(h, wqkv_arg, bqkv_arg).reshape(B, L, 3, H, D)
        q, k, v = qkv.unbind(2)
        q = _apply_rotary_real_from_angles(q, angles_arg)
        k = _apply_rotary_real_from_angles(k, angles_arg)
        q = _rms_norm_official(q, qnw_arg)
        k = _rms_norm_official(k, knw_arg)
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
        h_attn = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(
            q, k, v, 0.0, False, attn_mask=key_bias_arg, scale=None
        )[0]
        h_attn = h_attn.permute(0, 2, 1, 3).reshape(B, L, C)
        h_attn = F.linear(h_attn, wout_arg, bout_arg)
        x1 = x_arg + h_attn * gate_msa.unsqueeze(1)
        h2 = F.layer_norm(x1.float(), (C,), None, None, eps).to(x_arg.dtype)
        h2 = h2 * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp = F.linear(h2, wmlp0_arg, bmlp0_arg)
        mlp = F.gelu(mlp, approximate="tanh")
        mlp = F.linear(mlp, wmlp2_arg, bmlp2_arg)
        return x1 + mlp * gate_mlp.unsqueeze(1)

    compiled = torch.compile(
        path_fn,
        backend="inductor",
        mode=None if mode == "none" else mode,
        fullgraph=bool(fullgraph),
        dynamic=False,
    )
    _COMPILED_REALROPE_NEG_FULLBLOCK_CACHE[key] = compiled
    return compiled


def _unified_block_forward_with_key_bias_fullblock_compiled_realrope(block, x, mod, repo_layer, key_bias):
    if not bool(getattr(block, "modulation", False)):
        raise ValueError("negative full-block compiled real-RoPE currently expects modulated TripoSplat main blocks")
    if not bool(getattr(block, "share_mod", False)) or getattr(block, "shift_table", None) is None:
        raise ValueError("negative full-block compiled real-RoPE currently expects share_mod=True with shift_table")
    if not bool(getattr(block.attn, "use_rope", False)) or not bool(getattr(block.attn, "qk_rms_norm", False)):
        raise ValueError("negative full-block compiled real-RoPE requires use_rope and qk_rms_norm")
    B, L, C = x.shape
    H = int(block.attn.num_heads)
    D = int(block.attn.head_dim)
    mlp0 = block.mlp.mlp[0]
    mlp2 = block.mlp.mlp[2]
    compiled = _compiled_realrope_neg_fullblock_path(B, L, C, H, D, float(block.norm1.eps))
    angles = _repo_angles(repo_layer, x)
    return compiled(
        x, mod, block.shift_table, angles, key_bias,
        block.attn.qkv.weight, block.attn.qkv.bias,
        block.attn.out.weight, block.attn.out.bias,
        block.attn.q_norm.gamma, block.attn.k_norm.gamma,
        mlp0.weight, mlp0.bias, mlp2.weight, mlp2.bias,
    )


def warmup_triposplat_negative_fullblock_compiled_realrope(
    flow_model, torch_module, *, sequence_length: int, condition_tokens: int,
    negative_condition_index: int, dtype=None, device=None,
) -> dict[str, Any]:
    """Compile/load the negative compact key-bias full-block real-RoPE graph before sampler timing."""
    meta: dict[str, Any] = {
        "enabled": True,
        "kind": "negative_fullblock_compiled_realrope_warmup",
        "sequence_length": int(sequence_length),
        "condition_tokens": int(condition_tokens),
        "negative_condition_index": int(negative_condition_index),
    }
    if not hasattr(flow_model, "blocks") or not hasattr(flow_model, "repo_layers") or not flow_model.blocks:
        raise ValueError("flow_model has no main blocks/repo_layers for negative full-block compile warmup")
    block = flow_model.blocks[0]
    repo_layer = flow_model.repo_layers[0]
    C = int(getattr(block.attn, "channels", 1024))
    d = dtype if dtype is not None else next(flow_model.parameters()).dtype
    dev = device if device is not None else next(flow_model.parameters()).device
    x = torch_module.zeros((1, int(sequence_length), C), device=dev, dtype=d)
    mod = torch_module.zeros((1, 6 * C), device=dev, dtype=d)
    key_bias = torch_module.zeros(1, 1, 1, int(sequence_length), device=dev, dtype=torch_module.float32)
    key_bias[..., int(negative_condition_index)] = math.log(float(condition_tokens))
    import time

    started = time.time()
    with torch_module.inference_mode():
        out = _unified_block_forward_with_key_bias_fullblock_compiled_realrope(block, x, mod, repo_layer, key_bias)
        meta["materialized_scalar"] = float(out.reshape(-1)[0].detach().float().cpu())
    meta["warmup_elapsed_sec"] = time.time() - started
    meta["block_index"] = 0
    meta["channels"] = C
    meta["dtype"] = str(d).replace("torch.", "")
    meta["device"] = str(dev)
    meta["mode"] = os.environ.get("TRIPOSPLAT_NEG_FULLBLOCK_COMPILED_REALROPE_MODE", "none")
    meta["fullgraph"] = os.environ.get("TRIPOSPLAT_NEG_FULLBLOCK_COMPILED_REALROPE_FULLGRAPH", "").strip().lower() in {"1", "true", "yes", "on"}
    return meta


def _unified_block_forward_standard_inplace(block, x, mod, rotary_emb, *, attention_backend: str = "default", timing_callback=None):
    import time

    def _stage(name: str, fn, *, shape=None):
        if timing_callback is None:
            return fn()
        started = time.perf_counter()
        try:
            return fn()
        finally:
            timing_callback(name, time.perf_counter() - started, shape=shape)

    if block.modulation:
        if not block.share_mod:
            mod = block.adaLN_modulation(mod)
        if hasattr(block, "shift_table") and block.shift_table is not None:
            mod = _native_shift_table_add(block, mod, block.shift_table.type(mod.dtype))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        h = _stage("norm1", lambda: block.norm1(x), shape=[int(v) for v in x.shape])
        h = _stage("modulate_msa", lambda: _modulate_inplace_preserve_order(h, scale_msa, shift_msa), shape=[int(v) for v in h.shape])
        h = _stage(
            "attention.total",
            lambda: _rope_self_attention_standard(
                block.attn,
                h,
                rotary_emb,
                backend=attention_backend,
                compute_dtype="model",
                query_chunk_size=128,
                contiguous_qkv=True,
                timing_callback=timing_callback,
            ),
            shape=[int(v) for v in h.shape],
        )
        x = _stage("residual_msa", lambda: _residual_gate_inplace_preserve_order(x, h, gate_msa), shape=[int(v) for v in x.shape])
        h = _stage("norm2", lambda: block.norm2(x), shape=[int(v) for v in x.shape])
        h = _stage("modulate_mlp", lambda: _modulate_inplace_preserve_order(h, scale_mlp, shift_mlp), shape=[int(v) for v in h.shape])
        mlp_out = _stage("mlp", lambda: _feed_forward_mlp_timed(block.mlp, h, timing_callback=timing_callback), shape=[int(v) for v in h.shape])
        x = _stage("residual_mlp", lambda: _residual_gate_inplace_preserve_order(x, mlp_out, gate_mlp), shape=[int(v) for v in x.shape])
    else:
        x.add_(_stage(
            "attention.total",
            lambda: _rope_self_attention_standard(
                block.attn,
                block.norm1(x),
                rotary_emb,
                backend=attention_backend,
                compute_dtype="model",
                query_chunk_size=128,
                contiguous_qkv=True,
                timing_callback=timing_callback,
            ),
            shape=[int(v) for v in x.shape],
        ))
        x.add_(_stage("mlp", lambda: _feed_forward_mlp_timed(block.mlp, block.norm2(x), timing_callback=timing_callback), shape=[int(v) for v in x.shape]))
    return x


def _unified_block_forward_standard_inplace_compiled_realrope(block, x, mod, repo_layer):
    if block.modulation:
        if not block.share_mod:
            mod = block.adaLN_modulation(mod)
        if hasattr(block, "shift_table") and block.shift_table is not None:
            mod = _native_shift_table_add(block, mod, block.shift_table.type(mod.dtype))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        angles = _repo_angles(repo_layer, x)
        h = block.norm1(x)
        h = _modulate_inplace_preserve_order(h, scale_msa, shift_msa)
        h = _rope_self_attention_compiled_realrope(block.attn, h, angles)
        x = _residual_gate_inplace_preserve_order(x, h, gate_msa)
        h = block.norm2(x)
        h = _modulate_inplace_preserve_order(h, scale_mlp, shift_mlp)
        mlp_out = block.mlp(h)
        x = _residual_gate_inplace_preserve_order(x, mlp_out, gate_mlp)
    else:
        angles = _repo_angles(repo_layer, x)
        h = block.norm1(x)
        x.add_(_rope_self_attention_compiled_realrope(block.attn, h, angles))
        x.add_(block.mlp(block.norm2(x)))
    return x


def _unified_block_forward_with_key_bias(block, x, mod, rotary_emb, key_bias, *, use_addcmul_elementwise: bool = False, use_inplace_elementwise: bool = False, attention_backend: str = "default", timing_callback=None):
    import time

    def _stage(name: str, fn, *, shape=None):
        if timing_callback is None:
            return fn()
        started = time.perf_counter()
        try:
            return fn()
        finally:
            timing_callback(name, time.perf_counter() - started, shape=shape)

    if block.modulation:
        if not block.share_mod:
            mod = block.adaLN_modulation(mod)
        if hasattr(block, "shift_table") and block.shift_table is not None:
            mod = _native_shift_table_add(block, mod, block.shift_table.type(mod.dtype))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        h = _stage("norm1", lambda: block.norm1(x), shape=[int(v) for v in x.shape])
        if use_addcmul_elementwise:
            h = _stage("modulate_msa", lambda: _modulate_addcmul(h, scale_msa, shift_msa), shape=[int(v) for v in h.shape])
        elif use_inplace_elementwise:
            h = _stage("modulate_msa", lambda: _modulate_inplace_preserve_order(h, scale_msa, shift_msa), shape=[int(v) for v in h.shape])
        else:
            h = _stage("modulate_msa", lambda: h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1), shape=[int(v) for v in h.shape])
        h = _stage(
            "attention.total",
            lambda: _rope_self_attention_with_key_bias(block.attn, h, rotary_emb, key_bias, backend=attention_backend, timing_callback=timing_callback),
            shape=[int(v) for v in h.shape],
        )
        if use_addcmul_elementwise:
            x = _stage("residual_msa", lambda: _residual_gate_addcmul(x, h, gate_msa), shape=[int(v) for v in x.shape])
        elif use_inplace_elementwise:
            x = _stage("residual_msa", lambda: _residual_gate_inplace_preserve_order(x, h, gate_msa), shape=[int(v) for v in x.shape])
        else:
            x = _stage("residual_msa", lambda: x + h * gate_msa.unsqueeze(1), shape=[int(v) for v in x.shape])
        h = _stage("norm2", lambda: block.norm2(x), shape=[int(v) for v in x.shape])
        if use_addcmul_elementwise:
            h = _stage("modulate_mlp", lambda: _modulate_addcmul(h, scale_mlp, shift_mlp), shape=[int(v) for v in h.shape])
        elif use_inplace_elementwise:
            h = _stage("modulate_mlp", lambda: _modulate_inplace_preserve_order(h, scale_mlp, shift_mlp), shape=[int(v) for v in h.shape])
        else:
            h = _stage("modulate_mlp", lambda: h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1), shape=[int(v) for v in h.shape])
        mlp_out = _stage("mlp", lambda: _feed_forward_mlp_timed(block.mlp, h, timing_callback=timing_callback), shape=[int(v) for v in h.shape])
        if use_addcmul_elementwise:
            x = _stage("residual_mlp", lambda: _residual_gate_addcmul(x, mlp_out, gate_mlp), shape=[int(v) for v in x.shape])
        elif use_inplace_elementwise:
            x = _stage("residual_mlp", lambda: _residual_gate_inplace_preserve_order(x, mlp_out, gate_mlp), shape=[int(v) for v in x.shape])
        else:
            x = _stage("residual_mlp", lambda: x + mlp_out * gate_mlp.unsqueeze(1), shape=[int(v) for v in x.shape])
    else:
        if use_inplace_elementwise:
            x.add_(_stage("attention.total", lambda: _rope_self_attention_with_key_bias(block.attn, block.norm1(x), rotary_emb, key_bias, backend=attention_backend, timing_callback=timing_callback), shape=[int(v) for v in x.shape]))
            x.add_(_stage("mlp", lambda: _feed_forward_mlp_timed(block.mlp, block.norm2(x), timing_callback=timing_callback), shape=[int(v) for v in x.shape]))
        else:
            x = x + _stage("attention.total", lambda: _rope_self_attention_with_key_bias(block.attn, block.norm1(x), rotary_emb, key_bias, backend=attention_backend, timing_callback=timing_callback), shape=[int(v) for v in x.shape])
            x = x + _stage("mlp", lambda: _feed_forward_mlp_timed(block.mlp, block.norm2(x), timing_callback=timing_callback), shape=[int(v) for v in x.shape])
    return x


def _rope_self_attention_pos_neg_with_key_bias(attn, x_pos, x_neg, rope_pos, rope_neg, key_bias, *, backend: str = "default"):
    import math
    import torch
    import torch.nn.functional as F
    try:
        import model as triposplat_model
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("could not import TripoSplat model module") from exc

    B, Lp, C = x_pos.shape
    Ln = int(x_neg.shape[1])
    H = attn.num_heads
    D = attn.head_dim
    x_cat = torch.cat([x_pos, x_neg], dim=1)
    qkv = attn.qkv(x_cat).reshape(B, Lp + Ln, 3, H, D)
    q, k, v = qkv.unbind(2)
    q_pos, q_neg = q.split([Lp, Ln], dim=1)
    k_pos, k_neg = k.split([Lp, Ln], dim=1)
    v_pos, v_neg = v.split([Lp, Ln], dim=1)
    if attn.use_rope:
        q_pos = triposplat_model.apply_rotary_emb(q_pos, rope_pos)
        k_pos = triposplat_model.apply_rotary_emb(k_pos, rope_pos)
        q_neg = triposplat_model.apply_rotary_emb(q_neg, rope_neg)
        k_neg = triposplat_model.apply_rotary_emb(k_neg, rope_neg)
    if attn.qk_rms_norm:
        q_cat = attn.q_norm(torch.cat([q_pos, q_neg], dim=1))
        k_cat = attn.k_norm(torch.cat([k_pos, k_neg], dim=1))
        q_pos, q_neg = q_cat.split([Lp, Ln], dim=1)
        k_pos, k_neg = k_cat.split([Lp, Ln], dim=1)
    q_pos_bhld = q_pos.permute(0, 2, 1, 3).contiguous()
    k_pos_bhld = k_pos.permute(0, 2, 1, 3).contiguous()
    v_pos_bhld = v_pos.permute(0, 2, 1, 3).contiguous()
    q_neg_bhld = q_neg.permute(0, 2, 1, 3).contiguous()
    k_neg_bhld = k_neg.permute(0, 2, 1, 3).contiguous()
    v_neg_bhld = v_neg.permute(0, 2, 1, 3).contiguous()
    if backend in {"aten_flash_direct", "aten_flash_direct_scale", "aten_flash_direct_auto"}:
        direct_scale_pos = 1.0 / math.sqrt(float(q_pos_bhld.shape[-1])) if backend == "aten_flash_direct_scale" else None
        direct_scale_neg = 1.0 / math.sqrt(float(q_neg_bhld.shape[-1])) if backend == "aten_flash_direct_scale" else None
        h_pos = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(
            q_pos_bhld, k_pos_bhld, v_pos_bhld, 0.0, False, attn_mask=None, scale=direct_scale_pos
        )[0]
        h_neg = _direct_flash_attention_with_key_bias(q_neg_bhld, k_neg_bhld, v_neg_bhld, key_bias, direct_scale=direct_scale_neg)
    elif backend == "f_explicit_scale":
        scale_pos = 1.0 / math.sqrt(float(q_pos_bhld.shape[-1]))
        scale_neg = 1.0 / math.sqrt(float(q_neg_bhld.shape[-1]))
        h_pos = F.scaled_dot_product_attention(q_pos_bhld, k_pos_bhld, v_pos_bhld, scale=scale_pos)
        h_neg = F.scaled_dot_product_attention(q_neg_bhld, k_neg_bhld, v_neg_bhld, attn_mask=key_bias, scale=scale_neg)
    else:
        h_pos = F.scaled_dot_product_attention(q_pos_bhld, k_pos_bhld, v_pos_bhld)
        h_neg = F.scaled_dot_product_attention(q_neg_bhld, k_neg_bhld, v_neg_bhld, attn_mask=key_bias)
    h_pos = h_pos.permute(0, 2, 1, 3).reshape(B, Lp, C)
    h_neg = h_neg.permute(0, 2, 1, 3).reshape(B, Ln, C)
    out = attn.out(torch.cat([h_pos, h_neg], dim=1))
    return out.split([Lp, Ln], dim=1)



def _neg_compressed_to_full(x_neg, condition_token_count: int, q_token_length: int, cam_len: int):
    import torch
    lat = x_neg[:, :q_token_length]
    cond = x_neg[:, q_token_length : q_token_length + 1].expand(-1, int(condition_token_count), -1)
    if cam_len:
        return torch.cat([lat, cond, x_neg[:, -cam_len:]], dim=1)
    return torch.cat([lat, cond], dim=1)


def _neg_full_to_compressed(x_neg_full, q_token_length: int, cam_len: int):
    import torch
    lat = x_neg_full[:, :q_token_length]
    cond = x_neg_full[:, q_token_length : q_token_length + 1]
    if cam_len:
        return torch.cat([lat, cond, x_neg_full[:, -cam_len:]], dim=1)
    return torch.cat([lat, cond], dim=1)


def _compress_qkv_like_negative_full(x, q_token_length: int, cam_len: int):
    import torch
    lat = x[:, :q_token_length]
    cond = x[:, q_token_length : q_token_length + 1]
    if cam_len:
        return torch.cat([lat, cond, x[:, -cam_len:]], dim=1)
    return torch.cat([lat, cond], dim=1)


def _rope_self_attention_full_linear_neg_compressed(attn, x_pos, x_neg_full, rope_pos, rope_neg, key_bias, q_token_length: int, cam_len: int, *, backend: str = "default"):
    import math
    import torch
    import torch.nn.functional as F
    try:
        import model as triposplat_model
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("could not import TripoSplat model module") from exc

    B, Lp, C = x_pos.shape
    H = attn.num_heads
    D = attn.head_dim
    qkv = attn.qkv(torch.cat([x_pos, x_neg_full], dim=0)).reshape(2 * B, Lp, 3, H, D)
    q_pos, k_pos, v_pos = qkv[:B].unbind(2)
    q_neg_full, k_neg_full, v_neg_full = qkv[B:].unbind(2)
    q_neg = _compress_qkv_like_negative_full(q_neg_full, q_token_length, cam_len)
    k_neg = _compress_qkv_like_negative_full(k_neg_full, q_token_length, cam_len)
    v_neg = _compress_qkv_like_negative_full(v_neg_full, q_token_length, cam_len)
    if attn.use_rope:
        q_pos = triposplat_model.apply_rotary_emb(q_pos, rope_pos)
        k_pos = triposplat_model.apply_rotary_emb(k_pos, rope_pos)
        q_neg = triposplat_model.apply_rotary_emb(q_neg, rope_neg)
        k_neg = triposplat_model.apply_rotary_emb(k_neg, rope_neg)
    if attn.qk_rms_norm:
        q_pos = attn.q_norm(q_pos)
        k_pos = attn.k_norm(k_pos)
        q_neg = attn.q_norm(q_neg)
        k_neg = attn.k_norm(k_neg)
    q_pos_bhld = q_pos.permute(0, 2, 1, 3).contiguous()
    k_pos_bhld = k_pos.permute(0, 2, 1, 3).contiguous()
    v_pos_bhld = v_pos.permute(0, 2, 1, 3).contiguous()
    q_neg_bhld = q_neg.permute(0, 2, 1, 3).contiguous()
    k_neg_bhld = k_neg.permute(0, 2, 1, 3).contiguous()
    v_neg_bhld = v_neg.permute(0, 2, 1, 3).contiguous()
    if backend in {"aten_flash_direct", "aten_flash_direct_scale", "aten_flash_direct_auto"}:
        direct_scale_pos = 1.0 / math.sqrt(float(q_pos_bhld.shape[-1])) if backend == "aten_flash_direct_scale" else None
        direct_scale_neg = 1.0 / math.sqrt(float(q_neg_bhld.shape[-1])) if backend == "aten_flash_direct_scale" else None
        h_pos = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(
            q_pos_bhld, k_pos_bhld, v_pos_bhld, 0.0, False, attn_mask=None, scale=direct_scale_pos
        )[0]
        h_neg = _direct_flash_attention_with_key_bias(q_neg_bhld, k_neg_bhld, v_neg_bhld, key_bias, direct_scale=direct_scale_neg)
    elif backend == "f_explicit_scale":
        scale_pos = 1.0 / math.sqrt(float(q_pos_bhld.shape[-1]))
        scale_neg = 1.0 / math.sqrt(float(q_neg_bhld.shape[-1]))
        h_pos = F.scaled_dot_product_attention(q_pos_bhld, k_pos_bhld, v_pos_bhld, scale=scale_pos)
        h_neg = F.scaled_dot_product_attention(q_neg_bhld, k_neg_bhld, v_neg_bhld, attn_mask=key_bias, scale=scale_neg)
    else:
        h_pos = F.scaled_dot_product_attention(q_pos_bhld, k_pos_bhld, v_pos_bhld)
        h_neg = F.scaled_dot_product_attention(q_neg_bhld, k_neg_bhld, v_neg_bhld, attn_mask=key_bias)
    h_pos = h_pos.permute(0, 2, 1, 3).reshape(B, Lp, C)
    h_neg = h_neg.permute(0, 2, 1, 3).reshape(B, q_neg.shape[1], C)
    h_neg_full = _neg_compressed_to_full(h_neg, Lp - q_token_length - cam_len, q_token_length, cam_len)
    out = attn.out(torch.cat([h_pos, h_neg_full], dim=0))
    out_pos, out_neg_full = out[:B], out[B:]
    return out_pos, out_neg_full


def _unified_block_forward_full_linear_neg_compressed(block, x_pos, x_neg, mod, rope_pos, rope_neg, key_bias, condition_token_count: int, q_token_length: int, cam_len: int, *, attention_backend: str = "default"):
    import torch
    x_neg_full = _neg_compressed_to_full(x_neg, condition_token_count, q_token_length, cam_len)
    x_cat = torch.cat([x_pos, x_neg_full], dim=0)
    if block.modulation:
        if block.share_mod:
            mod_work = mod.repeat(2, *([1] * (mod.dim() - 1)))
        else:
            mod_in = mod.repeat(2, *([1] * (mod.dim() - 1)))
            mod_work = block.adaLN_modulation(mod_in)
        if hasattr(block, "shift_table") and block.shift_table is not None:
            mod_work = _native_shift_table_add(block, mod_work, block.shift_table.type(mod_work.dtype))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod_work.chunk(6, dim=1)
        h_cat = block.norm1(x_cat)
        h_cat = h_cat * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h_pos, h_neg_full = h_cat[:1], h_cat[1:]
        attn_pos, attn_neg_full = _rope_self_attention_full_linear_neg_compressed(
            block.attn,
            h_pos,
            h_neg_full,
            rope_pos,
            rope_neg,
            key_bias,
            q_token_length,
            cam_len,
            backend=attention_backend,
        )
        attn_cat = torch.cat([attn_pos, attn_neg_full], dim=0)
        x_cat = x_cat + attn_cat * gate_msa.unsqueeze(1)
        h2_cat = block.norm2(x_cat)
        h2_cat = h2_cat * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x_cat = x_cat + block.mlp(h2_cat) * gate_mlp.unsqueeze(1)
    else:
        h_cat = block.norm1(x_cat)
        h_pos, h_neg_full = h_cat[:1], h_cat[1:]
        attn_pos, attn_neg_full = _rope_self_attention_full_linear_neg_compressed(
            block.attn,
            h_pos,
            h_neg_full,
            rope_pos,
            rope_neg,
            key_bias,
            q_token_length,
            cam_len,
            backend=attention_backend,
        )
        x_cat = x_cat + torch.cat([attn_pos, attn_neg_full], dim=0)
        x_cat = x_cat + block.mlp(block.norm2(x_cat))
    return x_cat[:1], _neg_full_to_compressed(x_cat[1:], q_token_length, cam_len)

def _unified_block_forward_pos_neg_with_key_bias(block, x_pos, x_neg, mod, rope_pos, rope_neg, key_bias, *, attention_backend: str = "default"):
    if block.modulation:
        if not block.share_mod:
            mod = block.adaLN_modulation(mod)
        if hasattr(block, "shift_table") and block.shift_table is not None:
            mod = _native_shift_table_add(block, mod, block.shift_table.type(mod.dtype))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        lp = int(x_pos.shape[1])
        ln = int(x_neg.shape[1])
        h_cat = block.norm1(__import__("torch").cat([x_pos, x_neg], dim=1))
        h_pos, h_neg = h_cat.split([lp, ln], dim=1)
        h_pos = h_pos * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h_neg = h_neg * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h_pos, h_neg = _rope_self_attention_pos_neg_with_key_bias(block.attn, h_pos, h_neg, rope_pos, rope_neg, key_bias, backend=attention_backend)
        x_pos = x_pos + h_pos * gate_msa.unsqueeze(1)
        x_neg = x_neg + h_neg * gate_msa.unsqueeze(1)
        h2_cat = block.norm2(__import__("torch").cat([x_pos, x_neg], dim=1))
        h2_pos, h2_neg = h2_cat.split([lp, ln], dim=1)
        h2_pos = h2_pos * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h2_neg = h2_neg * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_pos, mlp_neg = block.mlp(__import__("torch").cat([h2_pos, h2_neg], dim=1)).split([lp, ln], dim=1)
        x_pos = x_pos + mlp_pos * gate_mlp.unsqueeze(1)
        x_neg = x_neg + mlp_neg * gate_mlp.unsqueeze(1)
    else:
        lp = int(x_pos.shape[1])
        ln = int(x_neg.shape[1])
        h_cat = block.norm1(__import__("torch").cat([x_pos, x_neg], dim=1))
        h_pos, h_neg = h_cat.split([lp, ln], dim=1)
        h_pos, h_neg = _rope_self_attention_pos_neg_with_key_bias(block.attn, h_pos, h_neg, rope_pos, rope_neg, key_bias, backend=attention_backend)
        x_pos = x_pos + h_pos
        x_neg = x_neg + h_neg
        h2_cat = block.norm2(__import__("torch").cat([x_pos, x_neg], dim=1))
        mlp_pos, mlp_neg = block.mlp(h2_cat).split([lp, ln], dim=1)
        x_pos = x_pos + mlp_pos
        x_neg = x_neg + mlp_neg
    return x_pos, x_neg


def _selected_latent_camera_idx(length: int, q_token_length: int, cam_len: int, device):
    import torch
    parts = [torch.arange(0, int(q_token_length), device=device, dtype=torch.long)]
    if int(cam_len):
        parts.append(torch.arange(int(length) - int(cam_len), int(length), device=device, dtype=torch.long))
    return torch.cat(parts)


def _unified_block_forward_selected_rows_only(block, x, mod, rotary_emb, selected_idx, key_bias=None, *, use_addcmul_elementwise: bool = False, use_inplace_elementwise: bool = False, attention_backend: str = "default", timing_callback=None):
    """Return only final-block rows consumed by TripoSplat outputs.

    This is exact for the last main block because downstream code only reads
    latent rows and, when present, camera rows. K/V attention still sees every
    input row; only the unused condition query/out/MLP rows are skipped.
    """
    import time

    def _stage(name: str, fn, *, shape=None):
        if timing_callback is None:
            return fn()
        started = time.perf_counter()
        try:
            return fn()
        finally:
            timing_callback(name, time.perf_counter() - started, shape=shape)

    if block.modulation:
        if not block.share_mod:
            mod = block.adaLN_modulation(mod)
        if hasattr(block, "shift_table") and block.shift_table is not None:
            mod = _native_shift_table_add(block, mod, block.shift_table.type(mod.dtype))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        h = _stage("norm1", lambda: block.norm1(x), shape=[int(v) for v in x.shape])
        if use_addcmul_elementwise:
            h = _stage("modulate_msa", lambda: _modulate_addcmul(h, scale_msa, shift_msa), shape=[int(v) for v in h.shape])
        elif use_inplace_elementwise:
            h = _stage("modulate_msa", lambda: _modulate_inplace_preserve_order(h, scale_msa, shift_msa), shape=[int(v) for v in h.shape])
        else:
            h = _stage("modulate_msa", lambda: h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1), shape=[int(v) for v in h.shape])
        h_sel = _stage(
            "attention.total",
            lambda: _rope_self_attention_selected_rows(
                block.attn,
                h,
                selected_idx,
                rotary_emb,
                backend=attention_backend,
                compute_dtype="model",
                query_chunk_size=128,
                key_bias=key_bias,
                timing_callback=timing_callback,
            ),
            shape=[int(v) for v in h.shape],
        )
        x_base_sel = _stage(
            "selected_base_index",
            lambda: x.index_select(1, selected_idx),
            shape=[int(x.shape[0]), int(selected_idx.numel()), int(x.shape[-1])],
        )
        if use_addcmul_elementwise:
            x_sel = _stage("residual_msa", lambda: _residual_gate_addcmul(x_base_sel, h_sel, gate_msa), shape=[int(v) for v in x_base_sel.shape])
        elif use_inplace_elementwise:
            x_sel = _stage("residual_msa", lambda: _residual_gate_inplace_preserve_order(x_base_sel, h_sel, gate_msa), shape=[int(v) for v in x_base_sel.shape])
        else:
            x_sel = _stage("residual_msa", lambda: x_base_sel + h_sel * gate_msa.unsqueeze(1), shape=[int(v) for v in x_base_sel.shape])
        h2 = _stage("norm2", lambda: block.norm2(x_sel), shape=[int(v) for v in x_sel.shape])
        if use_addcmul_elementwise:
            h2 = _stage("modulate_mlp", lambda: _modulate_addcmul(h2, scale_mlp, shift_mlp), shape=[int(v) for v in h2.shape])
        elif use_inplace_elementwise:
            h2 = _stage("modulate_mlp", lambda: _modulate_inplace_preserve_order(h2, scale_mlp, shift_mlp), shape=[int(v) for v in h2.shape])
        else:
            h2 = _stage("modulate_mlp", lambda: h2 * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1), shape=[int(v) for v in h2.shape])
        mlp_out = _stage("mlp", lambda: _feed_forward_mlp_timed(block.mlp, h2, timing_callback=timing_callback), shape=[int(v) for v in h2.shape])
        if use_addcmul_elementwise:
            return _stage("residual_mlp", lambda: _residual_gate_addcmul(x_sel, mlp_out, gate_mlp), shape=[int(v) for v in x_sel.shape])
        if use_inplace_elementwise:
            return _stage("residual_mlp", lambda: _residual_gate_inplace_preserve_order(x_sel, mlp_out, gate_mlp), shape=[int(v) for v in x_sel.shape])
        return _stage("residual_mlp", lambda: x_sel + mlp_out * gate_mlp.unsqueeze(1), shape=[int(v) for v in x_sel.shape])
    h_sel = _stage(
        "attention.total",
        lambda: _rope_self_attention_selected_rows(
            block.attn,
            block.norm1(x),
            selected_idx,
            rotary_emb,
            backend=attention_backend,
            compute_dtype="model",
            query_chunk_size=128,
            key_bias=key_bias,
            timing_callback=timing_callback,
        ),
        shape=[int(v) for v in x.shape],
    )
    x_sel = _stage("selected_base_index", lambda: x.index_select(1, selected_idx), shape=[int(x.shape[0]), int(selected_idx.numel()), int(x.shape[-1])]) + h_sel
    return x_sel + _stage("mlp", lambda: _feed_forward_mlp_timed(block.mlp, block.norm2(x_sel), timing_callback=timing_callback), shape=[int(v) for v in x_sel.shape])


def apply_triposplat_negative_condition_compression_patch(
    flow_model,
    *,
    enabled: bool = False,
    condition_token_count: int | None = None,
    require_static_condition_cache: bool = True,
    verify_negative_rows_identical: bool = True,
    combine_linear_blocks: bool = False,
    full_linear_compressed_sdpa_blocks: bool = False,
    selective_final_block: bool = False,
    selective_final_negative_branch: bool = True,
    addcmul_elementwise: bool = False,
    inplace_elementwise: bool = False,
    noise_refiner_inplace_elementwise: bool = False,
    positive_compiled_realrope: bool = False,
    positive_fullblock_compiled_realrope: bool = False,
    parallel_branches: bool = False,
    parallel_branch_workers: int = 2,
    attention_backend: str = "default",
    logbias_lse_adjust: bool = False,
    internal_timing: bool = False,
) -> dict[str, Any]:
    """Experimental exact-style CFG patch for repeated negative condition rows.

    The patch targets the runner-controlled CFG shape [positive, negative]. It
    keeps the positive branch as the normal full sequence and carries the
    negative branch as latent + one representative condition row + camera. The
    representative condition key receives an additive log(M) SDPA bias, which is
    algebraically equivalent to M identical condition keys/values.
    """
    if not enabled:
        return {"enabled": False}
    if not hasattr(flow_model, "blocks") or not hasattr(flow_model, "repo_layers"):
        raise ValueError("flow_model does not look like LatentSeqMMFlowModel")
    if positive_compiled_realrope and positive_fullblock_compiled_realrope:
        raise ValueError("choose only one positive compiled real-RoPE path")
    if positive_compiled_realrope and not inplace_elementwise:
        raise ValueError("positive compiled real-RoPE path currently requires inplace_elementwise=True to match the current-best positive block order")

    import math
    import time
    import types
    from concurrent.futures import ThreadPoolExecutor
    import torch
    import torch.nn.functional as F

    cache_key = "_triposplat_cached_h_cond"
    q_token_length = int(flow_model.q_token_length)
    cam_len = int(1 if getattr(flow_model, "cam_channels", None) is not None else 0)
    configured_condition_tokens = None if condition_token_count is None else int(condition_token_count)
    key_bias_cache: dict[tuple[str, int | None, int, int, int], torch.Tensor] = {}
    selected_idx_cache: dict[tuple[str, int | None, int, int, int], torch.Tensor] = {}
    timing_stats: dict[str, Any] = {
        "enabled": bool(internal_timing),
        "kind": "negative_condition_compression_internal_timing",
        "forward_calls": 0,
        "fallback_calls": 0,
        "events": {},
        "blocks": {},
        "note": "Inclusive wall-clock timers inside the patched CFG compression path. These catch main blocks that normal nn.Module forward hooks can miss.",
    }

    def _shape_list(value):
        shape = getattr(value, "shape", None)
        if shape is None:
            return None
        return [int(part) for part in shape]

    def _record_timing(event: str, elapsed_sec: float, *, block_index=None, branch=None, mode=None, shape=None) -> None:
        if not internal_timing:
            return
        elapsed = float(elapsed_sec)
        row = timing_stats["events"].setdefault(
            event,
            {"calls": 0, "total_sec": 0.0, "max_sec": 0.0, "last_sec": 0.0},
        )
        row["calls"] += 1
        row["total_sec"] += elapsed
        row["max_sec"] = max(float(row["max_sec"]), elapsed)
        row["last_sec"] = elapsed
        if shape is not None:
            row["last_shape"] = shape
        if block_index is None:
            return
        block_key = f"blocks.{int(block_index)}"
        block_row = timing_stats["blocks"].setdefault(
            block_key,
            {"calls": 0, "total_sec": 0.0, "max_sec": 0.0, "branches": {}},
        )
        block_row["calls"] += 1
        block_row["total_sec"] += elapsed
        block_row["max_sec"] = max(float(block_row["max_sec"]), elapsed)
        branch_key = branch or "unknown"
        if mode:
            branch_key = f"{branch_key}:{mode}"
        branch_row = block_row["branches"].setdefault(
            branch_key,
            {"calls": 0, "total_sec": 0.0, "max_sec": 0.0, "last_sec": 0.0},
        )
        branch_row["calls"] += 1
        branch_row["total_sec"] += elapsed
        branch_row["max_sec"] = max(float(branch_row["max_sec"]), elapsed)
        branch_row["last_sec"] = elapsed
        if shape is not None:
            branch_row["last_shape"] = shape

    def _record_stage(stage: str, elapsed_sec: float, *, block_index=None, branch=None, shape=None) -> None:
        if not internal_timing:
            return
        elapsed = float(elapsed_sec)
        event_key = f"stage.{stage}"
        row = timing_stats["events"].setdefault(
            event_key,
            {"calls": 0, "total_sec": 0.0, "max_sec": 0.0, "last_sec": 0.0},
        )
        row["calls"] += 1
        row["total_sec"] += elapsed
        row["max_sec"] = max(float(row["max_sec"]), elapsed)
        row["last_sec"] = elapsed
        if shape is not None:
            row["last_shape"] = shape
        if block_index is None:
            return
        block_key = f"blocks.{int(block_index)}"
        block_row = timing_stats["blocks"].setdefault(
            block_key,
            {"calls": 0, "total_sec": 0.0, "max_sec": 0.0, "branches": {}},
        )
        stages = block_row.setdefault("stages", {})
        stage_row = stages.setdefault(
            stage,
            {"calls": 0, "total_sec": 0.0, "max_sec": 0.0, "last_sec": 0.0, "branches": {}},
        )
        stage_row["calls"] += 1
        stage_row["total_sec"] += elapsed
        stage_row["max_sec"] = max(float(stage_row["max_sec"]), elapsed)
        stage_row["last_sec"] = elapsed
        if shape is not None:
            stage_row["last_shape"] = shape
        branch_key = branch or "unknown"
        branch_row = stage_row["branches"].setdefault(
            branch_key,
            {"calls": 0, "total_sec": 0.0, "max_sec": 0.0, "last_sec": 0.0},
        )
        branch_row["calls"] += 1
        branch_row["total_sec"] += elapsed
        branch_row["max_sec"] = max(float(branch_row["max_sec"]), elapsed)
        branch_row["last_sec"] = elapsed
        if shape is not None:
            branch_row["last_shape"] = shape

    def _make_stage_recorder(block_index: int, branch: str):
        if not internal_timing:
            return None

        def record(stage: str, elapsed_sec: float, *, shape=None):
            _record_stage(stage, elapsed_sec, block_index=block_index, branch=branch, shape=shape)

        return record

    def _timed_stage_value(stage: str, fn, *, block_index: int, branch: str, shape=None):
        if not internal_timing:
            return fn()
        started = time.perf_counter()
        try:
            return fn()
        finally:
            _record_stage(stage, time.perf_counter() - started, block_index=block_index, branch=branch, shape=shape)

    def _device_key(device):
        return (device.type, None if device.index is None else int(device.index))

    def _cached_key_bias(device, neg_len: int, cond_tokens: int, neg_cond_index: int):
        key = (*_device_key(device), int(neg_len), int(cond_tokens), int(neg_cond_index))
        cached = key_bias_cache.get(key)
        if cached is None:
            cached = torch.zeros(1, 1, 1, int(neg_len), device=device, dtype=torch.float32)
            cached[..., int(neg_cond_index)] = math.log(float(cond_tokens))
            cached._triposplat_logbias_lse_adjust = bool(logbias_lse_adjust)
            cached._triposplat_logbias_index = int(neg_cond_index)
            cached._triposplat_logbias_multiplicity = int(cond_tokens)
            key_bias_cache[key] = cached
        return cached

    def _cached_selected_idx(length: int, device):
        key = (*_device_key(device), int(length), q_token_length, cam_len)
        cached = selected_idx_cache.get(key)
        if cached is None:
            cached = _selected_latent_camera_idx(int(length), q_token_length, cam_len, device)
            selected_idx_cache[key] = cached
        return cached

    def _fallback(self, x_t, t, cond):
        if internal_timing:
            timing_stats["fallback_calls"] += 1
        return self._original_forward_negative_condition_compression(x_t, t, cond)

    def patched_forward(self, x_t, t, cond):
        forward_started = time.perf_counter() if internal_timing else None
        if internal_timing:
            timing_stats["forward_calls"] += 1
        if require_static_condition_cache and cache_key not in cond:
            return _fallback(self, x_t, t, cond)
        if cache_key not in cond:
            return _fallback(self, x_t, t, cond)
        if not torch.is_tensor(t) or int(t.shape[0]) != 2:
            return _fallback(self, x_t, t, cond)
        if not torch.equal(t[:1], t[1:]):
            return _fallback(self, x_t, t, cond)
        z_full_in = x_t["latent"]
        if int(z_full_in.shape[0]) != 2 or not torch.equal(z_full_in[:1], z_full_in[1:]):
            return _fallback(self, x_t, t, cond)
        if self.cam_channels is not None:
            cam_full_in = x_t.get("camera")
            if cam_full_in is None or int(cam_full_in.shape[0]) != 2 or not torch.equal(cam_full_in[:1], cam_full_in[1:]):
                return _fallback(self, x_t, t, cond)

        d = self.dtype
        z = z_full_in[:1].to(d)
        h_cond_pair = cond[cache_key].to(device=z.device, dtype=d)
        if int(h_cond_pair.shape[0]) != 2:
            return _fallback(self, x_t, t, cond)
        cond_tokens = int(h_cond_pair.shape[1])
        if configured_condition_tokens is not None and cond_tokens != configured_condition_tokens:
            return _fallback(self, x_t, t, cond)
        h_cond_pos = h_cond_pair[:1]
        h_cond_neg_full = h_cond_pair[1:2]
        h_cond_neg = h_cond_neg_full[:, :1]
        if verify_negative_rows_identical and not torch.equal(h_cond_neg_full, h_cond_neg.expand_as(h_cond_neg_full)):
            return _fallback(self, x_t, t, cond)

        self.pos_pe = self.pos_pe.to(z.device)
        h_x = self.input_layer(z)
        t_emb = self.t_embedder(t[:1])
        t_mod = self.adaLN_modulation(t_emb) if self.share_mod else t_emb
        h_x = _native_add_tensor_inplace(self, h_x, self.pos_embedder(self.pos_pe).to(d))
        for i, block in enumerate(self.noise_refiner):
            if noise_refiner_inplace_elementwise:
                h_x = _unified_block_forward_standard_inplace(
                    block,
                    h_x,
                    t_mod,
                    self.noise_repo_layers[i](h_x),
                    attention_backend=attention_backend,
                )
            else:
                h_x = block(h_x, mod=t_mod, rotary_emb=self.noise_repo_layers[i](h_x))
        if self.cam_channels is not None:
            cam = cam_full_in[:1].to(d)
            h_cam = self.cam_refiner(cam)
        else:
            cam = None
            h_cam = None

        h_pos = torch.cat([h_x, h_cond_pos], dim=1)
        h_neg = torch.cat([h_x, h_cond_neg], dim=1)
        if self.cam_channels is not None:
            h_pos = torch.cat([h_pos, h_cam], dim=1)
            h_neg = torch.cat([h_neg, h_cam], dim=1)

        neg_cond_index = q_token_length
        neg_len = int(h_neg.shape[1])
        key_bias = _cached_key_bias(h_neg.device, neg_len, cond_tokens, neg_cond_index)

        def _run_pos_block(block, h_value, block_index):
            with torch.inference_mode():
                if positive_fullblock_compiled_realrope:
                    return _unified_block_forward_standard_inplace_fullblock_compiled_realrope(
                        block,
                        h_value,
                        t_mod,
                        self.repo_layers[block_index],
                    )
                if positive_compiled_realrope:
                    return _unified_block_forward_standard_inplace_compiled_realrope(
                        block,
                        h_value,
                        t_mod,
                        self.repo_layers[block_index],
                    )
                if inplace_elementwise:
                    return _unified_block_forward_standard_inplace(
                        block,
                        h_value,
                        t_mod,
                        _timed_stage_value("repo_layer", lambda: self.repo_layers[block_index](h_value), block_index=block_index, branch="positive", shape=_shape_list(h_value)),
                        attention_backend=attention_backend,
                        timing_callback=_make_stage_recorder(block_index, "positive"),
                    )
                return block(h_value, mod=t_mod, rotary_emb=self.repo_layers[block_index](h_value))

        def _run_neg_block(block, h_value, block_index):
            with torch.inference_mode():
                if os.environ.get("TRIPOSPLAT_NEG_FULLBLOCK_COMPILED_REALROPE", "").strip().lower() in {"1", "true", "yes", "on"}:
                    return _unified_block_forward_with_key_bias_fullblock_compiled_realrope(
                        block,
                        h_value,
                        t_mod,
                        self.repo_layers[block_index],
                        key_bias,
                    )
                return _unified_block_forward_with_key_bias(
                    block,
                    h_value,
                    t_mod,
                    _timed_stage_value("repo_layer", lambda: self.repo_layers[block_index](h_value), block_index=block_index, branch="negative", shape=_shape_list(h_value)),
                    key_bias,
                    use_addcmul_elementwise=bool(addcmul_elementwise),
                    use_inplace_elementwise=bool(inplace_elementwise),
                    attention_backend=attention_backend,
                    timing_callback=_make_stage_recorder(block_index, "negative"),
                )

        def _run_pos_selected(block, h_value, block_index, selected_idx):
            with torch.inference_mode():
                if os.environ.get("TRIPOSPLAT_POS_FINAL_SELECTED_COMPILED_REALROPE", "").strip().lower() in {"1", "true", "yes", "on"}:
                    return _unified_block_forward_selected_rows_only_compiled_realrope(
                        block,
                        h_value,
                        t_mod,
                        self.repo_layers[block_index],
                        selected_idx,
                    )
                return _unified_block_forward_selected_rows_only(
                    block,
                    h_value,
                    t_mod,
                    _timed_stage_value("repo_layer", lambda: self.repo_layers[block_index](h_value), block_index=block_index, branch="positive", shape=_shape_list(h_value)),
                    selected_idx,
                    key_bias=None,
                    use_addcmul_elementwise=bool(addcmul_elementwise),
                    use_inplace_elementwise=bool(inplace_elementwise),
                    attention_backend=attention_backend,
                    timing_callback=_make_stage_recorder(block_index, "positive"),
                )

        def _run_neg_selected(block, h_value, block_index, selected_idx):
            with torch.inference_mode():
                return _unified_block_forward_selected_rows_only(
                    block,
                    h_value,
                    t_mod,
                    _timed_stage_value("repo_layer", lambda: self.repo_layers[block_index](h_value), block_index=block_index, branch="negative", shape=_shape_list(h_value)),
                    selected_idx,
                    key_bias=key_bias,
                    use_addcmul_elementwise=bool(addcmul_elementwise),
                    use_inplace_elementwise=bool(inplace_elementwise),
                    attention_backend=attention_backend,
                    timing_callback=_make_stage_recorder(block_index, "negative"),
                )

        last_block_index = len(self.blocks) - 1
        combined_linear_blocks = 0
        final_pos_selected = None
        final_neg_selected = None
        executor = ThreadPoolExecutor(max_workers=max(2, int(parallel_branch_workers))) if parallel_branches else None
        try:
            for i, block in enumerate(self.blocks):
                block_started = time.perf_counter() if internal_timing else None
                block_start_shape = {"pos": _shape_list(h_pos), "neg": _shape_list(h_neg)}
                block_mode = "unknown"
                if selective_final_block and i == last_block_index:
                    block_mode = "final_selective"
                    pos_selected_idx = _cached_selected_idx(int(h_pos.shape[1]), h_pos.device)
                    if parallel_branches and executor is not None:
                        future_pos = executor.submit(_run_pos_selected, block, h_pos, i, pos_selected_idx)
                        if selective_final_negative_branch:
                            neg_selected_idx = _cached_selected_idx(int(h_neg.shape[1]), h_neg.device)
                            future_neg = executor.submit(_run_neg_selected, block, h_neg, i, neg_selected_idx)
                            final_pos_selected = future_pos.result()
                            final_neg_selected = future_neg.result()
                        else:
                            future_neg = executor.submit(_run_neg_block, block, h_neg, i)
                            final_pos_selected = future_pos.result()
                            h_neg = future_neg.result()
                    else:
                        final_pos_selected = _run_pos_selected(block, h_pos, i, pos_selected_idx)
                        if selective_final_negative_branch:
                            neg_selected_idx = _cached_selected_idx(int(h_neg.shape[1]), h_neg.device)
                            final_neg_selected = _run_neg_selected(block, h_neg, i, neg_selected_idx)
                        else:
                            h_neg = _run_neg_block(block, h_neg, i)
                    if internal_timing and block_started is not None:
                        _record_timing("main_block", time.perf_counter() - block_started, block_index=i, branch="pos_neg", mode=block_mode, shape=block_start_shape)
                    continue
                if full_linear_compressed_sdpa_blocks and i < last_block_index:
                    block_mode = "full_linear_compressed_sdpa"
                    h_pos, h_neg = _unified_block_forward_full_linear_neg_compressed(
                        block,
                        h_pos,
                        h_neg,
                        t_mod,
                        self.repo_layers[i](h_pos),
                        self.repo_layers[i](h_neg),
                        key_bias,
                        cond_tokens,
                        q_token_length,
                        cam_len,
                        attention_backend=attention_backend,
                    )
                elif combine_linear_blocks and i < last_block_index:
                    block_mode = "combine_linear"
                    h_pos, h_neg = _unified_block_forward_pos_neg_with_key_bias(
                        block,
                        h_pos,
                        h_neg,
                        t_mod,
                        self.repo_layers[i](h_pos),
                        self.repo_layers[i](h_neg),
                        key_bias,
                        attention_backend=attention_backend,
                    )
                    combined_linear_blocks += 1
                elif parallel_branches and executor is not None:
                    block_mode = "parallel_branches"
                    future_pos = executor.submit(_run_pos_block, block, h_pos, i)
                    future_neg = executor.submit(_run_neg_block, block, h_neg, i)
                    h_pos = future_pos.result()
                    h_neg = future_neg.result()
                else:
                    block_mode = "sequential_pos_neg_inplace" if inplace_elementwise else "sequential_pos_neg_module"
                    h_pos = _run_pos_block(block, h_pos, i)
                    h_neg = _run_neg_block(block, h_neg, i)
                if internal_timing and block_started is not None:
                    _record_timing("main_block", time.perf_counter() - block_started, block_index=i, branch="pos_neg", mode=block_mode, shape=block_start_shape)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        if final_pos_selected is not None:
            norm_shape = final_pos_selected.shape[-1:]
            h_x_pos_raw = final_pos_selected[:, : z.shape[1]]
            h_x_pos = _native_functional_layer_norm(self, h_x_pos_raw)
            if final_neg_selected is not None:
                h_x_neg_raw = final_neg_selected[:, : z.shape[1]]
                h_x_neg = _native_functional_layer_norm(self, h_x_neg_raw)
            else:
                h_x_neg = _native_functional_layer_norm(self, h_neg[:, : z.shape[1]])
            if self.cam_channels is not None:
                h_cam_pos_raw = final_pos_selected[:, -cam.shape[1] :]
                h_cam_pos = _native_functional_layer_norm(self, h_cam_pos_raw)
                if final_neg_selected is not None:
                    h_cam_neg_raw = final_neg_selected[:, -cam.shape[1] :]
                    h_cam_neg = _native_functional_layer_norm(self, h_cam_neg_raw)
                else:
                    h_cam_neg = _native_functional_layer_norm(self, h_neg[:, -cam.shape[1] :])
        else:
            h_x_pos = _native_functional_layer_norm(self, h_pos[:, : z.shape[1]])
            h_x_neg = _native_functional_layer_norm(self, h_neg[:, : z.shape[1]])
            if self.cam_channels is not None:
                h_cam_pos = _native_functional_layer_norm(self, h_pos[:, -cam.shape[1] :])
                h_cam_neg = _native_functional_layer_norm(self, h_neg[:, -cam.shape[1] :])

        if self.use_shift_table:
            shift, scale = _native_final_modulation_params(self, self.shift_table, t_emb)
            h_x_pos = _native_modulate_tensor(self, h_x_pos, scale, shift)
            h_x_neg = _native_modulate_tensor(self, h_x_neg, scale, shift)
            if self.cam_channels is not None:
                h_cam_pos = _native_modulate_tensor(self, h_cam_pos, scale, shift)
                h_cam_neg = _native_modulate_tensor(self, h_cam_neg, scale, shift)

        out = {"latent": torch.cat([self.out_layer(h_x_pos), self.out_layer(h_x_neg)], dim=0)}
        if self.cam_channels is not None:
            out["camera"] = torch.cat([self.cam_out_layer(h_cam_pos), self.cam_out_layer(h_cam_neg)], dim=0)
        if internal_timing and forward_started is not None:
            _record_timing("forward_total", time.perf_counter() - forward_started)
        return out

    if not hasattr(flow_model, "_original_forward_negative_condition_compression"):
        flow_model._original_forward_negative_condition_compression = flow_model.forward
    flow_model.forward = types.MethodType(patched_forward, flow_model)
    return {
        "enabled": True,
        "kind": "negative_cfg_repeated_condition_token_compression_patch",
        "cache_key": cache_key,
        "latent_length": q_token_length,
        "camera_length": cam_len,
        "condition_token_count": configured_condition_tokens,
        "require_static_condition_cache": bool(require_static_condition_cache),
        "verify_negative_rows_identical": bool(verify_negative_rows_identical),
        "combine_linear_blocks": bool(combine_linear_blocks),
        "full_linear_compressed_sdpa_blocks": bool(full_linear_compressed_sdpa_blocks),
        "selective_final_block": bool(selective_final_block),
        "selective_final_negative_branch": bool(selective_final_negative_branch),
        "addcmul_elementwise": bool(addcmul_elementwise),
        "inplace_elementwise": bool(inplace_elementwise),
        "noise_refiner_inplace_elementwise": bool(noise_refiner_inplace_elementwise),
        "positive_compiled_realrope": bool(positive_compiled_realrope),
        "positive_fullblock_compiled_realrope": bool(positive_fullblock_compiled_realrope),
        "positive_standard_attention_detail_timing": bool(internal_timing and not positive_compiled_realrope and not positive_fullblock_compiled_realrope),
        "mlp_detail_timing": bool(internal_timing),
        "mlp_detail_timing_path": "FeedForwardNet-style Sequential(Linear, GELU(tanh), Linear) is timed as mlp.fc1/mlp.gelu/mlp.fc2 when no alternative MLP patch is active",
        "positive_standard_attention_path": "expanded self-attention qkv/rope/qk_norm/layout/sdpa/out_projection for the standard positive branch",
        "positive_final_selected_compiled_realrope": os.environ.get("TRIPOSPLAT_POS_FINAL_SELECTED_COMPILED_REALROPE", "").strip().lower() in {"1", "true", "yes", "on"},
        "negative_fullblock_compiled_realrope": os.environ.get("TRIPOSPLAT_NEG_FULLBLOCK_COMPILED_REALROPE", "").strip().lower() in {"1", "true", "yes", "on"},
        "positive_compiled_realrope_mode": os.environ.get("TRIPOSPLAT_POS_COMPILED_REALROPE_MODE", "reduce-overhead"),
        "positive_fullblock_compiled_realrope_mode": os.environ.get("TRIPOSPLAT_POS_FULLBLOCK_COMPILED_REALROPE_MODE", os.environ.get("TRIPOSPLAT_POS_COMPILED_REALROPE_MODE", "reduce-overhead")),
        "positive_final_selected_compiled_realrope_mode": os.environ.get("TRIPOSPLAT_POS_FINAL_SELECTED_COMPILED_REALROPE_MODE", "reduce-overhead"),
        "negative_fullblock_compiled_realrope_mode": os.environ.get("TRIPOSPLAT_NEG_FULLBLOCK_COMPILED_REALROPE_MODE", "none"),
        "negative_fullblock_compiled_realrope_scope": "negative compact latent + representative condition + camera full main blocks with key-bias SDPA",
        "positive_final_selected_compiled_realrope_scope": "positive final selected latent/camera rows only; key-bias negative branch remains on existing path",
        "positive_compiled_realrope_scope": "positive full main blocks only; final selective block and negative branch remain on existing path",
        "positive_fullblock_compiled_realrope_scope": "positive full main blocks only; final selective block and negative branch remain on existing path; compiles LayerNorm/modulation/attention/MLP/residual gate with official RePo input position",
        "key_bias_cache": True,
        "selected_idx_cache": True,
        "parallel_branches": bool(parallel_branches),
        "parallel_branch_workers": int(parallel_branch_workers),
        "attention_backend": attention_backend,
        "logbias_lse_adjust": bool(logbias_lse_adjust),
        "internal_timing": timing_stats,
        "logbias_lse_adjust_scope": "direct aten CPU flash only; replaces single-key log(M) mask with maskless SDPA plus logsumexp probability correction",
        "parallel_branch_scope": "positive/negative CFG branches inside compressed negative-condition main blocks; default off and intended for multi-physical-core CPU hosts",
        "selective_final_positive_selected_rows_contiguous_qkv": True,
        "semantics": "Experimental CFG-only path. Positive branch stays full length; negative branch keeps one representative repeated condition row and applies log(M) key bias in SDPA.",
        "fallback": "Falls back to the previous flow_model.forward when runner-controlled duplicated CFG/static negative condition assumptions are not met.",
        "selective_final_block_integration": bool(selective_final_block),
        "combined_linear_blocks": max(0, len(flow_model.blocks) - 1) if combine_linear_blocks else 0,
        "full_linear_compressed_sdpa_blocks_count": max(0, len(flow_model.blocks) - 1) if full_linear_compressed_sdpa_blocks else 0,
        "combined_linear_scope": "disabled by default; combined Linear experiment was slower than separate positive/negative block calls" if not combine_linear_blocks else "blocks.0 through penultimate main block; final block keeps existing selective-final positive path",
        "full_linear_compressed_sdpa_scope": "disabled by default" if not full_linear_compressed_sdpa_blocks else "blocks.0 through penultimate main block use full B=2 Linear/MLP with compressed negative SDPA",
    }
