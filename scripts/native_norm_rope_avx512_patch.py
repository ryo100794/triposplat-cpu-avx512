from __future__ import annotations

import ctypes
import time
import types
from pathlib import Path
from typing import Any


def apply_triposplat_native_norm_rope_avx512_patch(
    flow_model,
    *,
    enabled: bool = False,
    library_path: str = "artifacts/backends/libtriposplat_norm_rope_avx512.so",
    threads: int = 2,
    strict: bool = True,
) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False}

    import torch
    import torch.nn as nn
    import model as triposplat_model

    lib_path = Path(library_path)
    if not lib_path.exists():
        raise FileNotFoundError(f"native AVX-512 norm/RoPE library not found: {lib_path}")
    lib = ctypes.CDLL(lib_path.as_posix())
    layernorm_kernel = lib.triposplat_layernorm_f32_avx512
    layernorm_kernel.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int]
    rmsnorm_kernel = lib.triposplat_multihead_rmsnorm_f32_avx512
    rmsnorm_kernel.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int]
    rope_kernel = lib.triposplat_rope_complex_f32_avx512
    rope_kernel.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64, ctypes.c_int, ctypes.c_int]
    for kernel in (layernorm_kernel, rmsnorm_kernel, rope_kernel):
        kernel.restype = ctypes.c_int

    runtime = {
        "layernorm": {"calls": 0, "rows": 0, "seconds": 0.0, "fallbacks": 0},
        "rmsnorm": {"calls": 0, "vectors": 0, "seconds": 0.0, "fallbacks": 0},
        "rope": {"calls": 0, "vectors": 0, "seconds": 0.0, "fallbacks": 0, "input_copies": 0, "freq_copies": 0},
    }
    layernorm_selected = []
    rmsnorm_selected = []
    skipped = []

    def strict_error(kind: str, name: str, reason: str):
        runtime[kind]["fallbacks"] += 1
        if strict:
            raise RuntimeError(f"native AVX-512 {kind} strict violation for {name}: {reason}")

    def make_layernorm_forward(module_name: str):
        def forward(self, x):
            reason = None
            if torch.is_grad_enabled(): reason = "grad_enabled"
            elif x.device.type != "cpu" or x.dtype != torch.float32: reason = f"input_{x.device.type}_{x.dtype}"
            elif len(self.normalized_shape) != 1 or int(x.shape[-1]) != int(self.normalized_shape[0]): reason = "shape"
            if reason is not None:
                strict_error("layernorm", module_name, reason)
                return self._original_forward_native_norm_rope(x)
            source = x if x.is_contiguous() else x.contiguous()
            cols = int(source.shape[-1]); rows = int(source.numel() // cols)
            out = torch.empty_like(source)
            has_affine = self.weight is not None
            weight_ptr = self.weight.data_ptr() if has_affine else 0
            bias_ptr = self.bias.data_ptr() if has_affine else 0
            started = time.perf_counter()
            status = int(layernorm_kernel(source.data_ptr(), weight_ptr, bias_ptr, out.data_ptr(), rows, cols, float(self.eps), int(has_affine), int(threads)))
            elapsed = time.perf_counter() - started
            if status != 0: raise RuntimeError(f"native LayerNorm returned {status} for {module_name}")
            runtime["layernorm"]["calls"] += 1; runtime["layernorm"]["rows"] += rows; runtime["layernorm"]["seconds"] += elapsed
            return out.view_as(x)
        return forward

    def make_rmsnorm_forward(module_name: str):
        def forward(self, x):
            heads, dim = (int(v) for v in self.gamma.shape)
            reason = None
            if torch.is_grad_enabled(): reason = "grad_enabled"
            elif x.device.type != "cpu" or x.dtype != torch.float32: reason = f"input_{x.device.type}_{x.dtype}"
            elif tuple(int(v) for v in x.shape[-2:]) != (heads, dim): reason = "shape"
            if reason is not None:
                strict_error("rmsnorm", module_name, reason)
                return self._original_forward_native_norm_rope(x)
            source = x if x.is_contiguous() else x.contiguous()
            outer = int(source.numel() // (heads * dim)); out = torch.empty_like(source)
            started = time.perf_counter()
            status = int(rmsnorm_kernel(source.data_ptr(), self.gamma.data_ptr(), out.data_ptr(), outer, heads, dim, 1.0e-12, int(threads)))
            elapsed = time.perf_counter() - started
            if status != 0: raise RuntimeError(f"native RMSNorm returned {status} for {module_name}")
            vectors = outer * heads
            runtime["rmsnorm"]["calls"] += 1; runtime["rmsnorm"]["vectors"] += vectors; runtime["rmsnorm"]["seconds"] += elapsed
            return out.view_as(x)
        return forward

    for name, module in flow_model.named_modules():
        if isinstance(module, nn.LayerNorm):
            if len(module.normalized_shape) != 1:
                skipped.append(name); continue
            module._original_forward_native_norm_rope = module.forward
            module.forward = types.MethodType(make_layernorm_forward(name), module)
            layernorm_selected.append(name)
        elif module.__class__.__name__ == "MultiHeadRMSNorm":
            if not hasattr(module, "gamma") or module.gamma.ndim != 2:
                skipped.append(name); continue
            module._original_forward_native_norm_rope = module.forward
            module.forward = types.MethodType(make_rmsnorm_forward(name), module)
            rmsnorm_selected.append(name)

    if strict and skipped:
        raise ValueError(f"native norm strict mode left {len(skipped)} norm modules unpatched: {skipped[:8]}")

    original_rope = triposplat_model.apply_rotary_emb

    def native_rope(hidden_states, freqs):
        reason = None
        if torch.is_grad_enabled(): reason = "grad_enabled"
        elif hidden_states.device.type != "cpu" or hidden_states.dtype != torch.float32: reason = f"input_{hidden_states.device.type}_{hidden_states.dtype}"
        elif freqs.device.type != "cpu" or freqs.dtype != torch.complex64: reason = f"freq_{freqs.device.type}_{freqs.dtype}"
        elif int(hidden_states.shape[-1]) != 2 * int(freqs.shape[-1]): reason = "shape"
        if reason is not None:
            strict_error("rope", "model.apply_rotary_emb", reason)
            return original_rope(hidden_states, freqs)
        source = hidden_states
        if not source.is_contiguous():
            source = source.contiguous(); runtime["rope"]["input_copies"] += 1
        freq_float = torch.view_as_real(freqs).reshape(*freqs.shape[:-1], int(hidden_states.shape[-1]))
        if not freq_float.is_contiguous():
            freq_float = freq_float.contiguous(); runtime["rope"]["freq_copies"] += 1
        dim = int(source.shape[-1]); vectors = int(source.numel() // dim); out = torch.empty_like(source)
        started = time.perf_counter()
        status = int(rope_kernel(source.data_ptr(), freq_float.data_ptr(), out.data_ptr(), vectors, dim, int(threads)))
        elapsed = time.perf_counter() - started
        if status != 0: raise RuntimeError(f"native RoPE returned {status}")
        runtime["rope"]["calls"] += 1; runtime["rope"]["vectors"] += vectors; runtime["rope"]["seconds"] += elapsed
        return out.view_as(hidden_states)

    def native_functional_layernorm(x, eps=1.0e-5):
        reason = None
        if torch.is_grad_enabled(): reason = "grad_enabled"
        elif x.device.type != "cpu" or x.dtype != torch.float32: reason = f"input_{x.device.type}_{x.dtype}"
        elif x.ndim < 1 or int(x.shape[-1]) <= 0: reason = "shape"
        if reason is not None:
            strict_error("layernorm", "functional_final_layernorm", reason)
            return torch.nn.functional.layer_norm(x.float(), x.shape[-1:], eps=float(eps)).type_as(x)
        source = x if x.is_contiguous() else x.contiguous()
        cols = int(source.shape[-1]); rows = int(source.numel() / cols); out = torch.empty_like(source)
        started = time.perf_counter()
        status = int(layernorm_kernel(source.data_ptr(), 0, 0, out.data_ptr(), rows, cols, float(eps), 0, int(threads)))
        elapsed = time.perf_counter() - started
        if status != 0: raise RuntimeError(f"native functional LayerNorm returned {status}")
        runtime["layernorm"]["calls"] += 1; runtime["layernorm"]["rows"] += rows; runtime["layernorm"]["seconds"] += elapsed
        return out.view_as(x)

    flow_model._native_avx512_functional_layernorm = native_functional_layernorm
    triposplat_model.apply_rotary_emb = native_rope
    return {
        "enabled": True, "kind": "native_f32_avx512_layernorm_rmsnorm_rope_patch",
        "library_path": lib_path.as_posix(),
        "symbols": ["triposplat_layernorm_f32_avx512", "triposplat_multihead_rmsnorm_f32_avx512", "triposplat_rope_complex_f32_avx512"],
        "threads": int(threads), "strict": bool(strict),
        "layernorm_selected_count": len(layernorm_selected), "layernorm_selected": layernorm_selected,
        "rmsnorm_selected_count": len(rmsnorm_selected), "rmsnorm_selected": rmsnorm_selected,
        "skipped_count": len(skipped), "skipped": skipped, "rope_patched": True, "runtime": runtime,
    }
