from __future__ import annotations

import ctypes
import re
import time
import types
from pathlib import Path
from typing import Any


def apply_triposplat_native_linear_avx512_patch(
    flow_model,
    *,
    enabled: bool = False,
    include_regex: str | None = None,
    exclude_regex: str | None = None,
    library_path: str = "artifacts/backends/libtriposplat_gemm_f32_avx512.so",
    threads: int = 2,
    strict: bool = True,
) -> dict[str, Any]:
    """Replace selected CPU float32 Linear calls with full/range AVX-512 kernels."""
    if not enabled:
        return {"enabled": False}

    import torch
    import torch.nn as nn

    include = re.compile(include_regex) if include_regex else None
    exclude = re.compile(exclude_regex) if exclude_regex else None
    lib_path = Path(library_path)
    if not lib_path.exists():
        raise FileNotFoundError(f"native AVX-512 Linear library not found: {lib_path}")
    lib = ctypes.CDLL(lib_path.as_posix())
    full_kernel = lib.triposplat_gemm_f32_avx512
    full_kernel.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 6
    full_kernel.restype = ctypes.c_int
    range_kernel = lib.triposplat_gemm_f32_avx512_range
    range_kernel.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 8
    range_kernel.restype = ctypes.c_int

    runtime = {
        "calls": 0, "range_calls": 0, "rows": 0, "seconds": 0.0,
        "contiguous_copies": 0, "fallbacks": 0, "fallback_reasons": {}, "per_module": {},
    }
    selected = []
    selected_dims = {}
    skipped = []

    def matches(name: str) -> bool:
        return not ((include is not None and include.search(name) is None) or (exclude is not None and exclude.search(name) is not None))

    def fallback_or_raise(module, x, module_name: str, reason: str):
        runtime["fallbacks"] += 1
        reasons = runtime["fallback_reasons"]
        reasons[reason] = int(reasons.get(reason, 0)) + 1
        if strict:
            raise RuntimeError(f"native AVX-512 Linear strict violation for {module_name}: {reason}")
        return module._original_forward_native_avx512(x)

    def prepare_input(module, x, module_name: str):
        if torch.is_grad_enabled():
            return fallback_or_raise(module, x, module_name, "grad_enabled")
        if x.device.type != "cpu" or x.dtype != torch.float32:
            return fallback_or_raise(module, x, module_name, f"input_{x.device.type}_{x.dtype}")
        if int(x.shape[-1]) != int(module.in_features):
            return fallback_or_raise(module, x, module_name, "input_feature_mismatch")
        x2 = x.reshape(-1, int(module.in_features))
        if not x2.is_contiguous():
            x2 = x2.contiguous()
            runtime["contiguous_copies"] += 1
        return x2

    def record(module_name: str, rows: int, elapsed: float, *, is_range: bool):
        runtime["calls"] += 1
        runtime["range_calls"] += int(is_range)
        runtime["rows"] += rows
        runtime["seconds"] += elapsed
        item = runtime["per_module"].setdefault(module_name, {"calls": 0, "range_calls": 0, "rows": 0, "seconds": 0.0})
        item["calls"] += 1
        item["range_calls"] += int(is_range)
        item["rows"] += rows
        item["seconds"] += elapsed

    def make_forward(module_name: str):
        def patched_forward(self, x):
            x2 = prepare_input(self, x, module_name)
            if not torch.is_tensor(x2):
                return x2
            rows = int(x2.shape[0])
            out = torch.empty((rows, int(self.out_features)), dtype=torch.float32, device=x.device)
            started = time.perf_counter()
            status = int(full_kernel(
                x2.data_ptr(), self._native_avx512_weight_t.data_ptr(), self._native_avx512_bias.data_ptr(), out.data_ptr(),
                rows, int(self.in_features), int(self.out_features), int(self.in_features), int(self.out_features), int(threads),
            ))
            elapsed = time.perf_counter() - started
            if status != 0:
                return fallback_or_raise(self, x, module_name, f"kernel_status_{status}")
            record(module_name, rows, elapsed, is_range=False)
            return out.view(*x.shape[:-1], int(self.out_features))
        patched_forward.__name__ = f"native_avx512_linear_forward_{module_name.replace('.', '_')}"
        return patched_forward

    def make_range_forward(module_name: str):
        def range_forward(self, x, output_start: int, output_count: int):
            start = int(output_start)
            count = int(output_count)
            if start < 0 or count <= 0 or start + count > int(self.out_features):
                raise ValueError(f"invalid native output range for {module_name}: start={start}, count={count}")
            x2 = prepare_input(self, x, module_name)
            if not torch.is_tensor(x2):
                return x2
            rows = int(x2.shape[0])
            out = torch.empty((rows, count), dtype=torch.float32, device=x.device)
            started = time.perf_counter()
            status = int(range_kernel(
                x2.data_ptr(), self._native_avx512_weight_t.data_ptr(), self._native_avx512_bias.data_ptr(), out.data_ptr(),
                rows, int(self.in_features), int(self.out_features), start, count,
                int(self.in_features), count, int(threads),
            ))
            elapsed = time.perf_counter() - started
            if status != 0:
                raise RuntimeError(f"native AVX-512 range kernel returned {status} for {module_name}")
            record(module_name, rows, elapsed, is_range=True)
            return out.view(*x.shape[:-1], count)
        return range_forward

    for name, module in flow_model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not matches(name):
            skipped.append(name)
            continue
        if module.weight.device.type != "cpu" or module.weight.dtype != torch.float32:
            if strict:
                raise ValueError(f"native AVX-512 Linear requires CPU float32 weight: {name}")
            skipped.append(name)
            continue
        if not hasattr(module, "_original_forward_native_avx512"):
            module._original_forward_native_avx512 = module.forward
        weight_t = module.weight.detach().t().contiguous()
        bias = torch.zeros(int(module.out_features), dtype=torch.float32) if module.bias is None else module.bias.detach().contiguous()
        module.register_buffer("_native_avx512_weight_t", weight_t, persistent=False)
        module.register_buffer("_native_avx512_bias", bias, persistent=False)
        module.weight = nn.Parameter(torch.empty(0, dtype=torch.float32), requires_grad=False)
        if module.bias is not None:
            module.bias = nn.Parameter(torch.empty(0, dtype=torch.float32), requires_grad=False)
        module.forward = types.MethodType(make_forward(name), module)
        module._native_avx512_forward_range = types.MethodType(make_range_forward(name), module)
        selected.append(name)
        selected_dims[name] = f"{int(module.in_features)}x{int(module.out_features)}"

    if strict and skipped:
        raise ValueError(f"native AVX-512 strict mode left {len(skipped)} Linear modules unpatched")
    return {
        "enabled": bool(selected), "kind": "native_f32_avx512_full_and_range_linear_patch",
        "library_path": lib_path.as_posix(),
        "symbols": ["triposplat_gemm_f32_avx512", "triposplat_gemm_f32_avx512_range"],
        "threads": int(threads), "strict": bool(strict), "include_regex": include_regex,
        "exclude_regex": exclude_regex, "selected_count": len(selected), "selected": selected,
        "selected_dims": selected_dims, "skipped_count": len(skipped), "skipped": skipped,
        "runtime": runtime,
        "semantics": "Selected CPU float32 nn.Linear full and output-range projections use packed-weight native AVX-512/FMA; strict mode forbids fallback.",
    }
