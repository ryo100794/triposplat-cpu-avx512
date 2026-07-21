from __future__ import annotations

import ctypes
import re
import time
import types
from pathlib import Path
from typing import Any


def apply_triposplat_native_gelu_avx512_patch(
    flow_model,
    *,
    enabled: bool = False,
    include_regex: str | None = None,
    exclude_regex: str | None = None,
    library_path: str = "artifacts/backends/libtriposplat_gelu_avx512.so",
    threads: int = 2,
    strict: bool = True,
) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False}

    import torch
    import torch.nn as nn

    include = re.compile(include_regex) if include_regex else None
    exclude = re.compile(exclude_regex) if exclude_regex else None
    lib_path = Path(library_path)
    if not lib_path.exists():
        raise FileNotFoundError(f"native AVX-512 GELU library not found: {lib_path}")
    lib = ctypes.CDLL(lib_path.as_posix())
    kernel = lib.triposplat_gelu_tanh_f32_avx512
    kernel.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int]
    kernel.restype = ctypes.c_int
    runtime = {"calls": 0, "elements": 0, "seconds": 0.0, "fallbacks": 0, "fallback_reasons": {}, "per_module": {}}
    selected = []
    skipped = []

    def matches(name: str) -> bool:
        return not ((include is not None and include.search(name) is None) or (exclude is not None and exclude.search(name) is not None))

    def make_forward(module_name: str):
        def patched_forward(self, x):
            reason = None
            if torch.is_grad_enabled():
                reason = "grad_enabled"
            elif x.device.type != "cpu" or x.dtype != torch.float32:
                reason = f"input_{x.device.type}_{x.dtype}"
            elif getattr(self, "approximate", "none") != "tanh":
                reason = f"approximate_{getattr(self, 'approximate', 'none')}"
            if reason is not None:
                runtime["fallbacks"] += 1
                runtime["fallback_reasons"][reason] = int(runtime["fallback_reasons"].get(reason, 0)) + 1
                if strict:
                    raise RuntimeError(f"native AVX-512 GELU strict violation for {module_name}: {reason}")
                return self._original_forward_native_gelu_avx512(x)
            source = x if x.is_contiguous() else x.contiguous()
            out = torch.empty_like(source)
            started = time.perf_counter()
            status = int(kernel(source.data_ptr(), out.data_ptr(), source.numel(), int(threads)))
            elapsed = time.perf_counter() - started
            if status != 0:
                raise RuntimeError(f"native AVX-512 GELU returned {status} for {module_name}")
            runtime["calls"] += 1
            runtime["elements"] += int(source.numel())
            runtime["seconds"] += elapsed
            item = runtime["per_module"].setdefault(module_name, {"calls": 0, "elements": 0, "seconds": 0.0})
            item["calls"] += 1
            item["elements"] += int(source.numel())
            item["seconds"] += elapsed
            return out.view_as(x)
        return patched_forward

    for name, module in flow_model.named_modules():
        if not isinstance(module, nn.GELU):
            continue
        if not matches(name):
            skipped.append(name)
            continue
        if getattr(module, "approximate", "none") != "tanh":
            if strict:
                raise ValueError(f"native AVX-512 GELU requires approximate=tanh: {name}")
            skipped.append(name)
            continue
        module._original_forward_native_gelu_avx512 = module.forward
        module.forward = types.MethodType(make_forward(name), module)
        selected.append(name)
    if strict and skipped:
        raise ValueError(f"native AVX-512 GELU strict mode left {len(skipped)} GELU modules unpatched")
    return {
        "enabled": bool(selected), "kind": "native_f32_avx512_gelu_tanh_patch",
        "library_path": lib_path.as_posix(), "symbol": "triposplat_gelu_tanh_f32_avx512",
        "threads": int(threads), "strict": bool(strict), "selected_count": len(selected),
        "selected": selected, "skipped_count": len(skipped), "skipped": skipped, "runtime": runtime,
    }
