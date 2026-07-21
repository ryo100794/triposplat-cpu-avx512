from __future__ import annotations

import ctypes
import math
import re
import time
import types
from pathlib import Path
from typing import Any


def make_nf8_codebook(torch_module):
    """Return a deterministic, symmetric normal-float-like 8-bit codebook."""
    torch = torch_module
    probability = (torch.arange(255, dtype=torch.float64) + 0.5) / 255.0
    levels = math.sqrt(2.0) * torch.erfinv(2.0 * probability - 1.0)
    levels = levels / levels.abs().max()
    # 255 quantiles already contain zero. A duplicate zero fills all 256 codes
    # without introducing a one-sided extra level.
    return torch.sort(torch.cat((levels, torch.zeros(1, dtype=torch.float64))))[0].float().contiguous()


def quantize_nf8_per_output_channel(weight, codebook):
    """Pack [out, in] float32 weights as uint8 codes plus one scale per row."""
    import torch

    if weight.device.type != "cpu" or weight.dtype != torch.float32 or weight.ndim != 2:
        raise ValueError("NF8 packing requires a CPU float32 2D weight")
    source = weight.detach().contiguous()
    scales = source.abs().amax(dim=1).clamp_min(torch.finfo(torch.float32).tiny).contiguous()
    normalized = source / scales[:, None]
    midpoints = ((codebook[:-1] + codebook[1:]) * 0.5).contiguous()
    codes = torch.bucketize(normalized, midpoints).to(torch.uint8)
    reconstructed = codebook[codes.long()] * scales[:, None]
    diff = reconstructed - source
    error = {
        "weight_rmse": float(torch.sqrt(torch.mean(diff * diff)).item()),
        "weight_mae": float(torch.mean(diff.abs()).item()),
        "weight_max_abs": float(diff.abs().max().item()),
        "weight_rms": float(torch.sqrt(torch.mean(source * source)).item()),
    }
    error["weight_relative_rmse"] = error["weight_rmse"] / max(error["weight_rms"], 1.0e-30)
    return codes.t().contiguous(), scales, error


def apply_triposplat_native_nf8_avx512_patch(
    flow_model,
    *,
    enabled: bool = False,
    include_regex: str | None = None,
    exclude_regex: str | None = None,
    library_path: str = "artifacts/backends/libtriposplat_gemm_nf8_avx512.so",
    threads: int = 2,
    strict: bool = True,
) -> dict[str, Any]:
    """Replace selected Linear weights with packed nonlinear NF8 AVX-512 GEMM."""
    if not enabled:
        return {"enabled": False}
    if not strict:
        raise ValueError("NF8 weight release requires strict=True; float32 fallback is intentionally unavailable")

    import torch
    import torch.nn as nn

    include = re.compile(include_regex) if include_regex else None
    exclude = re.compile(exclude_regex) if exclude_regex else None
    lib_path = Path(library_path)
    if not lib_path.exists():
        raise FileNotFoundError(f"native NF8 AVX-512 library not found: {lib_path}")
    lib = ctypes.CDLL(lib_path.as_posix())
    full_kernel = lib.triposplat_gemm_nf8_avx512
    full_kernel.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 6
    full_kernel.restype = ctypes.c_int
    tail_kernel = lib.triposplat_gemm_nf8_avx512_tail
    tail_kernel.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 6
    tail_kernel.restype = ctypes.c_int
    range_kernel = lib.triposplat_gemm_nf8_avx512_range
    range_kernel.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 8
    range_kernel.restype = ctypes.c_int

    runtime = {
        "calls": 0,
        "range_calls": 0,
        "rows": 0,
        "seconds": 0.0,
        "contiguous_copies": 0,
        "fallbacks": 0,
        "per_module": {},
    }
    selected: list[str] = []
    selected_dims: dict[str, str] = {}
    skipped: list[str] = []
    packing: dict[str, Any] = {}
    codebook = make_nf8_codebook(torch)
    original_bytes = 0
    packed_bytes = 0
    weighted_squared_error = 0.0
    weight_elements = 0
    max_abs_error = 0.0

    def matches(name: str) -> bool:
        return not (
            (include is not None and include.search(name) is None)
            or (exclude is not None and exclude.search(name) is not None)
        )

    def prepare_input(module, x, module_name: str):
        if torch.is_grad_enabled():
            raise RuntimeError(f"native NF8 AVX-512 strict violation for {module_name}: grad_enabled")
        if x.device.type != "cpu" or x.dtype != torch.float32:
            raise RuntimeError(
                f"native NF8 AVX-512 strict violation for {module_name}: input_{x.device.type}_{x.dtype}"
            )
        if int(x.shape[-1]) != int(module.in_features):
            raise RuntimeError(f"native NF8 AVX-512 strict violation for {module_name}: input_feature_mismatch")
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
        item = runtime["per_module"].setdefault(
            module_name, {"calls": 0, "range_calls": 0, "rows": 0, "seconds": 0.0}
        )
        item["calls"] += 1
        item["range_calls"] += int(is_range)
        item["rows"] += rows
        item["seconds"] += elapsed

    def make_forward(module_name: str):
        def patched_forward(self, x):
            x2 = prepare_input(self, x, module_name)
            rows = int(x2.shape[0])
            out = torch.empty((rows, int(self.out_features)), dtype=torch.float32, device=x.device)
            started = time.perf_counter()
            status = int(
                (tail_kernel if int(self.out_features) % 16 else full_kernel)(
                    x2.data_ptr(),
                    self._native_nf8_codes_t.data_ptr(),
                    self._native_nf8_scales.data_ptr(),
                    self._native_nf8_codebook.data_ptr(),
                    self._native_nf8_bias.data_ptr(),
                    out.data_ptr(),
                    rows,
                    int(self.in_features),
                    int(self.out_features),
                    int(self.in_features),
                    int(self.out_features),
                    int(threads),
                )
            )
            elapsed = time.perf_counter() - started
            if status != 0:
                runtime["fallbacks"] += 1
                raise RuntimeError(f"native NF8 AVX-512 kernel returned {status} for {module_name}")
            record(module_name, rows, elapsed, is_range=False)
            return out.view(*x.shape[:-1], int(self.out_features))

        return patched_forward

    def make_range_forward(module_name: str):
        def range_forward(self, x, output_start: int, output_count: int):
            start = int(output_start)
            count = int(output_count)
            if start < 0 or count <= 0 or start + count > int(self.out_features):
                raise ValueError(f"invalid NF8 output range for {module_name}: start={start}, count={count}")
            x2 = prepare_input(self, x, module_name)
            rows = int(x2.shape[0])
            out = torch.empty((rows, count), dtype=torch.float32, device=x.device)
            started = time.perf_counter()
            status = int(
                range_kernel(
                    x2.data_ptr(),
                    self._native_nf8_codes_t.data_ptr(),
                    self._native_nf8_scales.data_ptr(),
                    self._native_nf8_codebook.data_ptr(),
                    self._native_nf8_bias.data_ptr(),
                    out.data_ptr(),
                    rows,
                    int(self.in_features),
                    int(self.out_features),
                    start,
                    count,
                    int(self.in_features),
                    count,
                    int(threads),
                )
            )
            elapsed = time.perf_counter() - started
            if status != 0:
                runtime["fallbacks"] += 1
                raise RuntimeError(f"native NF8 AVX-512 range kernel returned {status} for {module_name}")
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
            raise ValueError(f"native NF8 AVX-512 requires CPU float32 weight: {name}")

        weight = module.weight.detach()
        codes_t, scales, error = quantize_nf8_per_output_channel(weight, codebook)
        bias = (
            torch.zeros(int(module.out_features), dtype=torch.float32)
            if module.bias is None
            else module.bias.detach().contiguous()
        )
        module.register_buffer("_native_nf8_codes_t", codes_t, persistent=False)
        module.register_buffer("_native_nf8_scales", scales, persistent=False)
        module.register_buffer("_native_nf8_codebook", codebook.clone(), persistent=False)
        module.register_buffer("_native_nf8_bias", bias, persistent=False)
        module.weight = nn.Parameter(torch.empty(0, dtype=torch.float32), requires_grad=False)
        if module.bias is not None:
            module.bias = nn.Parameter(torch.empty(0, dtype=torch.float32), requires_grad=False)
        module.forward = types.MethodType(make_forward(name), module)
        module._native_avx512_forward_range = types.MethodType(make_range_forward(name), module)

        weight_count = int(codes_t.numel())
        module_original_bytes = weight_count * 4 + int(bias.numel()) * 4
        module_packed_bytes = (
            weight_count
            + int(scales.numel()) * 4
            + int(bias.numel()) * 4
            + int(codebook.numel()) * 4
        )
        original_bytes += module_original_bytes
        packed_bytes += module_packed_bytes
        weighted_squared_error += error["weight_rmse"] ** 2 * weight_count
        weight_elements += weight_count
        max_abs_error = max(max_abs_error, error["weight_max_abs"])
        packing[name] = {
            **error,
            "weight_elements": weight_count,
            "original_bytes": module_original_bytes,
            "packed_bytes": module_packed_bytes,
            "storage_ratio": module_packed_bytes / module_original_bytes,
        }
        selected.append(name)
        selected_dims[name] = f"{int(module.in_features)}x{int(module.out_features)}"

    if skipped:
        raise ValueError(f"native NF8 strict mode left {len(skipped)} Linear modules unquantized")
    aggregate_rmse = math.sqrt(weighted_squared_error / max(weight_elements, 1))
    return {
        "enabled": bool(selected),
        "kind": "native_nonlinear_nf8_weight_only_avx512_linear_patch",
        "quantization": "fixed normal-quantile nonlinear 8-bit codebook with per-output-channel absmax scale",
        "activation_dtype": "float32",
        "weight_storage": "uint8 code, float32 per-output scale, shared-shape 256-entry float32 codebook",
        "float32_weight_retained": False,
        "library_path": lib_path.as_posix(),
        "symbols": ["triposplat_gemm_nf8_avx512", "triposplat_gemm_nf8_avx512_range", "triposplat_gemm_nf8_avx512_tail"],
        "threads": int(threads),
        "strict": True,
        "include_regex": include_regex,
        "exclude_regex": exclude_regex,
        "selected_count": len(selected),
        "selected": selected,
        "selected_dims": selected_dims,
        "skipped_count": 0,
        "original_bytes": original_bytes,
        "packed_bytes": packed_bytes,
        "storage_ratio": packed_bytes / max(original_bytes, 1),
        "aggregate_weight_rmse": aggregate_rmse,
        "aggregate_weight_max_abs": max_abs_error,
        "codebook": codebook.tolist(),
        "packing": packing,
        "runtime": runtime,
        "semantics": "Every selected Linear reads packed nonlinear NF8 codes directly in AVX-512 GEMM; no float32 weight or PyTorch Linear fallback is retained.",
    }
