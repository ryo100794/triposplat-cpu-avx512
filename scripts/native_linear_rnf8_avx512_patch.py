from __future__ import annotations

import ctypes
import math
import re
import time
import types
from pathlib import Path
from typing import Any

from native_linear_nf8_avx512_patch import make_nf8_codebook


def quantize_nf24_per_output_channel(weight, codebook):
    import torch

    if weight.device.type != "cpu" or weight.dtype != torch.float32 or weight.ndim != 2:
        raise ValueError("NF24 packing requires a CPU float32 2D weight")
    source = weight.detach().contiguous()
    tiny = torch.finfo(torch.float32).tiny
    max_abs = source.abs().amax(dim=1).clamp_min(tiny)
    base_scale = (max_abs / 127.0).contiguous()

    alphabet = torch.unique(torch.round(codebook * 127.0).clamp_(-127, 127)).to(torch.float32)
    midpoints = ((alphabet[:-1] + alphabet[1:]) * 0.5).contiguous()
    q0 = alphabet[torch.bucketize(source / base_scale[:, None], midpoints)].to(torch.int8)
    residual1 = source - q0.to(torch.float32) * base_scale[:, None]

    residual1_scale = base_scale / 4.0
    q1_unclamped = torch.floor(residual1 / residual1_scale[:, None] + 0.5)
    q1 = q1_unclamped.clamp(-128, 127).to(torch.int8)
    residual2 = residual1 - q1.to(torch.float32) * residual1_scale[:, None]

    final_scale = (base_scale / 1024.0).contiguous()
    q2_unclamped = torch.floor(residual2 / final_scale[:, None] + 0.5)
    q2 = q2_unclamped.clamp(-128, 127).to(torch.int8)
    residual3 = residual2 - q2.to(torch.float32) * final_scale[:, None]

    stage_tensors = (residual1, residual2, residual3)
    quantizers = (
        "nf8_derived_int8_alphabet",
        "shared_scale_int8_radix4",
        "shared_scale_int8_radix256",
    )
    raw_codes = (q0, q1, q2)
    raw_unclamped = (None, q1_unclamped, q2_unclamped)
    stage_errors = []
    for stage, (remaining, quantizer, unclamped) in enumerate(
        zip(stage_tensors, quantizers, raw_unclamped), start=1
    ):
        stage_errors.append(
            {
                "stage": stage,
                "quantizer": quantizer,
                "residual_rmse": float(torch.sqrt(torch.mean(remaining * remaining)).item()),
                "residual_max_abs": float(remaining.abs().max().item()),
                "saturated_values": 0 if unclamped is None else int(((unclamped < -128) | (unclamped > 127)).sum().item()),
            }
        )

    weight_rms = float(torch.sqrt(torch.mean(source * source)).item())
    error = {
        "weight_rmse": stage_errors[-1]["residual_rmse"],
        "weight_max_abs": stage_errors[-1]["residual_max_abs"],
        "weight_rms": weight_rms,
        "weight_relative_rmse": stage_errors[-1]["residual_rmse"] / max(weight_rms, 1.0e-30),
        "residual_mode": "nf24",
        "nf8_derived_alphabet_size": int(alphabet.numel()),
        "stages": stage_errors,
    }
    codes_t = [codes.t().contiguous() for codes in raw_codes]
    return codes_t, [final_scale, final_scale, final_scale], error


def quantize_nf24_i16_per_output_channel(weight, codebook):
    codes, scales, error = quantize_nf24_per_output_channel(weight, codebook)
    import torch

    code01_t = (
        codes[0].to(torch.int16) * 4 + codes[1].to(torch.int16)
    ).contiguous()
    packed_error = {
        **error,
        "residual_mode": "nf24_i16",
        "storage_layout": "int16(q0*4+q1) + int8(q2)",
    }
    return [code01_t, codes[2], codes[2]], scales, packed_error


def quantize_rnf8_per_output_channel(weight, codebook, stages: int, residual_mode: str = "nf8"):
    import torch

    if residual_mode == "nf24_i16":
        if stages != 3:
            raise ValueError("NF24 int16 layout requires exactly 3 stages")
        return quantize_nf24_i16_per_output_channel(weight, codebook)
    if stages not in (2, 3):
        raise ValueError("residual NF8 stages must be 2 or 3")
    if residual_mode not in ("nf8", "symmetric_int8", "nf24_i16"):
        raise ValueError(f"unsupported residual quantizer: {residual_mode}")
    if weight.device.type != "cpu" or weight.dtype != torch.float32 or weight.ndim != 2:
        raise ValueError("residual NF8 packing requires a CPU float32 2D weight")
    source = weight.detach().contiguous()
    residual = source.clone()
    midpoints = ((codebook[:-1] + codebook[1:]) * 0.5).contiguous()
    codes_t = []
    scales = []
    stage_errors = []
    tiny = torch.finfo(torch.float32).tiny
    for stage in range(stages):
        max_abs = residual.abs().amax(dim=1).clamp_min(tiny)
        if stage == 0 or residual_mode == "nf8":
            scale = max_abs.contiguous()
            codes = torch.bucketize(residual / scale[:, None], midpoints).to(torch.uint8)
            approximation = codebook[codes.long()] * scale[:, None]
            quantizer = "nf8"
        else:
            scale = (max_abs / 127.0).contiguous()
            codes = torch.round(residual / scale[:, None]).clamp_(-127, 127).to(torch.int8)
            approximation = codes.to(torch.float32) * scale[:, None]
            quantizer = "symmetric_int8"
        residual = residual - approximation
        codes_t.append(codes.t().contiguous())
        scales.append(scale)
        stage_errors.append(
            {
                "stage": stage + 1,
                "quantizer": quantizer,
                "residual_rmse": float(torch.sqrt(torch.mean(residual * residual)).item()),
                "residual_max_abs": float(residual.abs().max().item()),
            }
        )
    weight_rms = float(torch.sqrt(torch.mean(source * source)).item())
    error = {
        "weight_rmse": stage_errors[-1]["residual_rmse"],
        "weight_max_abs": stage_errors[-1]["residual_max_abs"],
        "weight_rms": weight_rms,
        "weight_relative_rmse": stage_errors[-1]["residual_rmse"] / max(weight_rms, 1.0e-30),
        "residual_mode": residual_mode,
        "stages": stage_errors,
    }
    return codes_t, scales, error


def apply_triposplat_native_rnf8_avx512_patch(
    flow_model,
    *,
    enabled: bool = False,
    include_regex: str | None = None,
    exclude_regex: str | None = None,
    library_path: str = "artifacts/backends/libtriposplat_gemm_rnf8_avx512.so",
    optimized_library_path: str | None = None,
    optimized_shapes: tuple[tuple[int, int], ...] = (),
    threads: int = 2,
    strict: bool = True,
    stages: int = 2,
    residual_mode: str = "nf8",
) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False}
    if not strict:
        raise ValueError("residual NF8 releases float32 weights and requires strict=True")
    if stages not in (2, 3):
        raise ValueError("residual NF8 stages must be 2 or 3")
    if residual_mode not in ("nf8", "symmetric_int8", "nf24_i16"):
        raise ValueError(f"unsupported residual quantizer: {residual_mode}")
    if residual_mode == "nf24_i16" and stages != 3:
        raise ValueError("NF24 int16 layout requires exactly 3 stages")

    import torch
    import torch.nn as nn

    include = re.compile(include_regex) if include_regex else None
    exclude = re.compile(exclude_regex) if exclude_regex else None
    lib_path = Path(library_path)
    if not lib_path.is_file():
        raise FileNotFoundError(lib_path)
    lib = ctypes.CDLL(lib_path.as_posix())
    row_tile_fn = getattr(lib, "triposplat_gemm_rnf8_avx512_row_tile", None)
    if row_tile_fn is None:
        row_tile = None
    else:
        row_tile_fn.argtypes = []
        row_tile_fn.restype = ctypes.c_int
        row_tile = int(row_tile_fn())
    residual_mode_fn = getattr(lib, "triposplat_gemm_rnf8_avx512_residual_mode", None)
    if residual_mode_fn is None:
        library_residual_mode = "nf8"
    else:
        residual_mode_fn.argtypes = []
        residual_mode_fn.restype = ctypes.c_int
        mode_id = int(residual_mode_fn())
        library_residual_mode = {0: "nf8", 1: "symmetric_int8", 4: "nf24_i16"}.get(mode_id)
        if library_residual_mode is None:
            raise RuntimeError(f"unsupported residual mode id from library: {mode_id}")
    if library_residual_mode != residual_mode:
        raise RuntimeError(
            f"residual quantizer/library mismatch: requested={residual_mode}, "
            f"library={library_residual_mode}"
        )
    pointer_count = 10
    full_kernel = lib.triposplat_gemm_rnf8_avx512
    full_kernel.argtypes = [ctypes.c_void_p] * pointer_count + [ctypes.c_int] * 7
    full_kernel.restype = ctypes.c_int
    tail_kernel = lib.triposplat_gemm_rnf8_avx512_tail
    tail_kernel.argtypes = [ctypes.c_void_p] * pointer_count + [ctypes.c_int] * 7
    tail_kernel.restype = ctypes.c_int
    range_kernel = lib.triposplat_gemm_rnf8_avx512_range
    range_kernel.argtypes = [ctypes.c_void_p] * pointer_count + [ctypes.c_int] * 9
    range_kernel.restype = ctypes.c_int

    optimized_shapes_set = {tuple(int(value) for value in shape) for shape in optimized_shapes}
    optimized_full_kernel = None
    optimized_lib_path = None
    optimized_lib = None
    if optimized_library_path is not None:
        optimized_lib_path = Path(optimized_library_path)
        if not optimized_lib_path.is_file():
            raise FileNotFoundError(optimized_lib_path)
        optimized_lib = ctypes.CDLL(optimized_lib_path.as_posix())
        optimized_mode_fn = getattr(optimized_lib, "triposplat_gemm_rnf8_avx512_residual_mode", None)
        if optimized_mode_fn is None:
            optimized_mode = "nf8"
        else:
            optimized_mode_fn.argtypes = []
            optimized_mode_fn.restype = ctypes.c_int
            optimized_mode = {0: "nf8", 1: "symmetric_int8", 4: "nf24_i16"}.get(int(optimized_mode_fn()))
        if optimized_mode != residual_mode:
            raise RuntimeError(
                f"optimized residual quantizer/library mismatch: requested={residual_mode}, "
                f"library={optimized_mode}"
            )
        optimized_full_kernel = optimized_lib.triposplat_gemm_rnf8_avx512
        optimized_full_kernel.argtypes = [ctypes.c_void_p] * pointer_count + [ctypes.c_int] * 7
        optimized_full_kernel.restype = ctypes.c_int

    runtime = {
        "calls": 0,
        "range_calls": 0,
        "optimized_calls": 0,
        "rows": 0,
        "seconds": 0.0,
        "contiguous_copies": 0,
        "fallbacks": 0,
        "per_module": {},
    }
    selected = []
    selected_dims = {}
    skipped = []
    packing = {}
    codebook = make_nf8_codebook(torch)
    prepacked_manifest = getattr(flow_model, "_native_rnf8_prepacked_manifest", None)
    if prepacked_manifest is not None and (stages != 3 or residual_mode != "nf24_i16"):
        raise ValueError("prepacked NF24 int16 requires stages=3 and residual_mode=nf24_i16")
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
            raise RuntimeError(f"residual NF8 strict violation for {module_name}: grad_enabled")
        if x.device.type != "cpu" or x.dtype != torch.float32:
            raise RuntimeError(f"residual NF8 strict violation for {module_name}: input_{x.device.type}_{x.dtype}")
        if int(x.shape[-1]) != int(module.in_features):
            raise RuntimeError(f"residual NF8 strict violation for {module_name}: input_feature_mismatch")
        x2 = x.reshape(-1, int(module.in_features))
        if not x2.is_contiguous():
            x2 = x2.contiguous()
            runtime["contiguous_copies"] += 1
        return x2

    def pointers(module):
        codes2 = module._native_rnf8_codes2_t if stages == 3 else module._native_rnf8_codes1_t
        scales2 = module._native_rnf8_scales2 if stages == 3 else module._native_rnf8_scales1
        return (
            module._native_rnf8_codes0_t.data_ptr(),
            module._native_rnf8_codes1_t.data_ptr(),
            codes2.data_ptr(),
            module._native_rnf8_scales0.data_ptr(),
            module._native_rnf8_scales1.data_ptr(),
            scales2.data_ptr(),
            module._native_rnf8_codebook.data_ptr(),
            module._native_rnf8_bias.data_ptr(),
        )

    def record(name: str, rows: int, elapsed: float, is_range: bool, optimized: bool = False):
        runtime["calls"] += 1
        runtime["range_calls"] += int(is_range)
        runtime["optimized_calls"] += int(optimized)
        runtime["rows"] += rows
        runtime["seconds"] += elapsed
        item = runtime["per_module"].setdefault(
            name, {"calls": 0, "range_calls": 0, "optimized_calls": 0, "rows": 0, "seconds": 0.0}
        )
        item["calls"] += 1
        item["range_calls"] += int(is_range)
        item["optimized_calls"] += int(optimized)
        item["rows"] += rows
        item["seconds"] += elapsed

    def make_forward(module_name: str):
        def forward(self, x):
            x2 = prepare_input(self, x, module_name)
            rows = int(x2.shape[0])
            out = torch.empty((rows, int(self.out_features)), dtype=torch.float32)
            code0, code1, code2, scale0, scale1, scale2, codebook_ptr, bias_ptr = pointers(self)
            shape = (int(self.in_features), int(self.out_features))
            use_optimized = optimized_full_kernel is not None and shape in optimized_shapes_set and shape[1] % 16 == 0
            kernel = optimized_full_kernel if use_optimized else (tail_kernel if shape[1] % 16 else full_kernel)
            started = time.perf_counter()
            status = int(
                kernel(
                    x2.data_ptr(), code0, code1, code2, scale0, scale1, scale2,
                    codebook_ptr, bias_ptr, out.data_ptr(), rows, int(self.in_features),
                    int(self.out_features), int(self.in_features), int(self.out_features),
                    int(threads), int(stages),
                )
            )
            elapsed = time.perf_counter() - started
            if status != 0:
                runtime["fallbacks"] += 1
                raise RuntimeError(f"residual NF8 kernel returned {status} for {module_name}")
            record(module_name, rows, elapsed, False, use_optimized)
            return out.view(*x.shape[:-1], int(self.out_features))

        return forward

    def make_range_forward(module_name: str):
        def range_forward(self, x, output_start: int, output_count: int):
            start, count = int(output_start), int(output_count)
            if start < 0 or count <= 0 or start + count > int(self.out_features):
                raise ValueError(f"invalid residual NF8 output range for {module_name}")
            x2 = prepare_input(self, x, module_name)
            rows = int(x2.shape[0])
            out = torch.empty((rows, count), dtype=torch.float32)
            code0, code1, code2, scale0, scale1, scale2, codebook_ptr, bias_ptr = pointers(self)
            started = time.perf_counter()
            status = int(
                range_kernel(
                    x2.data_ptr(), code0, code1, code2, scale0, scale1, scale2,
                    codebook_ptr, bias_ptr, out.data_ptr(), rows, int(self.in_features),
                    int(self.out_features), start, count, int(self.in_features), count,
                    int(threads), int(stages),
                )
            )
            elapsed = time.perf_counter() - started
            if status != 0:
                runtime["fallbacks"] += 1
                raise RuntimeError(f"residual NF8 range kernel returned {status} for {module_name}")
            record(module_name, rows, elapsed, True)
            return out.view(*x.shape[:-1], count)

        return range_forward

    for name, module in flow_model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not matches(name):
            skipped.append(name)
            continue
        if prepacked_manifest is None:
            if module.weight.device.type != "cpu" or module.weight.dtype != torch.float32:
                raise ValueError(f"residual NF8 requires CPU float32 weight: {name}")
            codes, scales, error = quantize_rnf8_per_output_channel(
                module.weight.detach(), codebook, stages, residual_mode
            )
            bias = (
                torch.zeros(int(module.out_features), dtype=torch.float32)
                if module.bias is None
                else module.bias.detach().contiguous()
            )
            module.register_buffer("_native_rnf8_codes0_t", codes[0], persistent=False)
            module.register_buffer("_native_rnf8_codes1_t", codes[1], persistent=False)
            module.register_buffer("_native_rnf8_scales0", scales[0], persistent=False)
            module.register_buffer("_native_rnf8_scales1", scales[1], persistent=False)
            if stages == 3:
                module.register_buffer("_native_rnf8_codes2_t", codes[2], persistent=False)
                module.register_buffer("_native_rnf8_scales2", scales[2], persistent=False)
            module.register_buffer("_native_rnf8_bias", bias, persistent=False)
        else:
            item = prepacked_manifest["linears"].get(name)
            if item is None:
                raise ValueError(f"prepacked checkpoint has no Linear entry: {name}")
            codes = [
                module._native_rnf8_codes0_t,
                module._native_rnf8_codes1_t,
                module._native_rnf8_codes2_t,
            ]
            scales = [
                module._native_rnf8_scales0,
                module._native_rnf8_scales1,
                module._native_rnf8_scales2,
            ]
            bias = module._native_rnf8_bias
            error = item["quantization"]
        module.register_buffer("_native_rnf8_codebook", codebook.clone(), persistent=False)
        module.weight = nn.Parameter(torch.empty(0, dtype=torch.float32), requires_grad=False)
        if module.bias is not None:
            module.bias = nn.Parameter(torch.empty(0, dtype=torch.float32), requires_grad=False)
        module.forward = types.MethodType(make_forward(name), module)
        module._native_avx512_forward_range = types.MethodType(make_range_forward(name), module)

        count = int(codes[0].numel())
        original = count * 4 + int(bias.numel()) * 4
        if residual_mode == "nf24_i16":
            packed = count * 3 + int(scales[0].numel()) * 4 + int(bias.numel()) * 4 + 1024
            bits_per_weight = 24
        else:
            packed = count * stages + int(scales[0].numel()) * 4 * stages + int(bias.numel()) * 4 + 1024
            bits_per_weight = stages * 8
        original_bytes += original
        packed_bytes += packed
        weight_elements += count
        weighted_squared_error += error["weight_rmse"] ** 2 * count
        max_abs_error = max(max_abs_error, error["weight_max_abs"])
        packing[name] = {**error, "weight_elements": count, "original_bytes": original, "packed_bytes": packed}
        selected.append(name)
        selected_dims[name] = f"{int(module.in_features)}x{int(module.out_features)}"

    if skipped:
        raise ValueError(f"residual NF8 strict mode left {len(skipped)} Linear modules unquantized")
    return {
        "enabled": bool(selected),
        "kind": "native_residual_nonlinear_nf8_weight_only_avx512_linear_patch",
        "residual_stages": int(stages),
        "first_stage_quantizer": (
            "nf8_derived_int8_alphabet"
            if residual_mode == "nf24_i16"
            else "nf8"
        ),
        "residual_quantizer": residual_mode,
        "bits_per_weight": int(bits_per_weight),
        "activation_dtype": "float32",
        "float32_weight_retained": False,
        "library_path": lib_path.as_posix(),
        "optimized_library_path": None if optimized_lib_path is None else optimized_lib_path.as_posix(),
        "optimized_shapes": [list(shape) for shape in sorted(optimized_shapes_set)],
        "row_tile": row_tile,
        "symbols": ["triposplat_gemm_rnf8_avx512", "triposplat_gemm_rnf8_avx512_range", "triposplat_gemm_rnf8_avx512_tail", "triposplat_gemm_rnf8_avx512_row_tile", "triposplat_gemm_rnf8_avx512_residual_mode"],
        "threads": int(threads),
        "strict": True,
        "prepacked_load": getattr(flow_model, "_native_rnf8_prepacked_load", {"enabled": False}),
        "selected_count": len(selected),
        "selected": selected,
        "selected_dims": selected_dims,
        "skipped_count": 0,
        "original_bytes": original_bytes,
        "packed_bytes": packed_bytes,
        "storage_ratio": packed_bytes / max(original_bytes, 1),
        "aggregate_weight_rmse": math.sqrt(weighted_squared_error / max(weight_elements, 1)),
        "aggregate_weight_max_abs": max_abs_error,
        "packing": packing,
        "runtime": runtime,
        "semantics": "All selected Linear weights are decoded directly inside AVX-512 GEMM; stage 1 is nonlinear NF8 and its residual uses residual_quantizer.",
    }
