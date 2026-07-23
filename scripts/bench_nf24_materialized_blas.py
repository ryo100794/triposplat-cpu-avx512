#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import time
from pathlib import Path

import torch

from native_linear_nf24_prepacked import decode_nf24_i16_weight


def bind(path: Path):
    lib = ctypes.CDLL(path.resolve().as_posix())
    fn = lib.triposplat_gemm_rnf8_avx512
    fn.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 7
    fn.restype = ctypes.c_int
    return fn


def bind_f32(path: Path):
    lib = ctypes.CDLL(path.resolve().as_posix())
    fn = lib.triposplat_gemm_f32_avx512
    fn.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 6
    fn.restype = ctypes.c_int
    return fn


def median_call(callback, warmup: int, repeat: int):
    for _ in range(warmup):
        callback()
    values = []
    for _ in range(repeat):
        started = time.perf_counter()
        callback()
        values.append(time.perf_counter() - started)
    return {"samples_sec": values, "median_sec": statistics.median(values)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--f32-library", type=Path)
    parser.add_argument("--shape", action="append", default=[])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    kernel = bind(args.library)
    f32_kernel = bind_f32(args.f32_library) if args.f32_library else None
    results = []
    for shape_text in args.shape or ["12294,1024,4096", "12294,4096,1024"]:
        m, k, n = (int(value) for value in shape_text.split(","))
        torch.manual_seed(79 + n)
        x = torch.randn(m, k, dtype=torch.float32)
        code01_t = torch.randint(-540, 541, (k, n), dtype=torch.int16)
        q2_t = torch.randint(-128, 128, (k, n), dtype=torch.int8)
        scale = torch.rand(n, dtype=torch.float32) * 1.0e-5
        bias = torch.randn(n, dtype=torch.float32) * 0.01
        codebook = torch.zeros(256, dtype=torch.float32)
        out_native = torch.empty(m, n, dtype=torch.float32)
        weight = decode_nf24_i16_weight(code01_t, q2_t, scale)
        weight_t = weight.t().contiguous()
        out_f32 = torch.empty(m, n, dtype=torch.float32)

        def native():
            status = kernel(
                x.data_ptr(), code01_t.data_ptr(), q2_t.data_ptr(), q2_t.data_ptr(),
                scale.data_ptr(), scale.data_ptr(), scale.data_ptr(), codebook.data_ptr(),
                bias.data_ptr(), out_native.data_ptr(), m, k, n, k, n, args.threads, 3,
            )
            if status:
                raise RuntimeError(f"NF24 kernel status={status}")

        def blas():
            return torch.nn.functional.linear(x, weight, bias)

        def native_f32():
            status = f32_kernel(
                x.data_ptr(), weight_t.data_ptr(), bias.data_ptr(), out_f32.data_ptr(),
                m, k, n, k, n, args.threads,
            )
            if status:
                raise RuntimeError(f"FP32 kernel status={status}")

        native_stats = median_call(native, args.warmup, args.repeat)
        blas_stats = median_call(blas, args.warmup, args.repeat)
        f32_stats = (
            median_call(native_f32, args.warmup, args.repeat)
            if f32_kernel is not None else None
        )
        out_blas = blas()
        diff = out_native - out_blas
        f32_diff = out_native - out_f32 if f32_stats is not None else None
        results.append({
            "shape": [m, k, n],
            "native_nf24": native_stats,
            "materialized_blas": blas_stats,
            "blas_over_native": blas_stats["median_sec"] / native_stats["median_sec"],
            "materialized_native_f32": f32_stats,
            "native_f32_over_nf24": (
                f32_stats["median_sec"] / native_stats["median_sec"]
                if f32_stats is not None else None
            ),
            "output_rmse": float(torch.sqrt(torch.mean(diff.square())).item()),
            "output_max_abs": float(diff.abs().max().item()),
            "native_f32_output_rmse": (
                float(torch.sqrt(torch.mean(f32_diff.square())).item())
                if f32_diff is not None else None
            ),
            "native_f32_output_max_abs": (
                float(f32_diff.abs().max().item()) if f32_diff is not None else None
            ),
            "packed_weight_bytes": int(code01_t.numel() * 3 + scale.numel() * 4),
            "materialized_weight_bytes": int(weight.numel() * 4),
        })

    output = {
        "kind": "nf24_materialized_fp32_blas_pair_benchmark",
        "threads": args.threads,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "torch_version": torch.__version__,
        "results": results,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
