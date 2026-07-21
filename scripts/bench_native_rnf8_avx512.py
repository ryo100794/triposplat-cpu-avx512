#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import time
from pathlib import Path

import torch

from native_linear_nf8_avx512_patch import make_nf8_codebook


def bind(path: Path):
    lib = ctypes.CDLL(path.as_posix())
    fn = lib.triposplat_gemm_rnf8_avx512
    fn.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 7
    fn.restype = ctypes.c_int
    return fn


def benchmark(fn, tensors, shape, threads: int, warmup: int, repeat: int):
    x, code0, code1, code2, scale0, scale1, scale2, codebook, bias, out = tensors
    m, k, n = shape

    def invoke():
        status = fn(
            x.data_ptr(),
            code0.data_ptr(),
            code1.data_ptr(),
            code2.data_ptr(),
            scale0.data_ptr(),
            scale1.data_ptr(),
            scale2.data_ptr(),
            codebook.data_ptr(),
            bias.data_ptr(),
            out.data_ptr(),
            m,
            k,
            n,
            k,
            n,
            threads,
            3,
        )
        if status != 0:
            raise RuntimeError(f"kernel status={status}")

    for _ in range(warmup):
        invoke()
    samples = []
    for _ in range(repeat):
        started = time.perf_counter()
        invoke()
        samples.append(time.perf_counter() - started)
    return {
        "samples_sec": samples,
        "median_sec": statistics.median(samples),
        "min_sec": min(samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-code0-dtype", choices=("uint8", "int16"), default="uint8")
    parser.add_argument("--shape", action="append", default=["20488,1024,4096"])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    baseline = bind(args.baseline)
    candidate = bind(args.candidate)
    results = []
    for shape_text in args.shape:
        shape = tuple(int(value) for value in shape_text.split(","))
        if len(shape) != 3:
            raise ValueError(f"invalid shape: {shape_text}")
        m, k, n = shape
        torch.manual_seed(29 + n)
        tensors = (
            torch.randn(m, k, dtype=torch.float32),
            torch.randint(0, 256, (k, n), dtype=torch.uint8),
            torch.randint(-128, 128, (k, n), dtype=torch.int8),
            torch.randint(-128, 128, (k, n), dtype=torch.int8),
            torch.rand(n, dtype=torch.float32) * 0.01,
            torch.rand(n, dtype=torch.float32) * 0.0001,
            torch.rand(n, dtype=torch.float32) * 0.000001,
            make_nf8_codebook(torch),
            torch.randn(n, dtype=torch.float32) * 0.01,
            torch.empty(m, n, dtype=torch.float32),
        )
        candidate_tensors = tensors
        if args.candidate_code0_dtype == "int16":
            candidate_tensors = (
                tensors[0],
                torch.randint(-540, 541, (k, n), dtype=torch.int16),
                tensors[2],
                tensors[2],
                *tensors[4:],
            )
        baseline_stats = benchmark(
            baseline, tensors, shape, args.threads, args.warmup, args.repeat
        )
        candidate_stats = benchmark(
            candidate, candidate_tensors, shape, args.threads, args.warmup, args.repeat
        )
        results.append(
            {
                "shape": shape,
                "baseline": baseline_stats,
                "candidate": candidate_stats,
                "candidate_over_baseline": (
                    candidate_stats["median_sec"] / baseline_stats["median_sec"]
                ),
            }
        )
    print(json.dumps({"threads": args.threads, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
