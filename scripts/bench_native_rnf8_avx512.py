#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
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


def invoke(fn, tensors, shape, threads: int):
    x, code0, code1, code2, scale0, scale1, scale2, codebook, bias, out = tensors
    m, k, n = shape
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


def benchmark_pair(
    baseline,
    candidate,
    baseline_tensors,
    candidate_tensors,
    shape,
    threads,
    warmup,
    repeat,
):
    for _ in range(warmup):
        invoke(baseline, baseline_tensors, shape, threads)
        invoke(candidate, candidate_tensors, shape, threads)
    samples = {"baseline": [], "candidate": []}
    orders = (
        (("baseline", baseline, baseline_tensors), ("candidate", candidate, candidate_tensors)),
        (("candidate", candidate, candidate_tensors), ("baseline", baseline, baseline_tensors)),
    )
    for index in range(repeat):
        for name, fn, tensors in orders[index % 2]:
            started = time.perf_counter()
            invoke(fn, tensors, shape, threads)
            samples[name].append(time.perf_counter() - started)
    return {
        name: {
            "samples_sec": values,
            "median_sec": statistics.median(values),
            "min_sec": min(values),
            "max_sec": max(values),
        }
        for name, values in samples.items()
    }


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--baseline-code0-dtype", choices=("uint8", "int16"), default="uint8"
    )
    parser.add_argument(
        "--candidate-code0-dtype", choices=("uint8", "int16"), default="uint8"
    )
    parser.add_argument("--shape", action="append", default=[])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    baseline = bind(args.baseline)
    candidate = bind(args.candidate)
    results = []
    for shape_text in args.shape or ["20488,1024,4096"]:
        shape = tuple(int(value) for value in shape_text.split(","))
        if len(shape) != 3:
            raise ValueError(f"invalid shape: {shape_text}")
        m, k, n = shape
        torch.manual_seed(29 + n)
        x = torch.randn(m, k, dtype=torch.float32)
        code0_uint8 = torch.randint(0, 256, (k, n), dtype=torch.uint8)
        code0_int16 = torch.randint(-540, 541, (k, n), dtype=torch.int16)
        common = (
            torch.randint(-128, 128, (k, n), dtype=torch.int8),
            torch.randint(-128, 128, (k, n), dtype=torch.int8),
            torch.rand(n, dtype=torch.float32) * 0.01,
            torch.rand(n, dtype=torch.float32) * 0.0001,
            torch.rand(n, dtype=torch.float32) * 0.000001,
            make_nf8_codebook(torch),
            torch.randn(n, dtype=torch.float32) * 0.01,
        )
        code0 = {"uint8": code0_uint8, "int16": code0_int16}
        baseline_tensors = (
            x,
            code0[args.baseline_code0_dtype],
            *common,
            torch.empty(m, n, dtype=torch.float32),
        )
        candidate_tensors = (
            x,
            code0[args.candidate_code0_dtype],
            *common,
            torch.empty(m, n, dtype=torch.float32),
        )
        stats = benchmark_pair(
            baseline,
            candidate,
            baseline_tensors,
            candidate_tensors,
            shape,
            args.threads,
            args.warmup,
            args.repeat,
        )
        diff = baseline_tensors[-1] - candidate_tensors[-1]
        results.append(
            {
                "shape": shape,
                "baseline": stats["baseline"],
                "candidate": stats["candidate"],
                "candidate_over_baseline": (
                    stats["candidate"]["median_sec"]
                    / stats["baseline"]["median_sec"]
                ),
                "same_code0_dtype": (
                    args.baseline_code0_dtype == args.candidate_code0_dtype
                ),
                "output_rmse": torch.sqrt(torch.mean(diff.square())).item(),
                "output_max_abs": diff.abs().max().item(),
            }
        )
    output = {
        "kind": "native_rnf8_avx512_pair_benchmark",
        "created_at_unix": time.time(),
        "platform": platform.platform(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "torch_version": torch.__version__,
        "threads": args.threads,
        "baseline": {
            "path": args.baseline.as_posix(),
            "sha256": file_sha256(args.baseline),
            "code0_dtype": args.baseline_code0_dtype,
        },
        "candidate": {
            "path": args.candidate.as_posix(),
            "sha256": file_sha256(args.candidate),
            "code0_dtype": args.candidate_code0_dtype,
        },
        "results": results,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
