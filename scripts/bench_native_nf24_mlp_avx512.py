#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_gemm(path: Path):
    lib = ctypes.CDLL(path.as_posix())
    fn = lib.triposplat_gemm_rnf8_avx512
    fn.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 7
    fn.restype = ctypes.c_int
    return fn


def load_gelu(path: Path):
    lib = ctypes.CDLL(path.as_posix())
    fn = lib.triposplat_gelu_tanh_f32_avx512
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int]
    fn.restype = ctypes.c_int
    return fn


def load_fused(path: Path):
    lib = ctypes.CDLL(path.as_posix())
    fn = lib.triposplat_mlp_nf24_gelu_f32_avx512
    fn.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 7
    fn.restype = ctypes.c_int
    row_tile = lib.triposplat_mlp_nf24_row_tile
    row_tile.argtypes = []
    row_tile.restype = ctypes.c_int
    workspace = lib.triposplat_mlp_nf24_hidden_workspace_bytes_per_thread
    workspace.argtypes = []
    workspace.restype = ctypes.c_int
    return fn, int(row_tile()), int(workspace())


def summarize(samples: list[float]) -> dict[str, object]:
    return {
        "samples_sec": samples,
        "min_sec": min(samples),
        "median_sec": statistics.median(samples),
        "max_sec": max(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemm-library", type=Path, required=True)
    parser.add_argument("--gelu-library", type=Path, required=True)
    parser.add_argument("--fused-library", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=12294)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    m, k, h, n = int(args.rows), 1024, 4096, 1024

    x = torch.randn((m, k), generator=generator, dtype=torch.float32).contiguous()
    w1_code01 = torch.randint(-508, 509, (k, h), generator=generator, dtype=torch.int16).contiguous()
    w1_code2 = torch.randint(-127, 128, (k, h), generator=generator, dtype=torch.int8).contiguous()
    w1_scale = (torch.rand((h,), generator=generator, dtype=torch.float32) * 2.0e-5 + 1.0e-6).contiguous()
    w1_bias = (torch.randn((h,), generator=generator, dtype=torch.float32) * 0.01).contiguous()
    w2_code01 = torch.randint(-508, 509, (h, n), generator=generator, dtype=torch.int16).contiguous()
    w2_code2 = torch.randint(-127, 128, (h, n), generator=generator, dtype=torch.int8).contiguous()
    w2_scale = (torch.rand((n,), generator=generator, dtype=torch.float32) * 2.0e-5 + 1.0e-6).contiguous()
    w2_bias = (torch.randn((n,), generator=generator, dtype=torch.float32) * 0.01).contiguous()
    codebook = torch.zeros((256,), dtype=torch.float32)
    hidden = torch.empty((m, h), dtype=torch.float32)
    baseline_out = torch.empty((m, n), dtype=torch.float32)
    fused_out = torch.empty_like(baseline_out)

    gemm = load_gemm(args.gemm_library)
    gelu = load_gelu(args.gelu_library)
    fused, row_tile, workspace_bytes = load_fused(args.fused_library)

    def call_gemm(
        source, code01, code2, scale, bias, output, rows: int, in_features: int, out_features: int
    ) -> None:
        status = int(
            gemm(
                source.data_ptr(), code01.data_ptr(), code2.data_ptr(), code2.data_ptr(),
                scale.data_ptr(), scale.data_ptr(), scale.data_ptr(), codebook.data_ptr(),
                bias.data_ptr(), output.data_ptr(), rows, in_features, out_features,
                in_features, out_features, args.threads, 3,
            )
        )
        if status != 0:
            raise RuntimeError(f"NF24 GEMM returned {status}")

    def baseline() -> None:
        call_gemm(x, w1_code01, w1_code2, w1_scale, w1_bias, hidden, m, k, h)
        status = int(gelu(hidden.data_ptr(), hidden.data_ptr(), hidden.numel(), args.threads))
        if status != 0:
            raise RuntimeError(f"GELU returned {status}")
        call_gemm(hidden, w2_code01, w2_code2, w2_scale, w2_bias, baseline_out, m, h, n)

    def candidate() -> None:
        status = int(
            fused(
                x.data_ptr(), w1_code01.data_ptr(), w1_code2.data_ptr(), w1_scale.data_ptr(),
                w1_bias.data_ptr(), w2_code01.data_ptr(), w2_code2.data_ptr(), w2_scale.data_ptr(),
                w2_bias.data_ptr(), fused_out.data_ptr(), m, k, h, n, k, n, args.threads,
            )
        )
        if status != 0:
            raise RuntimeError(f"fused NF24 MLP returned {status}")

    for _ in range(args.warmup):
        baseline()
        candidate()

    baseline_samples: list[float] = []
    candidate_samples: list[float] = []
    for index in range(args.repeat):
        order = (("baseline", baseline), ("candidate", candidate))
        if index % 2:
            order = tuple(reversed(order))
        for name, fn in order:
            started = time.perf_counter()
            fn()
            elapsed = time.perf_counter() - started
            (baseline_samples if name == "baseline" else candidate_samples).append(elapsed)

    diff = fused_out - baseline_out
    baseline_stats = summarize(baseline_samples)
    candidate_stats = summarize(candidate_samples)
    result = {
        "kind": "native_nf24_mlp_fusion_benchmark",
        "shape": [m, k, h, n],
        "threads": args.threads,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "libraries": {
            "gemm": {"path": args.gemm_library.as_posix(), "sha256": sha256(args.gemm_library)},
            "gelu": {"path": args.gelu_library.as_posix(), "sha256": sha256(args.gelu_library)},
            "fused": {"path": args.fused_library.as_posix(), "sha256": sha256(args.fused_library)},
        },
        "row_tile": row_tile,
        "hidden_workspace_bytes_per_thread": workspace_bytes,
        "baseline": baseline_stats,
        "candidate": candidate_stats,
        "candidate_over_baseline": candidate_stats["median_sec"] / baseline_stats["median_sec"],
        "output_rmse": float(torch.sqrt(torch.mean(diff.square())).item()),
        "output_max_abs": float(diff.abs().max().item()),
        "finite": bool(torch.isfinite(fused_out).all().item()),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
