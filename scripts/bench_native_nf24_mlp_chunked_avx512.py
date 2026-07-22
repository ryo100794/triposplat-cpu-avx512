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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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


def load_chunked(path: Path):
    lib = ctypes.CDLL(path.as_posix())
    fn = lib.triposplat_mlp_nf24_gelu_chunked_f32_avx512
    fn.argtypes = [ctypes.c_void_p] * 11 + [ctypes.c_int] * 9
    fn.restype = ctypes.c_int
    return fn


def summarize(samples: list[float]) -> dict[str, object]:
    return {
        "samples_sec": samples,
        "min_sec": min(samples),
        "median_sec": statistics.median(samples),
        "max_sec": max(samples),
    }


def parse_chunks(raw: str, rows: int) -> list[int]:
    chunks = []
    for value in raw.split(","):
        chunk = min(int(value), rows)
        if chunk <= 0:
            raise ValueError("chunk rows must be positive")
        if chunk not in chunks:
            chunks.append(chunk)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemm-library", type=Path, required=True)
    parser.add_argument("--gelu-library", type=Path, required=True)
    parser.add_argument("--chunked-library", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=12294)
    parser.add_argument("--chunks", default="64,256,512,1024,2048,4096,12294")
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
    chunks = parse_chunks(args.chunks, m)
    max_chunk = max(chunks)

    x = torch.randn((m, k), generator=generator, dtype=torch.float32).contiguous()
    w1_code01 = torch.randint(-508, 509, (k, h), generator=generator, dtype=torch.int16).contiguous()
    w1_code2 = torch.randint(-127, 128, (k, h), generator=generator, dtype=torch.int8).contiguous()
    w1_scale = (torch.rand((h,), generator=generator) * 2.0e-5 + 1.0e-6).contiguous()
    w1_bias = (torch.randn((h,), generator=generator) * 0.01).contiguous()
    w2_code01 = torch.randint(-508, 509, (h, n), generator=generator, dtype=torch.int16).contiguous()
    w2_code2 = torch.randint(-127, 128, (h, n), generator=generator, dtype=torch.int8).contiguous()
    w2_scale = (torch.rand((n,), generator=generator) * 2.0e-5 + 1.0e-6).contiguous()
    w2_bias = (torch.randn((n,), generator=generator) * 0.01).contiguous()
    codebook = torch.zeros((256,), dtype=torch.float32)
    baseline_hidden = torch.empty((m, h), dtype=torch.float32)
    candidate_hidden = torch.empty((max_chunk, h), dtype=torch.float32)
    baseline_out = torch.empty((m, n), dtype=torch.float32)
    candidate_out = torch.empty_like(baseline_out)

    gemm = load_gemm(args.gemm_library)
    gelu = load_gelu(args.gelu_library)
    chunked = load_chunked(args.chunked_library)

    def call_gemm(source, code01, code2, scale, bias, output, rows, in_features, out_features):
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
        call_gemm(x, w1_code01, w1_code2, w1_scale, w1_bias, baseline_hidden, m, k, h)
        status = int(gelu(baseline_hidden.data_ptr(), baseline_hidden.data_ptr(), baseline_hidden.numel(), args.threads))
        if status != 0:
            raise RuntimeError(f"GELU returned {status}")
        call_gemm(baseline_hidden, w2_code01, w2_code2, w2_scale, w2_bias, baseline_out, m, h, n)

    def candidate(chunk_rows: int) -> None:
        status = int(
            chunked(
                x.data_ptr(), w1_code01.data_ptr(), w1_code2.data_ptr(), w1_scale.data_ptr(),
                w1_bias.data_ptr(), w2_code01.data_ptr(), w2_code2.data_ptr(), w2_scale.data_ptr(),
                w2_bias.data_ptr(), candidate_hidden.data_ptr(), candidate_out.data_ptr(),
                m, k, h, n, k, n, max_chunk, chunk_rows, args.threads,
            )
        )
        if status != 0:
            raise RuntimeError(f"chunked NF24 MLP returned {status} for chunk {chunk_rows}")

    baseline()
    correctness: dict[str, dict[str, object]] = {}
    for chunk in chunks:
        candidate(chunk)
        diff = candidate_out - baseline_out
        correctness[str(chunk)] = {
            "output_rmse": float(torch.sqrt(torch.mean(diff.square())).item()),
            "output_max_abs": float(diff.abs().max().item()),
            "finite": bool(torch.isfinite(candidate_out).all().item()),
        }

    functions = [("baseline", baseline)] + [
        (f"chunk_{chunk}", lambda chunk=chunk: candidate(chunk)) for chunk in chunks
    ]
    for _ in range(args.warmup):
        for _, fn in functions:
            fn()

    samples = {name: [] for name, _ in functions}
    for repeat in range(args.repeat):
        ordered = functions[repeat % len(functions):] + functions[:repeat % len(functions)]
        for name, fn in ordered:
            started = time.perf_counter()
            fn()
            samples[name].append(time.perf_counter() - started)

    baseline_stats = summarize(samples["baseline"])
    candidates = {}
    for chunk in chunks:
        stats = summarize(samples[f"chunk_{chunk}"])
        candidates[str(chunk)] = {
            **stats,
            "candidate_over_baseline": stats["median_sec"] / baseline_stats["median_sec"],
            "hidden_workspace_bytes": chunk * h * 4,
            **correctness[str(chunk)],
        }

    result = {
        "kind": "native_nf24_mlp_chunk_sweep_benchmark",
        "shape": [m, k, h, n],
        "threads": args.threads,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "libraries": {
            "gemm": {"path": args.gemm_library.as_posix(), "sha256": sha256(args.gemm_library)},
            "gelu": {"path": args.gelu_library.as_posix(), "sha256": sha256(args.gelu_library)},
            "chunked": {"path": args.chunked_library.as_posix(), "sha256": sha256(args.chunked_library)},
        },
        "baseline": baseline_stats,
        "candidates": candidates,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
