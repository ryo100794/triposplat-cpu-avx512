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


def summarize(samples: list[float]) -> dict[str, object]:
    return {
        "samples_sec": samples,
        "min_sec": min(samples),
        "median_sec": statistics.median(samples),
        "max_sec": max(samples),
    }


def load_standard(path: Path):
    lib = ctypes.CDLL(path.as_posix())
    fn = lib.triposplat_sdpa_f32_avx512_exact_q8t512
    fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 8
    fn.restype = ctypes.c_int
    return fn


def load_packed(path: Path):
    lib = ctypes.CDLL(path.as_posix())
    fn = lib.triposplat_sdpa_f32_avx512_exact_q8t512_packed_blhd
    fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 9
    fn.restype = ctypes.c_int
    return fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard-library", type=Path, required=True)
    parser.add_argument("--packed-library", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--length", type=int, default=6147)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    b, heads, length, dim = args.batch, args.heads, args.length, 64
    length_padded = (length + 15) & ~15
    q = torch.randn((b, heads, length, dim), generator=generator, dtype=torch.float32).contiguous()
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    packed_k = torch.zeros((b, heads, dim, length_padded), dtype=torch.float32)
    packed_v = torch.zeros_like(packed_k)
    packed_k[..., :length].copy_(k.permute(0, 1, 3, 2))
    packed_v[..., :length].copy_(v.permute(0, 1, 3, 2))
    standard_out = torch.empty_like(q)
    packed_out = torch.empty((b, length, heads, dim), dtype=torch.float32)

    standard_kernel = load_standard(args.standard_library)
    packed_kernel = load_packed(args.packed_library)

    def standard() -> torch.Tensor:
        status = int(
            standard_kernel(
                q.data_ptr(), k.data_ptr(), v.data_ptr(), 0, standard_out.data_ptr(),
                b, heads, length, length, dim, 0, 0, args.threads,
            )
        )
        if status != 0:
            raise RuntimeError(f"standard SDPA returned {status}")
        return standard_out.permute(0, 2, 1, 3).contiguous()

    def candidate() -> None:
        status = int(
            packed_kernel(
                q.data_ptr(), packed_k.data_ptr(), packed_v.data_ptr(), 0, packed_out.data_ptr(),
                b, heads, length, length, dim, length_padded, 0, 0, args.threads,
            )
        )
        if status != 0:
            raise RuntimeError(f"packed SDPA returned {status}")

    expected = standard()
    candidate()
    difference = packed_out - expected
    correctness = {
        "rmse": float(torch.sqrt(torch.mean(difference.square())).item()),
        "max_abs": float(difference.abs().max().item()),
        "finite": bool(torch.isfinite(packed_out).all().item()),
    }
    del expected, difference

    for _ in range(args.warmup):
        standard()
        candidate()

    samples = {"standard": [], "packed": []}
    for repeat in range(args.repeat):
        ordered = (("standard", standard), ("packed", candidate))
        if repeat & 1:
            ordered = tuple(reversed(ordered))
        for name, function in ordered:
            started = time.perf_counter()
            function()
            samples[name].append(time.perf_counter() - started)

    standard_stats = summarize(samples["standard"])
    packed_stats = summarize(samples["packed"])
    result = {
        "kind": "native_sdpa_prepacked_kv_blhd_output_benchmark",
        "shape": {"batch": b, "heads": heads, "length": length, "head_dim": dim, "length_padded": length_padded},
        "threads": args.threads,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "libraries": {
            "standard": {"path": args.standard_library.as_posix(), "sha256": sha256(args.standard_library)},
            "packed": {"path": args.packed_library.as_posix(), "sha256": sha256(args.packed_library)},
        },
        "standard_with_blhd_materialization": standard_stats,
        "packed_with_direct_blhd_output": packed_stats,
        "packed_over_standard": packed_stats["median_sec"] / standard_stats["median_sec"],
        "correctness": correctness,
        "semantics": "Identical q8/key-tile-512 online softmax order; only K/V input and output layouts differ.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
