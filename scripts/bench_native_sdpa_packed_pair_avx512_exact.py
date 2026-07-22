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


def load(path: Path):
    lib = ctypes.CDLL(path.as_posix())
    fn = lib.triposplat_sdpa_f32_avx512_exact_q8t512_packed_blhd
    fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 9
    fn.restype = ctypes.c_int
    return fn


def summary(samples: list[float]) -> dict[str, object]:
    return {
        "samples_sec": samples,
        "median_sec": statistics.median(samples),
        "min_sec": min(samples),
        "max_sec": max(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--length", type=int, action="append", default=[])
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    baseline = load(args.baseline)
    candidate = load(args.candidate)
    results = []
    for length in args.length or [8194, 12294]:
        generator = torch.Generator(device="cpu").manual_seed(args.seed + length)
        b, h, d = args.batch, args.heads, 64
        padded = (length + 15) & ~15
        q = torch.randn((b, h, length, d), generator=generator, dtype=torch.float32)
        k = torch.randn((b, h, d, padded), generator=generator, dtype=torch.float32)
        v = torch.randn((b, h, d, padded), generator=generator, dtype=torch.float32)
        outputs = {
            "baseline": torch.empty((b, length, h, d), dtype=torch.float32),
            "candidate": torch.empty((b, length, h, d), dtype=torch.float32),
        }

        def invoke(name: str) -> None:
            fn = baseline if name == "baseline" else candidate
            status = int(fn(
                q.data_ptr(), k.data_ptr(), v.data_ptr(), 0, outputs[name].data_ptr(),
                b, h, length, length, d, padded, 0, 0, args.threads,
            ))
            if status != 0:
                raise RuntimeError(f"{name} returned status={status}")

        invoke("baseline")
        invoke("candidate")
        difference = outputs["candidate"] - outputs["baseline"]
        correctness = {
            "rmse": float(torch.sqrt(torch.mean(difference.square())).item()),
            "max_abs": float(difference.abs().max().item()),
            "finite": bool(torch.isfinite(outputs["candidate"]).all().item()),
        }
        for _ in range(args.warmup):
            invoke("baseline")
            invoke("candidate")
        samples = {"baseline": [], "candidate": []}
        for repeat in range(args.repeat):
            order = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
            for name in order:
                started = time.perf_counter()
                invoke(name)
                samples[name].append(time.perf_counter() - started)
        timing = {name: summary(values) for name, values in samples.items()}
        results.append({
            "length": length,
            "timing": timing,
            "candidate_over_baseline": timing["candidate"]["median_sec"] / timing["baseline"]["median_sec"],
            "correctness": correctness,
        })

    result = {
        "kind": "native_sdpa_packed_exact_pair_benchmark",
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "threads": args.threads,
        "baseline": {"path": args.baseline.as_posix(), "sha256": sha256(args.baseline)},
        "candidate": {"path": args.candidate.as_posix(), "sha256": sha256(args.candidate)},
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
