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


def load_norm_rope(path: Path):
    lib = ctypes.CDLL(path.as_posix())
    rope = lib.triposplat_rope_complex_f32_avx512
    rope.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64, ctypes.c_int, ctypes.c_int]
    rope.restype = ctypes.c_int
    rms = lib.triposplat_multihead_rmsnorm_f32_avx512
    rms.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int]
    rms.restype = ctypes.c_int
    return rope, rms


def load_candidate(path: Path):
    lib = ctypes.CDLL(path.as_posix())
    fn = lib.triposplat_qkv_rope_rmsnorm_pack_f32_avx512
    fn.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 7
    fn.restype = ctypes.c_int
    tile = lib.triposplat_qkv_postprocess_token_tile
    tile.argtypes = []
    tile.restype = ctypes.c_int
    return fn, int(tile())


def tensor_error(candidate: torch.Tensor, baseline: torch.Tensor) -> dict[str, object]:
    difference = candidate - baseline
    return {
        "rmse": float(torch.sqrt(torch.mean(difference.square())).item()),
        "max_abs": float(difference.abs().max().item()),
        "finite": bool(torch.isfinite(candidate).all().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm-rope-library", type=Path, required=True)
    parser.add_argument("--candidate-library", type=Path, required=True)
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
    b, length, heads, dim = args.batch, args.length, args.heads, 64
    length_padded = (length + 15) & ~15

    qkv = torch.randn((b, length, 3, heads, dim), generator=generator, dtype=torch.float32).contiguous()
    angles = torch.randn((b, length, heads, dim // 2), generator=generator, dtype=torch.float32)
    frequencies = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1).reshape(b, length, heads, dim).contiguous()
    q_gamma = (torch.randn((heads, dim), generator=generator) * 0.05 + 1.0).contiguous()
    k_gamma = (torch.randn((heads, dim), generator=generator) * 0.05 + 1.0).contiguous()
    candidate_q = torch.empty((b, heads, length, dim), dtype=torch.float32)
    candidate_k = torch.empty((b, heads, dim, length_padded), dtype=torch.float32)
    candidate_v = torch.empty_like(candidate_k)

    rope, rms = load_norm_rope(args.norm_rope_library)
    candidate_kernel, token_tile = load_candidate(args.candidate_library)

    def checked_rope(source: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(source)
        status = int(rope(source.data_ptr(), frequencies.data_ptr(), output.data_ptr(), source.numel() // dim, dim, args.threads))
        if status != 0:
            raise RuntimeError(f"RoPE returned {status}")
        return output

    def checked_rms(source: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(source)
        status = int(rms(source.data_ptr(), gamma.data_ptr(), output.data_ptr(), b * length, heads, dim, 1.0e-12, args.threads))
        if status != 0:
            raise RuntimeError(f"RMSNorm returned {status}")
        return output

    def baseline():
        q = qkv[:, :, 0].contiguous()
        k = qkv[:, :, 1].contiguous()
        v = qkv[:, :, 2]
        q = checked_rms(checked_rope(q), q_gamma)
        k = checked_rms(checked_rope(k), k_gamma)
        q_bhld = q.permute(0, 2, 1, 3).contiguous()
        packed_k = torch.zeros((b, heads, dim, length_padded), dtype=torch.float32)
        packed_v = torch.zeros_like(packed_k)
        packed_k[..., :length].copy_(k.permute(0, 2, 3, 1))
        packed_v[..., :length].copy_(v.permute(0, 2, 3, 1))
        return q_bhld, packed_k, packed_v

    def candidate() -> None:
        status = int(
            candidate_kernel(
                qkv.data_ptr(), frequencies.data_ptr(), q_gamma.data_ptr(), k_gamma.data_ptr(),
                candidate_q.data_ptr(), candidate_k.data_ptr(), candidate_v.data_ptr(),
                b, length, heads, dim, length_padded, b, args.threads,
            )
        )
        if status != 0:
            raise RuntimeError(f"QKV postprocess returned {status}")

    baseline_q, baseline_k, baseline_v = baseline()
    candidate()
    correctness = {
        "q_bhld": tensor_error(candidate_q, baseline_q),
        "packed_k": tensor_error(candidate_k, baseline_k),
        "packed_v": tensor_error(candidate_v, baseline_v),
    }
    del baseline_q, baseline_k, baseline_v

    for _ in range(args.warmup):
        baseline()
        candidate()

    samples = {"baseline": [], "candidate": []}
    for repeat in range(args.repeat):
        ordered = (("baseline", baseline), ("candidate", candidate))
        if repeat & 1:
            ordered = tuple(reversed(ordered))
        for name, function in ordered:
            started = time.perf_counter()
            function()
            samples[name].append(time.perf_counter() - started)

    baseline_stats = summarize(samples["baseline"])
    candidate_stats = summarize(samples["candidate"])
    packed_bytes = b * heads * dim * length_padded * 4
    result = {
        "kind": "native_qkv_rope_rmsnorm_direct_pack_benchmark",
        "shape": {"batch": b, "length": length, "heads": heads, "head_dim": dim, "length_padded": length_padded},
        "threads": args.threads,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "token_tile": token_tile,
        "libraries": {
            "norm_rope": {"path": args.norm_rope_library.as_posix(), "sha256": sha256(args.norm_rope_library)},
            "candidate": {"path": args.candidate_library.as_posix(), "sha256": sha256(args.candidate_library)},
        },
        "baseline": baseline_stats,
        "candidate": candidate_stats,
        "candidate_over_baseline": candidate_stats["median_sec"] / baseline_stats["median_sec"],
        "packed_k_bytes": packed_bytes,
        "packed_v_bytes": packed_bytes,
        "q_bytes": b * heads * length * dim * 4,
        "correctness": correctness,
        "semantics": "RoPE then RMSNorm, followed by Q BHLD and K/V BHDL-padded layout materialization; no approximation.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
