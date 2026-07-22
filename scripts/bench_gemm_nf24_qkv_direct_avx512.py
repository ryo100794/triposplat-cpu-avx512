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


def stats(samples: list[float]) -> dict[str, object]:
    return {
        "samples_sec": samples,
        "min_sec": min(samples),
        "median_sec": statistics.median(samples),
        "max_sec": max(samples),
    }


def tensor_error(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, object]:
    difference = candidate - reference
    return {
        "rmse": float(torch.sqrt(torch.mean(difference.square())).item()),
        "max_abs": float(difference.abs().max().item()),
        "finite": bool(torch.isfinite(candidate).all().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemm-library", type=Path, required=True)
    parser.add_argument("--postprocess-library", type=Path, required=True)
    parser.add_argument("--candidate-library", type=Path, required=True)
    parser.add_argument("--length", type=int, default=12294)
    parser.add_argument("--channels", type=int, default=1024)
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
    b, length, channels, heads, dim = 1, args.length, args.channels, args.heads, 64
    n = 3 * channels
    length_padded = (length + 15) & ~15

    x = torch.randn((b * length, channels), generator=generator, dtype=torch.float32).contiguous()
    code01 = torch.randint(-500, 501, (channels, n), generator=generator, dtype=torch.int16).contiguous()
    code2 = torch.randint(-127, 128, (channels, n), generator=generator, dtype=torch.int8).contiguous()
    scales = (torch.rand((n,), generator=generator, dtype=torch.float32) * 1.0e-5 + 1.0e-6).contiguous()
    bias = (torch.randn((n,), generator=generator, dtype=torch.float32) * 0.01).contiguous()
    codebook = torch.zeros(256, dtype=torch.float32)
    angles = torch.randn((b, length, heads, dim // 2), generator=generator, dtype=torch.float32)
    frequencies = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1).reshape(b, length, heads, dim).contiguous()
    q_gamma = (torch.randn((heads, dim), generator=generator) * 0.05 + 1.0).contiguous()
    k_gamma = (torch.randn((heads, dim), generator=generator) * 0.05 + 1.0).contiguous()

    gemm_lib = ctypes.CDLL(args.gemm_library.as_posix())
    gemm = gemm_lib.triposplat_gemm_rnf8_avx512
    gemm.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 7
    gemm.restype = ctypes.c_int
    post_lib = ctypes.CDLL(args.postprocess_library.as_posix())
    postprocess = post_lib.triposplat_qkv_rope_rmsnorm_pack_f32_avx512_v2
    postprocess.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 7
    postprocess.restype = ctypes.c_int
    candidate_lib = ctypes.CDLL(args.candidate_library.as_posix())
    candidate = candidate_lib.triposplat_gemm_nf24_qkv_rope_rmsnorm_pack_f32_avx512
    candidate.argtypes = [ctypes.c_void_p] * 15 + [ctypes.c_int] * 10
    candidate.restype = ctypes.c_int

    qkv = torch.empty((b * length, n), dtype=torch.float32)
    reference = (
        torch.empty((b, heads, length, dim), dtype=torch.float32),
        torch.empty((b, heads, dim, length_padded), dtype=torch.float32),
        torch.empty((b, heads, dim, length_padded), dtype=torch.float32),
    )
    direct = tuple(torch.empty_like(value) for value in reference)

    def baseline() -> None:
        status = int(gemm(
            x.data_ptr(), code01.data_ptr(), code2.data_ptr(), code2.data_ptr(),
            scales.data_ptr(), scales.data_ptr(), scales.data_ptr(), codebook.data_ptr(),
            bias.data_ptr(), qkv.data_ptr(), b * length, channels, n, channels, n, args.threads, 3,
        ))
        if status != 0:
            raise RuntimeError(f"baseline GEMM returned {status}")
        status = int(postprocess(
            qkv.data_ptr(), frequencies.data_ptr(), q_gamma.data_ptr(), k_gamma.data_ptr(),
            reference[0].data_ptr(), reference[1].data_ptr(), reference[2].data_ptr(),
            b, length, heads, dim, length_padded, b, args.threads,
        ))
        if status != 0:
            raise RuntimeError(f"baseline postprocess returned {status}")

    def fused() -> None:
        status = int(candidate(
            x.data_ptr(), code01.data_ptr(), code2.data_ptr(), code2.data_ptr(),
            scales.data_ptr(), scales.data_ptr(), scales.data_ptr(), codebook.data_ptr(), bias.data_ptr(),
            frequencies.data_ptr(), q_gamma.data_ptr(), k_gamma.data_ptr(),
            direct[0].data_ptr(), direct[1].data_ptr(), direct[2].data_ptr(),
            b, length, channels, n, heads, dim, length_padded, b, args.threads, 3,
        ))
        if status != 0:
            raise RuntimeError(f"fused QKV GEMM returned {status}")

    baseline()
    fused()
    correctness = {
        name: tensor_error(direct[index], reference[index])
        for index, name in enumerate(("q", "packed_k", "packed_v"))
    }
    for _ in range(args.warmup):
        baseline()
        fused()
    samples = {"baseline": [], "fused": []}
    for repeat in range(args.repeat):
        order = (("baseline", baseline), ("fused", fused))
        if repeat & 1:
            order = tuple(reversed(order))
        for name, function in order:
            started = time.perf_counter()
            function()
            samples[name].append(time.perf_counter() - started)
    summaries = {name: stats(values) for name, values in samples.items()}
    result = {
        "kind": "nf24_qkv_gemm_direct_rope_rmsnorm_pack",
        "shape": {"batch": b, "length": length, "channels": channels, "outputs": n, "heads": heads, "head_dim": dim},
        "threads": args.threads,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "libraries": {
            "gemm": {"path": args.gemm_library.as_posix(), "sha256": sha256(args.gemm_library)},
            "postprocess": {"path": args.postprocess_library.as_posix(), "sha256": sha256(args.postprocess_library)},
            "candidate": {"path": args.candidate_library.as_posix(), "sha256": sha256(args.candidate_library)},
        },
        "timing": summaries,
        "fused_over_baseline": summaries["fused"]["median_sec"] / summaries["baseline"]["median_sec"],
        "qkv_intermediate_bytes_eliminated": qkv.numel() * qkv.element_size(),
        "correctness": correctness,
        "semantics": "Same NF24 FMA order per output, immediately followed by exact RoPE/RMSNorm and direct packed K/V stores.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
