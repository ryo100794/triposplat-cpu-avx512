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


def error(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, object]:
    difference = candidate - reference
    return {
        "rmse": float(torch.sqrt(torch.mean(difference.square())).item()),
        "max_abs": float(difference.abs().max().item()),
        "finite": bool(torch.isfinite(candidate).all().item()),
    }


def load_v1(path: Path):
    lib = ctypes.CDLL(path.as_posix())
    fn = lib.triposplat_qkv_rope_rmsnorm_pack_f32_avx512
    fn.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 7
    fn.restype = ctypes.c_int
    return fn


def load_v2(path: Path):
    lib = ctypes.CDLL(path.as_posix())
    regular = lib.triposplat_qkv_rope_rmsnorm_pack_f32_avx512_v2
    regular.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 7
    regular.restype = ctypes.c_int
    selected = lib.triposplat_q_kv_selected_rope_rmsnorm_pack_f32_avx512_v2
    selected.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int] * 8
    selected.restype = ctypes.c_int
    return regular, selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-library", type=Path, required=True)
    parser.add_argument("--v2-library", type=Path, required=True)
    parser.add_argument("--length", type=int, default=12294)
    parser.add_argument("--selected-length", type=int, default=8193)
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
    b, length, selected_length, heads, dim = 1, args.length, args.selected_length, args.heads, 64
    if selected_length > length:
        raise ValueError("selected length cannot exceed full length")
    length_padded = (length + 15) & ~15

    qkv = torch.randn((b, length, 3, heads, dim), generator=generator, dtype=torch.float32).contiguous()
    angles = torch.randn((b, length, heads, dim // 2), generator=generator, dtype=torch.float32)
    frequencies = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1).reshape(b, length, heads, dim).contiguous()
    q_gamma = (torch.randn((heads, dim), generator=generator) * 0.05 + 1.0).contiguous()
    k_gamma = (torch.randn((heads, dim), generator=generator) * 0.05 + 1.0).contiguous()

    selected_indices = torch.arange(selected_length, dtype=torch.int64)
    if selected_length < length:
        selected_indices[-1] = length - 1
    q_selected = qkv[:, selected_indices, 0].contiguous()
    kv = qkv[:, :, 1:].contiguous()

    v1 = load_v1(args.v1_library)
    v2_regular, v2_selected = load_v2(args.v2_library)
    regular_outputs = {
        name: (
            torch.empty((b, heads, length, dim), dtype=torch.float32),
            torch.full((b, heads, dim, length_padded), float("nan"), dtype=torch.float32),
            torch.full((b, heads, dim, length_padded), float("nan"), dtype=torch.float32),
        )
        for name in ("v1", "v2")
    }

    def regular(name: str) -> None:
        q, k, v = regular_outputs[name]
        function = v1 if name == "v1" else v2_regular
        status = int(function(
            qkv.data_ptr(), frequencies.data_ptr(), q_gamma.data_ptr(), k_gamma.data_ptr(),
            q.data_ptr(), k.data_ptr(), v.data_ptr(),
            b, length, heads, dim, length_padded, b, args.threads,
        ))
        if status != 0:
            raise RuntimeError(f"{name} regular returned {status}")

    regular("v1")
    regular("v2")
    regular_correctness = {
        name: error(regular_outputs["v2"][index], regular_outputs["v1"][index])
        for index, name in enumerate(("q", "packed_k", "packed_v"))
    }

    selected_outputs = (
        torch.empty((b, heads, selected_length, dim), dtype=torch.float32),
        torch.full((b, heads, dim, length_padded), float("nan"), dtype=torch.float32),
        torch.full((b, heads, dim, length_padded), float("nan"), dtype=torch.float32),
    )

    def selected() -> None:
        q, k, v = selected_outputs
        status = int(v2_selected(
            q_selected.data_ptr(), kv.data_ptr(), frequencies.data_ptr(), selected_indices.data_ptr(),
            q_gamma.data_ptr(), k_gamma.data_ptr(), q.data_ptr(), k.data_ptr(), v.data_ptr(),
            b, selected_length, length, heads, dim, length_padded, b, args.threads,
        ))
        if status != 0:
            raise RuntimeError(f"selected v2 returned {status}")

    selected()
    selected_correctness = {
        "q": error(selected_outputs[0], regular_outputs["v1"][0][:, :, selected_indices]),
        "packed_k": error(selected_outputs[1], regular_outputs["v1"][1]),
        "packed_v": error(selected_outputs[2], regular_outputs["v1"][2]),
    }

    for _ in range(args.warmup):
        regular("v1")
        regular("v2")
        selected()
    samples = {"v1_regular": [], "v2_regular": [], "v2_selected": []}
    functions = {"v1_regular": lambda: regular("v1"), "v2_regular": lambda: regular("v2"), "v2_selected": selected}
    for repeat in range(args.repeat):
        order = list(functions)
        if repeat & 1:
            order.reverse()
        for name in order:
            started = time.perf_counter()
            functions[name]()
            samples[name].append(time.perf_counter() - started)
    summaries = {name: stats(values) for name, values in samples.items()}

    tail = length_padded - length
    result = {
        "kind": "native_qkv_postprocess_v2_tail_clear_and_selected_rows",
        "shape": {
            "batch": b,
            "length": length,
            "selected_length": selected_length,
            "heads": heads,
            "head_dim": dim,
            "length_padded": length_padded,
            "tail_tokens": tail,
        },
        "threads": args.threads,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "libraries": {
            "v1": {"path": args.v1_library.as_posix(), "sha256": sha256(args.v1_library)},
            "v2": {"path": args.v2_library.as_posix(), "sha256": sha256(args.v2_library)},
        },
        "timing": summaries,
        "v2_regular_over_v1": summaries["v2_regular"]["median_sec"] / summaries["v1_regular"]["median_sec"],
        "v2_selected_over_v1_full": summaries["v2_selected"]["median_sec"] / summaries["v1_regular"]["median_sec"],
        "correctness": {"regular": regular_correctness, "selected": selected_correctness},
        "tail_is_zero": {
            "regular_k": bool(tail == 0 or torch.count_nonzero(regular_outputs["v2"][1][..., length:]).item() == 0),
            "regular_v": bool(tail == 0 or torch.count_nonzero(regular_outputs["v2"][2][..., length:]).item() == 0),
            "selected_k": bool(tail == 0 or torch.count_nonzero(selected_outputs[1][..., length:]).item() == 0),
            "selected_v": bool(tail == 0 or torch.count_nonzero(selected_outputs[2][..., length:]).item() == 0),
        },
        "semantics": "Exact RoPE/RMSNorm and packed layouts; v2 clears only the final zero-padding tail and selected mode computes Q only for selected rows.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
