#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
from statistics import mean
import time


def load_fn(lib_path: Path, symbol: str):
    lib = ctypes.CDLL(lib_path.as_posix())
    fn = getattr(lib, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    fn.restype = ctypes.c_int
    return fn


def error_stats(torch, ref, out) -> dict:
    diff = (out.float() - ref.float()).reshape(-1)
    mse = torch.mean(diff * diff).item()
    denom = torch.mean(ref.float().reshape(-1) ** 2).item()
    return {
        "rmse": math.sqrt(float(mse)),
        "rel_rmse": math.sqrt(float(mse) / max(float(denom), 1.0e-30)),
        "max_abs": float(diff.abs().max().item()),
    }


def timed(fn, repeats: int) -> tuple[list[float], object]:
    last = None
    times = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        last = fn()
        times.append(time.perf_counter() - t0)
    return times, last


def run_case(torch, F, fn, case: dict, repeats: int, threads: int) -> dict:
    B, H, Lq, Lk, D = case["shape"]
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(case["seed"]))
    q = torch.randn((B, H, Lq, D), dtype=torch.float32, generator=gen).contiguous()
    k = torch.randn((B, H, Lk, D), dtype=torch.float32, generator=gen).contiguous()
    v = torch.randn((B, H, Lk, D), dtype=torch.float32, generator=gen).contiguous()
    mask = None
    if case.get("mask"):
        mask = torch.zeros((1, 1, 1, Lk), dtype=torch.float32).contiguous()
        mask[..., 0] = math.log(float(case.get("mask_multiplicity", 7)))

    def baseline():
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    def custom():
        out = torch.empty_like(q)
        status = int(fn(
            int(q.data_ptr()),
            int(k.data_ptr()),
            int(v.data_ptr()),
            0 if mask is None else int(mask.data_ptr()),
            int(out.data_ptr()),
            B,
            H,
            Lq,
            Lk,
            D,
            0 if mask is None else 1,
            0 if mask is None else Lk,
            int(threads),
        ))
        if status != 0:
            raise RuntimeError(f"custom SDPA returned {status}")
        return out

    base_times, ref = timed(baseline, repeats)
    custom_times, out = timed(custom, repeats)
    return {
        "name": case["name"],
        "shape": case["shape"],
        "mask": bool(case.get("mask")),
        "baseline_mean_sec": float(mean(base_times)),
        "custom_mean_sec": float(mean(custom_times)),
        "speedup_vs_baseline": float(mean(base_times) / mean(custom_times)),
        "error_vs_baseline": error_stats(torch, ref, out),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", type=Path, default=Path("artifacts/backends/libtriposplat_sdpa_reference.so"))
    parser.add_argument("--symbol", default="triposplat_sdpa_f32")
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/audits/custom_sdpa_reference_smoke_20260713.json"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=int(os.environ.get("TRIPOSPLAT_CUSTOM_SDPA_THREADS", "1")))
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F

    if "OMP_NUM_THREADS" in os.environ:
        torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    fn = load_fn(args.lib, args.symbol)
    cases = [
        {"name": "self_l16_d64", "shape": [1, 2, 16, 16, 64], "seed": 11},
        {"name": "masked_self_l17_d64", "shape": [1, 2, 17, 17, 64], "seed": 12, "mask": True},
        {"name": "cross_q15_k19_d64", "shape": [1, 2, 15, 19, 64], "seed": 13},
    ]
    results = [run_case(torch, F, fn, case, int(args.repeats), int(args.threads)) for case in cases]
    payload = {
        "library": args.lib.as_posix(),
        "symbol": args.symbol,
        "threads": int(args.threads),
        "torch_version": torch.__version__,
        "results": results,
        "pass": all(row["error_vs_baseline"]["max_abs"] < 1e-4 for row in results),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
