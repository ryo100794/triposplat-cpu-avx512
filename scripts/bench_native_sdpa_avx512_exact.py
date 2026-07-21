#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def load_kernel(path: Path, symbol: str):
    lib = ctypes.CDLL(path.as_posix())
    fn = getattr(lib, symbol)
    fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 8
    fn.restype = ctypes.c_int
    return fn


def call_kernel(fn, q, k, v, bias, out, threads):
    b, h, lq, d = (int(value) for value in q.shape)
    lk = int(k.shape[2])
    status = int(
        fn(
            q.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
            0 if bias is None else bias.data_ptr(),
            out.data_ptr(),
            b,
            h,
            lq,
            lk,
            d,
            int(bias is not None),
            0 if bias is None else int(bias.numel()),
            int(threads),
        )
    )
    if status != 0:
        raise RuntimeError(f"native SDPA returned {status}")


def timed(fn, warmup, repeat):
    values = []
    for index in range(warmup + repeat):
        started = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - started
        if index >= warmup:
            values.append(elapsed)
    return {
        "times_sec": values,
        "min_sec": min(values),
        "mean_sec": sum(values) / len(values),
        "max_sec": max(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, default=Path("artifacts/backends/libtriposplat_sdpa_avx512_exact_q8.so"))
    parser.add_argument("--symbol", default="triposplat_sdpa_f32_avx512_exact_q8")
    parser.add_argument("--case", action="append", default=[], help="name,Lq,Lk,bias_index,bias_value; use -1,0 for no bias")
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.torch_threads))
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    fn = load_kernel(args.library, args.symbol)
    cases = args.case or ["self256,256,256,-1,0", "masked256,256,256,7,2.0", "cross193x256,193,256,-1,0"]
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    rows = []
    for spec in cases:
        name, lq_raw, lk_raw, bias_index_raw, bias_value_raw = spec.split(",")
        lq, lk = int(lq_raw), int(lk_raw)
        q = torch.randn((1, args.heads, lq, 64), generator=generator, dtype=torch.float32).contiguous()
        k = torch.randn((1, args.heads, lk, 64), generator=generator, dtype=torch.float32).contiguous()
        v = torch.randn((1, args.heads, lk, 64), generator=generator, dtype=torch.float32).contiguous()
        bias_index = int(bias_index_raw)
        bias = None
        mask = None
        if bias_index >= 0:
            bias = torch.zeros((lk,), dtype=torch.float32)
            bias[bias_index] = float(bias_value_raw)
            mask = bias.view(1, 1, 1, lk)
        expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        observed = torch.empty_like(expected)
        call_kernel(fn, q, k, v, bias, observed, args.threads)
        diff = observed - expected
        rmse = torch.sqrt(torch.mean(diff.square())).item()
        ref_rmse = torch.sqrt(torch.mean(expected.square())).item()
        torch_stats = timed(lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=mask), args.warmup, args.repeat)
        native_stats = timed(lambda: call_kernel(fn, q, k, v, bias, observed, args.threads), args.warmup, args.repeat)
        rows.append({
            "name": name,
            "shape_q": list(q.shape),
            "shape_kv": list(k.shape),
            "has_key_bias": bias is not None,
            "torch": torch_stats,
            "native_avx512": native_stats,
            "speedup_vs_torch_mean": torch_stats["mean_sec"] / native_stats["mean_sec"],
            "rmse": rmse,
            "relative_rmse": rmse / max(ref_rmse, 1.0e-30),
            "max_abs": diff.abs().max().item(),
        })
        del q, k, v, bias, mask, expected, observed, diff
    result = {
        "kind": "native_sdpa_avx512_exact_benchmark",
        "library": args.library.as_posix(),
        "symbol": args.symbol,
        "threads": args.threads,
        "torch_threads": args.torch_threads,
        "cases": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
