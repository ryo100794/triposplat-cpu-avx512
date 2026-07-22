#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def load_kernel(path: Path, symbol: str):
    lib = ctypes.CDLL(path.as_posix())
    fn = getattr(lib, symbol)
    fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 8
    fn.restype = ctypes.c_int
    profile_reset = getattr(lib, "triposplat_sdpa_q8t512_profile_reset", None)
    profile_get = getattr(lib, "triposplat_sdpa_q8t512_profile_get", None)
    if profile_reset is not None:
        profile_reset.argtypes = []
        profile_reset.restype = None
    if profile_get is not None:
        profile_get.argtypes = [ctypes.POINTER(ctypes.c_uint64)] * 5
        profile_get.restype = ctypes.c_int
    workspace_get = getattr(lib, "triposplat_sdpa_q8t512_workspace_stats", None)
    if workspace_get is not None:
        workspace_get.argtypes = [ctypes.POINTER(ctypes.c_uint64)] * 2
        workspace_get.restype = ctypes.c_int
    key_tile = getattr(lib, "triposplat_sdpa_key_tile", None)
    if key_tile is not None:
        key_tile.argtypes = []
        key_tile.restype = ctypes.c_int
    query_block = getattr(lib, "triposplat_sdpa_query_block", None)
    if query_block is not None:
        query_block.argtypes = []
        query_block.restype = ctypes.c_int
    return fn, profile_reset, profile_get, workspace_get, key_tile, query_block


def read_profile(profile_get, workspace_get, key_tile, query_block):
    if profile_get is None:
        return None
    values = [ctypes.c_uint64() for _ in range(5)]
    status = profile_get(*(ctypes.byref(value) for value in values))
    if status != 0:
        raise RuntimeError(f"native SDPA profile returned {status}")
    calls, allocate_ns, pack_ns, compute_ns, free_ns = (value.value for value in values)
    total_ns = allocate_ns + pack_ns + compute_ns + free_ns
    workspace = None
    if workspace_get is not None:
        allocations = ctypes.c_uint64()
        capacity_bytes = ctypes.c_uint64()
        workspace_status = workspace_get(
            ctypes.byref(allocations), ctypes.byref(capacity_bytes)
        )
        if workspace_status != 0:
            raise RuntimeError(f"native SDPA workspace profile returned {workspace_status}")
        workspace = {
            "allocations_since_reset": allocations.value,
            "capacity_bytes": capacity_bytes.value,
        }
    return {
        "calls": calls,
        "key_tile": None if key_tile is None else key_tile(),
        "query_block": None if query_block is None else query_block(),
        "workspace": workspace,
        "allocate_sec": allocate_ns / 1.0e9,
        "pack_sec": pack_ns / 1.0e9,
        "compute_sec": compute_ns / 1.0e9,
        "free_sec": free_ns / 1.0e9,
        "total_sec": total_ns / 1.0e9,
        "per_call_sec": {
            "allocate": allocate_ns / max(calls, 1) / 1.0e9,
            "pack": pack_ns / max(calls, 1) / 1.0e9,
            "compute": compute_ns / max(calls, 1) / 1.0e9,
            "free": free_ns / max(calls, 1) / 1.0e9,
        },
    }


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
        "median_sec": statistics.median(values),
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
    parser.add_argument("--skip-torch-timing", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.torch_threads))
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    fn, profile_reset, profile_get, workspace_get, key_tile, query_block = load_kernel(args.library, args.symbol)
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
        torch_stats = None
        if not args.skip_torch_timing:
            torch_stats = timed(
                lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=mask),
                args.warmup,
                args.repeat,
            )
        for _ in range(args.warmup):
            call_kernel(fn, q, k, v, bias, observed, args.threads)
        if profile_reset is not None:
            profile_reset()
        native_stats = timed(
            lambda: call_kernel(fn, q, k, v, bias, observed, args.threads),
            0,
            args.repeat,
        )
        native_profile = read_profile(profile_get, workspace_get, key_tile, query_block)
        rows.append({
            "name": name,
            "shape_q": list(q.shape),
            "shape_kv": list(k.shape),
            "has_key_bias": bias is not None,
            "torch": torch_stats,
            "native_avx512": native_stats,
            "native_profile": native_profile,
            "speedup_vs_torch_mean": (
                None if torch_stats is None
                else torch_stats["mean_sec"] / native_stats["mean_sec"]
            ),
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
