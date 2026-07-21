#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import torch

from native_linear_nf8_avx512_patch import make_nf8_codebook
from native_linear_rnf8_avx512_patch import quantize_rnf8_per_output_channel


def bind(lib, name: str, pointer_count: int, integer_count: int):
    fn = getattr(lib, name)
    fn.argtypes = [ctypes.c_void_p] * pointer_count + [ctypes.c_int] * integer_count
    fn.restype = ctypes.c_int
    return fn


def run_case(lib, m: int, k: int, n: int, stages: int, threads: int, residual_mode: str):
    torch.manual_seed(19 + stages + n)
    x = torch.randn(m, k, dtype=torch.float32)
    weight = torch.randn(n, k, dtype=torch.float32) * 0.03
    bias = torch.randn(n, dtype=torch.float32) * 0.01
    codebook = make_nf8_codebook(torch)
    codes, scales, error = quantize_rnf8_per_output_channel(weight, codebook, stages, residual_mode)
    if residual_mode == "nf24_i16":
        combined = (
            codes[0].t().to(torch.int32) * 256
            + codes[1].t().to(torch.int32)
        )
        reconstructed = combined.to(torch.float32) * scales[0][:, None]
    else:
        reconstructed = codebook[codes[0].t().long()] * scales[0][:, None]
        for stage in range(1, stages):
            if residual_mode == "nf8":
                reconstructed = reconstructed + codebook[codes[stage].t().long()] * scales[stage][:, None]
            elif residual_mode == "symmetric_int8":
                reconstructed = reconstructed + codes[stage].t().to(torch.float32) * scales[stage][:, None]
            else:
                reconstructed = reconstructed + codes[stage].t().to(torch.float32)
    reference = torch.nn.functional.linear(x, reconstructed, bias)
    code2 = codes[2] if stages == 3 else codes[1]
    scale2 = scales[2] if stages == 3 else scales[1]
    out = torch.empty(m, n, dtype=torch.float32)
    suffix = "tail" if n % 16 else ""
    symbol = "triposplat_gemm_rnf8_avx512" + ("_tail" if suffix else "")
    kernel = bind(lib, symbol, 10, 7)
    status = kernel(
        x.data_ptr(), codes[0].data_ptr(), codes[1].data_ptr(), code2.data_ptr(),
        scales[0].data_ptr(), scales[1].data_ptr(), scale2.data_ptr(),
        codebook.data_ptr(), bias.data_ptr(), out.data_ptr(),
        m, k, n, k, n, threads, stages,
    )
    if status != 0:
        raise RuntimeError(f"{symbol} status={status}")
    torch.testing.assert_close(out, reference, rtol=2e-5, atol=2e-5)
    return {"shape": [m, k, n], "stages": stages, "residual_mode": residual_mode, "kernel_max_abs": float((out - reference).abs().max()), "packing": error}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, default=Path("artifacts/backends/libtriposplat_gemm_rnf8_avx512.so"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--residual-mode", choices=("nf8", "symmetric_int8", "nf24_i16"), default="nf8")
    args = parser.parse_args()
    lib = ctypes.CDLL(args.library.as_posix())
    stages = (3,) if args.residual_mode == "nf24_i16" else (2, 3)
    cases = []
    for stage_count in stages:
        cases.extend(
            (
                run_case(lib, 25, 96, 64, stage_count, args.threads, args.residual_mode),
                run_case(lib, 40, 96, 64, stage_count, args.threads, args.residual_mode),
                run_case(lib, 10, 96, 64, stage_count, args.threads, args.residual_mode),
                run_case(lib, 3, 96, 5, stage_count, args.threads, args.residual_mode),
            )
        )
    print({"status": "pass", "cases": cases})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
