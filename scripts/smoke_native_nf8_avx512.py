#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import torch

from native_linear_nf8_avx512_patch import make_nf8_codebook, quantize_nf8_per_output_channel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, default=Path("artifacts/backends/libtriposplat_gemm_nf8_avx512.so"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(7)
    m, k, n = 17, 96, 64
    x = torch.randn(m, k, dtype=torch.float32)
    weight = torch.randn(n, k, dtype=torch.float32) * 0.03
    bias = torch.randn(n, dtype=torch.float32) * 0.01
    codebook = make_nf8_codebook(torch)
    codes_t, scales, packing_error = quantize_nf8_per_output_channel(weight, codebook)
    dequant = (codebook[codes_t.t().long()] * scales[:, None]).contiguous()
    reference = torch.nn.functional.linear(x, dequant, bias)

    lib = ctypes.CDLL(args.library.as_posix())
    full = lib.triposplat_gemm_nf8_avx512
    full.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 6
    full.restype = ctypes.c_int
    actual = torch.empty(m, n, dtype=torch.float32)
    status = full(
        x.data_ptr(), codes_t.data_ptr(), scales.data_ptr(), codebook.data_ptr(), bias.data_ptr(), actual.data_ptr(),
        m, k, n, k, n, args.threads,
    )
    if status != 0:
        raise RuntimeError(f"full kernel status={status}")
    torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-5)

    range_kernel = lib.triposplat_gemm_nf8_avx512_range
    range_kernel.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 8
    range_kernel.restype = ctypes.c_int
    start, count = 16, 32
    actual_range = torch.empty(m, count, dtype=torch.float32)
    status = range_kernel(
        x.data_ptr(), codes_t.data_ptr(), scales.data_ptr(), codebook.data_ptr(), bias.data_ptr(), actual_range.data_ptr(),
        m, k, n, start, count, k, count, args.threads,
    )
    if status != 0:
        raise RuntimeError(f"range kernel status={status}")
    torch.testing.assert_close(actual_range, reference[:, start : start + count], rtol=2e-5, atol=2e-5)
    print({"status": "pass", "packing_error": packing_error, "max_kernel_abs": float((actual - reference).abs().max())})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
