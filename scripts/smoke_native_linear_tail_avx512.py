#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import torch

from native_linear_nf8_avx512_patch import make_nf8_codebook, quantize_nf8_per_output_channel


def bind(library: ctypes.CDLL, symbol: str, pointer_count: int):
    kernel = getattr(library, symbol)
    kernel.argtypes = [ctypes.c_void_p] * pointer_count + [ctypes.c_int] * 6
    kernel.restype = ctypes.c_int
    return kernel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nf8-library", type=Path, default=Path("artifacts/backends/libtriposplat_gemm_nf8_avx512.so"))
    parser.add_argument("--f32-library", type=Path, default=Path("artifacts/backends/libtriposplat_gemm_f32_avx512.so"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(11)
    m, k, n = 3, 96, 5
    x = torch.randn(m, k, dtype=torch.float32)
    weight = torch.randn(n, k, dtype=torch.float32) * 0.03
    bias = torch.randn(n, dtype=torch.float32) * 0.01

    f32_weight_t = weight.t().contiguous()
    f32_out = torch.empty(m, n, dtype=torch.float32)
    f32_kernel = bind(ctypes.CDLL(args.f32_library.as_posix()), "triposplat_gemm_f32_avx512_tail", 4)
    status = f32_kernel(
        x.data_ptr(), f32_weight_t.data_ptr(), bias.data_ptr(), f32_out.data_ptr(),
        m, k, n, k, n, args.threads,
    )
    if status != 0:
        raise RuntimeError(f"float32 tail status={status}")
    f32_reference = torch.nn.functional.linear(x, weight, bias)
    torch.testing.assert_close(f32_out, f32_reference, rtol=2e-5, atol=2e-5)

    codebook = make_nf8_codebook(torch)
    codes_t, scales, _ = quantize_nf8_per_output_channel(weight, codebook)
    dequant = codebook[codes_t.t().long()] * scales[:, None]
    nf8_reference = torch.nn.functional.linear(x, dequant, bias)
    nf8_out = torch.empty(m, n, dtype=torch.float32)
    nf8_kernel = bind(ctypes.CDLL(args.nf8_library.as_posix()), "triposplat_gemm_nf8_avx512_tail", 6)
    status = nf8_kernel(
        x.data_ptr(), codes_t.data_ptr(), scales.data_ptr(), codebook.data_ptr(), bias.data_ptr(), nf8_out.data_ptr(),
        m, k, n, k, n, args.threads,
    )
    if status != 0:
        raise RuntimeError(f"NF8 tail status={status}")
    torch.testing.assert_close(nf8_out, nf8_reference, rtol=2e-5, atol=2e-5)
    print(
        {
            "status": "pass",
            "shape": [m, k, n],
            "f32_max_abs": float((f32_out - f32_reference).abs().max()),
            "nf8_kernel_max_abs": float((nf8_out - nf8_reference).abs().max()),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
