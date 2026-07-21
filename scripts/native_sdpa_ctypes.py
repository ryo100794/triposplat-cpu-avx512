#!/usr/bin/env python3
from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from pathlib import Path


DEFAULT_LIBRARY = "artifacts/backends/libtriposplat_sdpa_avx512_exact_q8.so"
DEFAULT_SYMBOL = "triposplat_sdpa_f32_avx512_exact_q8"


@lru_cache(maxsize=4)
def _load_kernel(path: str, symbol: str):
    lib = ctypes.CDLL(path)
    fn = getattr(lib, symbol)
    fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 8
    fn.restype = ctypes.c_int
    return fn


def native_sdpa_f32(q, k, v, *, key_bias=None, library=None, symbol=None, threads=None):
    import torch

    if q.device.type != "cpu" or k.device.type != "cpu" or v.device.type != "cpu":
        raise ValueError("native AVX-512 SDPA is CPU-only")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q/k/v must be [B,H,L,D]")
    if q.shape[:2] != k.shape[:2] or k.shape != v.shape:
        raise ValueError(f"unsupported q/k/v shapes: q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")
    if int(q.shape[-1]) != 64 or int(k.shape[-1]) != 64:
        raise ValueError("native AVX-512 SDPA requires head_dim=64")

    orig_dtype = q.dtype
    q_f = q.to(dtype=torch.float32).contiguous()
    k_f = k.to(dtype=torch.float32).contiguous()
    v_f = v.to(dtype=torch.float32).contiguous()
    bias_f = None
    if key_bias is not None:
        if int(key_bias.numel()) != int(k_f.shape[-2]):
            raise ValueError(f"key bias length mismatch: bias={key_bias.numel()} keys={k_f.shape[-2]}")
        bias_f = key_bias.to(dtype=torch.float32).reshape(-1).contiguous()

    out = torch.empty(
        (int(q_f.shape[0]), int(q_f.shape[1]), int(q_f.shape[2]), 64),
        dtype=torch.float32,
        device="cpu",
    )
    lib_path = Path(library or os.environ.get("TRIPOSPLAT_NATIVE_SDPA_LIBRARY", DEFAULT_LIBRARY)).resolve()
    symbol_name = symbol or os.environ.get("TRIPOSPLAT_NATIVE_SDPA_SYMBOL", DEFAULT_SYMBOL)
    thread_count = int(threads or os.environ.get("TRIPOSPLAT_NATIVE_SDPA_THREADS", os.environ.get("OMP_NUM_THREADS", "2")))
    fn = _load_kernel(lib_path.as_posix(), symbol_name)
    status = int(
        fn(
            q_f.data_ptr(),
            k_f.data_ptr(),
            v_f.data_ptr(),
            0 if bias_f is None else bias_f.data_ptr(),
            out.data_ptr(),
            int(q_f.shape[0]),
            int(q_f.shape[1]),
            int(q_f.shape[2]),
            int(k_f.shape[2]),
            64,
            int(bias_f is not None),
            0 if bias_f is None else int(bias_f.numel()),
            thread_count,
        )
    )
    if status != 0:
        raise RuntimeError(
            f"native AVX-512 SDPA returned status {status}; "
            f"q={tuple(q_f.shape)} k={tuple(k_f.shape)} bias={bias_f is not None}"
        )
    return out.to(dtype=orig_dtype)
