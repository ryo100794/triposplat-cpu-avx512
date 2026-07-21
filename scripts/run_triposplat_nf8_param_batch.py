#!/usr/bin/env python3
"""Run the research sampler with its native Linear hook redirected to NF8."""

from __future__ import annotations

import runpy
from pathlib import Path

import native_linear_avx512_range_patch
from native_linear_nf8_avx512_patch import apply_triposplat_native_nf8_avx512_patch


def redirect_native_linear(flow_model, **kwargs):
    return apply_triposplat_native_nf8_avx512_patch(flow_model, **kwargs)


native_linear_avx512_range_patch.apply_triposplat_native_linear_avx512_patch = redirect_native_linear
runpy.run_path(
    (Path(__file__).resolve().parent / "run_triposplat_quantized_param_batch.py").as_posix(),
    run_name="__main__",
)
