#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
from pathlib import Path

import native_linear_avx512_range_patch
from native_linear_rnf8_avx512_patch import apply_triposplat_native_rnf8_avx512_patch


def redirect_native_linear(flow_model, **kwargs):
    return apply_triposplat_native_rnf8_avx512_patch(
        flow_model, stages=int(os.environ.get("TRIPOSPLAT_RNF8_STAGES", "2")), **kwargs
    )


native_linear_avx512_range_patch.apply_triposplat_native_linear_avx512_patch = redirect_native_linear
runpy.run_path(
    (Path(__file__).resolve().parent / "run_triposplat_quantized_param_batch.py").as_posix(),
    run_name="__main__",
)
