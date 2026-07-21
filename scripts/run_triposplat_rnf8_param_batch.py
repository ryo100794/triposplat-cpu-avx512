#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import native_linear_avx512_range_patch
from native_linear_nf24_prepacked import load_nf24_i16_prepacked_model
from native_linear_rnf8_avx512_patch import apply_triposplat_native_rnf8_avx512_patch


def _install_prepacked_loader() -> None:
    packed_dir = os.environ.get("TRIPOSPLAT_RNF8_PREPACKED_DIR")
    if not packed_dir:
        return
    upstream = Path(os.environ["TRIPOSPLAT_REPO"]).resolve()
    sys.path.insert(0, upstream.as_posix())
    import torch
    import triposplat

    verify = os.environ.get("TRIPOSPLAT_RNF8_PREPACKED_VERIFY", "0") == "1"

    def load_prepacked_flow_model(_path, device=None, dtype=None):
        requested_device = torch.device("cpu" if device is None else device)
        requested_dtype = torch.float32 if dtype is None else dtype
        if requested_device.type != "cpu" or requested_dtype != torch.float32:
            raise ValueError("NF24 prepacked loader supports CPU float32 execution only")
        model = load_nf24_i16_prepacked_model(
            packed_dir,
            lambda: triposplat.LatentSeqMMFlowModel(**triposplat.FLOW_MODEL_ARGS),
            verify_checksums=verify,
        )
        return model.eval()

    triposplat.load_flow_model = load_prepacked_flow_model


def redirect_native_linear(flow_model, **kwargs):
    return apply_triposplat_native_rnf8_avx512_patch(
        flow_model,
        stages=int(os.environ.get("TRIPOSPLAT_RNF8_STAGES", "2")),
        residual_mode=os.environ.get("TRIPOSPLAT_RNF8_RESIDUAL_MODE", "nf8"),
        **kwargs,
    )


_install_prepacked_loader()
native_linear_avx512_range_patch.apply_triposplat_native_linear_avx512_patch = redirect_native_linear
runpy.run_path(
    (Path(__file__).resolve().parent / "run_triposplat_quantized_param_batch.py").as_posix(),
    run_name="__main__",
)
