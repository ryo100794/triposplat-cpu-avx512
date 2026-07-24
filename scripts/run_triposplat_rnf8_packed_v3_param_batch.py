#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import native_linear_avx512_range_patch
import triposplat_attention_patch
from native_linear_nf24_prepacked import load_nf24_i16_prepacked_model
from native_linear_rnf8_avx512_patch import apply_triposplat_native_rnf8_avx512_patch
from native_qkv_packed_attention_patch_v3 import apply_triposplat_native_qkv_packed_attention_helpers_patch


def install_prepacked_loader() -> None:
    packed_dir = os.environ.get("TRIPOSPLAT_RNF8_PREPACKED_DIR")
    if not packed_dir:
        return
    upstream = Path(os.environ["TRIPOSPLAT_REPO"]).resolve()
    sys.path.insert(0, upstream.as_posix())
    import torch
    import triposplat

    def load_prepacked_flow_model(_path, device=None, dtype=None):
        if torch.device("cpu" if device is None else device).type != "cpu":
            raise ValueError("NF24 prepacked loader supports CPU only")
        if (torch.float32 if dtype is None else dtype) != torch.float32:
            raise ValueError("NF24 prepacked loader supports float32 only")
        return load_nf24_i16_prepacked_model(
            packed_dir,
            lambda: triposplat.LatentSeqMMFlowModel(**triposplat.FLOW_MODEL_ARGS),
            verify_checksums=os.environ.get("TRIPOSPLAT_RNF8_PREPACKED_VERIFY", "0") == "1",
        ).eval()

    triposplat.load_flow_model = load_prepacked_flow_model


def redirect_native_linear(flow_model, **kwargs):
    shape_text = os.environ.get("TRIPOSPLAT_RNF8_OPTIMIZED_SHAPES", "")
    optimized_shapes = tuple(
        tuple(int(value) for value in item.split("x"))
        for item in shape_text.split(",") if item
    )
    return apply_triposplat_native_rnf8_avx512_patch(
        flow_model,
        optimized_library_path=os.environ.get("TRIPOSPLAT_RNF8_OPTIMIZED_LIBRARY"),
        optimized_shapes=optimized_shapes,
        stages=int(os.environ.get("TRIPOSPLAT_RNF8_STAGES", "2")),
        residual_mode=os.environ.get("TRIPOSPLAT_RNF8_RESIDUAL_MODE", "nf8"),
        **kwargs,
    )


def install_packed_helpers() -> None:
    original = triposplat_attention_patch.apply_triposplat_module_attention_patch

    def apply_with_helpers(flow_model, **kwargs):
        base = original(flow_model, **kwargs)
        packed = apply_triposplat_native_qkv_packed_attention_helpers_patch(
            flow_model,
            postprocess_library=os.environ["TRIPOSPLAT_QKV_POSTPROCESS_LIBRARY"],
            sdpa_library=os.environ["TRIPOSPLAT_PACKED_SDPA_LIBRARY"],
            threads=int(os.environ.get("TRIPOSPLAT_NATIVE_SDPA_THREADS", "4")),
            include_regex=os.environ.get("TRIPOSPLAT_QKV_PACKED_INCLUDE_REGEX", r"^blocks[.][0-9]+[.]attn$"),
            strict=os.environ.get("TRIPOSPLAT_QKV_PACKED_STRICT", "1") == "1",
        )
        return {"base": base, "packed_qkv_helpers": packed}

    triposplat_attention_patch.apply_triposplat_module_attention_patch = apply_with_helpers


install_prepacked_loader()
install_packed_helpers()
native_linear_avx512_range_patch.apply_triposplat_native_linear_avx512_patch = redirect_native_linear
runpy.run_path(
    (Path(__file__).resolve().parent / "run_triposplat_quantized_param_batch.py").as_posix(),
    run_name="__main__",
)
