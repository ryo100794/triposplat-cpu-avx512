#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
from pathlib import Path

import native_linear_avx512_range_patch
import triposplat_attention_patch
from native_linear_avx512_range_patch import apply_triposplat_native_linear_avx512_patch as apply_f32_linear_patch
from native_linear_nf24_prepacked import load_nf24_i16_prepacked_model, materialize_nf24_i16_linears
from native_linear_rnf8_avx512_patch import apply_triposplat_native_rnf8_avx512_patch
from native_qkv_packed_attention_patch_v3 import apply_triposplat_native_qkv_packed_attention_helpers_patch


def install_prepacked_loader() -> None:
    packed_dir = os.environ.get("TRIPOSPLAT_RNF8_PREPACKED_DIR")
    if not packed_dir:
        return
    import sys

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
    include_regex = os.environ.get(
        "TRIPOSPLAT_NF24_MATERIALIZE_INCLUDE_REGEX",
        r"^(noise_refiner|context_refiner|blocks)[.][0-9]+[.]attn[.](qkv|out)$",
    )
    packed = apply_triposplat_native_rnf8_avx512_patch(
        flow_model,
        stages=int(os.environ.get("TRIPOSPLAT_RNF8_STAGES", "3")),
        residual_mode=os.environ.get("TRIPOSPLAT_RNF8_RESIDUAL_MODE", "nf24_i16"),
        **kwargs,
    )
    materialized = materialize_nf24_i16_linears(
        flow_model,
        include_regex=include_regex,
        native_weight_t_view=True,
        release_packed=os.environ.get("TRIPOSPLAT_NF24_KEEP_PACKED", "0") != "1",
        profile=False,
    )
    f32 = apply_f32_linear_patch(
        flow_model,
        enabled=True,
        include_regex=include_regex,
        library_path=os.environ.get(
            "TRIPOSPLAT_NF24_MATERIALIZED_F32_LIBRARY",
            "artifacts/backends/libtriposplat_gemm_f32_avx512.so",
        ),
        threads=int(os.environ.get("TRIPOSPLAT_NATIVE_SDPA_THREADS", "4")),
        strict=False,
    )
    if f32["selected_count"] != materialized["selected_count"]:
        raise RuntimeError(
            "materialized/native FP32 selection mismatch: "
            f"{materialized['selected_count']} != {f32['selected_count']}"
        )
    result = {
        **materialized,
        "kind": "nf24_i16_materialized_native_f32_avx512_linear",
        "runtime": f32["runtime"],
        "native_f32": {
            key: value for key, value in f32.items() if key not in ("selected", "runtime")
        },
        "remaining_packed_runtime": packed["runtime"],
    }
    result["source_packing"] = {
        key: packed[key]
        for key in (
            "kind",
            "bits_per_weight",
            "residual_quantizer",
            "selected_count",
            "packed_bytes",
            "aggregate_weight_rmse",
            "aggregate_weight_max_abs",
        )
    }
    result["semantics"] = (
        "Selected QKV/out weights materialize the same NF24 values once and use the "
        "native FP32 AVX-512/FMA kernel; remaining Linear modules decode NF24 in-kernel."
    )
    return result


def install_packed_helpers() -> None:
    original = triposplat_attention_patch.apply_triposplat_module_attention_patch

    def apply_with_helpers(flow_model, **kwargs):
        base = original(flow_model, **kwargs)
        packed = apply_triposplat_native_qkv_packed_attention_helpers_patch(
            flow_model,
            postprocess_library=os.environ["TRIPOSPLAT_QKV_POSTPROCESS_LIBRARY"],
            sdpa_library=os.environ["TRIPOSPLAT_PACKED_SDPA_LIBRARY"],
            threads=int(os.environ.get("TRIPOSPLAT_NATIVE_SDPA_THREADS", "4")),
            include_regex=os.environ.get(
                "TRIPOSPLAT_QKV_PACKED_INCLUDE_REGEX", r"^blocks[.][0-9]+[.]attn$"
            ),
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
