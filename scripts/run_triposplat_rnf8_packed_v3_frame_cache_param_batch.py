#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import sys
import time
from pathlib import Path

import native_linear_avx512_range_patch
import triposplat_attention_patch
import triposplat_quantized_sampler
from native_linear_nf24_prepacked import load_nf24_i16_prepacked_model
from native_linear_rnf8_avx512_patch import apply_triposplat_native_rnf8_avx512_patch
from native_qkv_packed_attention_patch_v3 import apply_triposplat_native_qkv_packed_attention_helpers_patch
from triposplat_timestep_cache_patch import apply_triposplat_timestep_modulation_cache


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
    return apply_triposplat_native_rnf8_avx512_patch(
        flow_model,
        stages=int(os.environ.get("TRIPOSPLAT_RNF8_STAGES", "2")),
        residual_mode=os.environ.get("TRIPOSPLAT_RNF8_RESIDUAL_MODE", "nf8"),
        **kwargs,
    )


def install_packed_helpers_and_frame_caches() -> None:
    original_module_patch = triposplat_attention_patch.apply_triposplat_module_attention_patch
    original_position_cache = triposplat_attention_patch.apply_triposplat_position_embed_cache_patch

    def apply_with_helpers(flow_model, **kwargs):
        base = original_module_patch(flow_model, **kwargs)
        packed = apply_triposplat_native_qkv_packed_attention_helpers_patch(
            flow_model,
            postprocess_library=os.environ["TRIPOSPLAT_QKV_POSTPROCESS_LIBRARY"],
            sdpa_library=os.environ["TRIPOSPLAT_PACKED_SDPA_LIBRARY"],
            threads=int(os.environ.get("TRIPOSPLAT_NATIVE_SDPA_THREADS", "4")),
            include_regex=os.environ.get("TRIPOSPLAT_QKV_PACKED_INCLUDE_REGEX", r"^blocks[.][0-9]+[.]attn$"),
            strict=os.environ.get("TRIPOSPLAT_QKV_PACKED_STRICT", "1") == "1",
        )
        return {"base": base, "packed_qkv_helpers": packed}

    def apply_position_and_timestep_cache(flow_model, *, enabled=False):
        position = original_position_cache(flow_model, enabled=enabled)
        timestep = apply_triposplat_timestep_modulation_cache(
            flow_model,
            enabled=os.environ.get("TRIPOSPLAT_TIMESTEP_CACHE", "1") == "1",
            max_entries=int(os.environ.get("TRIPOSPLAT_TIMESTEP_CACHE_ENTRIES", "64")),
        )
        return {"position": position, "timestep_modulation": timestep}

    triposplat_attention_patch.apply_triposplat_module_attention_patch = apply_with_helpers
    triposplat_attention_patch.apply_triposplat_position_embed_cache_patch = apply_position_and_timestep_cache


def install_repeated_frame_probe() -> None:
    repeat = int(os.environ.get("TRIPOSPLAT_FRAME_CACHE_REPEAT", "1"))
    if repeat <= 1:
        return
    original = triposplat_quantized_sampler.FlowEulerCfgMultiVariantSampler.sample

    def sample_repeated(self, model, noise, cond, neg_cond, variants, show_progress=False, trace_callback=None):
        import torch

        runs = []
        first_outputs = None
        final_outputs = None
        final_metadata = None
        for frame_index in range(repeat):
            started = time.perf_counter()
            outputs, metadata = original(
                self,
                model,
                noise,
                cond,
                neg_cond,
                variants,
                show_progress=show_progress,
                trace_callback=trace_callback,
            )
            elapsed = time.perf_counter() - started
            equal = True
            if first_outputs is None:
                first_outputs = outputs
            else:
                for variant_name, state in first_outputs.items():
                    for state_name, value in state.items():
                        equal = equal and bool(torch.equal(value, outputs[variant_name][state_name]))
            runs.append({"frame_index": frame_index, "elapsed_sec": elapsed, "equal_to_first": equal})
            final_outputs, final_metadata = outputs, metadata
        final_metadata = dict(final_metadata)
        final_metadata["frame_reuse_probe"] = {
            "enabled": True,
            "repeat": repeat,
            "runs": runs,
            "all_outputs_equal": all(item["equal_to_first"] for item in runs),
            "semantics": "Repeats the same prepared frame to prove process-local cache hits and exact output reuse boundaries.",
        }
        return final_outputs, final_metadata

    triposplat_quantized_sampler.FlowEulerCfgMultiVariantSampler.sample = sample_repeated


install_prepacked_loader()
install_packed_helpers_and_frame_caches()
install_repeated_frame_probe()
native_linear_avx512_range_patch.apply_triposplat_native_linear_avx512_patch = redirect_native_linear
runpy.run_path(
    (Path(__file__).resolve().parent / "run_triposplat_quantized_param_batch.py").as_posix(),
    run_name="__main__",
)
