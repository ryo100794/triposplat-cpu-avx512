#!/usr/bin/env python3
"""Run batched/quantized TripoSplat flow parameter experiments.

The output of this script is sampled latent NPZ files, not final Gaussian PLYs.
Use the existing external-noise runner with --latent-npz to decode any selected
latent. This keeps expensive flow exploration separate from cheap decode/export.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get("GS_PROJECT_ROOT", SCRIPT_DIR.parent)).resolve()
BASE = Path(os.environ.get("GS_BASE", PROJECT.parent)).resolve()
REPO = Path(os.environ.get("TRIPOSPLAT_REPO", PROJECT / "vendor" / "TripoSplat")).resolve()
CKPTS = Path(os.environ.get("TRIPOSPLAT_CKPTS", PROJECT / "models" / "TripoSplat" / "ckpts")).resolve()


def _state_to_dtype(values: dict, dtype):
    return {key: value.to(dtype=dtype) for key, value in values.items()}


def _state_roundtrip(values: dict, storage_dtype, runtime_dtype):
    return {key: value.to(dtype=storage_dtype).to(dtype=runtime_dtype) for key, value in values.items()}


def _repeat_state_for_batch(values: dict, batch: int):
    if int(batch) <= 1:
        return {key: value for key, value in values.items()}
    return {key: value.repeat(int(batch), *([1] * (value.dim() - 1))) for key, value in values.items()}


def _max_variant_group_batch(variants) -> int:
    counts = {}
    for variant in variants:
        steps = int(variant.steps)
        counts[steps] = counts.get(steps, 0) + 1
    return max(counts.values()) if counts else 1


def _compile_flow_model_if_requested(
    model,
    torch_module,
    *,
    enabled: bool,
    backend: str,
    mode: str,
    fullgraph: bool,
    dynamic: bool,
    warmup: bool,
    cfg_split_forward: bool,
    cfg_single_state_forward: bool,
    variants,
    noise: dict | None,
    cond: dict | None,
    neg_cond: dict | None,
):
    meta = {
        "enabled": bool(enabled),
        "backend": backend,
        "mode": mode,
        "fullgraph": bool(fullgraph),
        "dynamic": bool(dynamic),
        "warmup": bool(warmup),
    }
    if not enabled:
        return model, meta
    if not hasattr(torch_module, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch build")

    compile_kwargs = {
        "backend": backend,
        "fullgraph": bool(fullgraph),
        "dynamic": bool(dynamic),
    }
    if mode != "none":
        compile_kwargs["mode"] = mode
    started = time.time()
    compiled = torch_module.compile(model, **compile_kwargs)
    meta["wrap_elapsed_sec"] = time.time() - started
    meta["compiled_class"] = compiled.__class__.__name__

    if warmup:
        if noise is None or cond is None or neg_cond is None:
            raise RuntimeError("torch.compile warmup requires prepared noise/condition tensors")
        batch = _max_variant_group_batch(variants)
        sample_b = _repeat_state_for_batch(noise, batch)
        cond_b = _repeat_state_for_batch(cond, batch)
        neg_b = _repeat_state_for_batch(neg_cond, batch)
        t = torch_module.full(
            (batch,),
            1000.0,
            dtype=torch_module.float32,
            device=next(iter(sample_b.values())).device,
        )
        if cfg_split_forward:
            warm_state = sample_b
            warm_t = t
            warm_cond = cond_b
            warm_effective_batch = batch
        else:
            warm_t = torch_module.cat([t, t], dim=0)
            warm_cond = {key: torch_module.cat([cond_b[key], neg_b[key]], dim=0) for key in cond_b}
            if cfg_single_state_forward:
                warm_state = sample_b
            else:
                warm_state = {key: torch_module.cat([value, value], dim=0) for key, value in sample_b.items()}
            warm_effective_batch = batch * 2
        started = time.time()
        warm_out = compiled(warm_state, warm_t, warm_cond)
        first_tensor = next(iter(warm_out.values()))
        meta["warmup_materialized_scalar"] = float(first_tensor.reshape(-1)[0].detach().float().cpu())
        meta["warmup_elapsed_sec"] = time.time() - started
        meta["warmup_effective_batch"] = int(warm_effective_batch)
        meta["warmup_state_shapes"] = {key: list(value.shape) for key, value in warm_state.items()}
        meta["warmup_condition_shapes"] = {key: list(value.shape) for key, value in warm_cond.items()}
    return compiled, meta


def _uniform_patch_grid_indices(torch_module, *, original_grid: int, target_grid: int, keep_prefix: int, device):
    if target_grid >= original_grid:
        patch_idx = torch_module.arange(original_grid * original_grid, device=device, dtype=torch_module.long)
    else:
        coords = torch_module.linspace(0, original_grid - 1, int(target_grid), device=device).round().to(dtype=torch_module.long)
        yy, xx = torch_module.meshgrid(coords, coords, indexing="ij")
        patch_idx = (yy.reshape(-1) * original_grid + xx.reshape(-1)).to(dtype=torch_module.long)
    if keep_prefix > 0:
        prefix = torch_module.arange(int(keep_prefix), device=device, dtype=torch_module.long)
        return torch_module.cat([prefix, patch_idx + int(keep_prefix)], dim=0)
    return patch_idx


def _reduce_condition_tokens(cond: dict, torch_module, *, mode: str = "none", patch_grid_size: int | None = None, keep_prefix: int = 5):
    if mode == "none":
        return cond, {"enabled": False, "mode": mode}
    if mode != "grid":
        raise ValueError(f"unsupported condition token reduction mode: {mode}")
    if patch_grid_size is None or int(patch_grid_size) <= 0:
        raise ValueError("--condition-patch-grid-size must be positive when --condition-token-mode=grid")
    if "feature1" not in cond:
        raise ValueError("condition token reduction requires feature1")
    feature1 = cond["feature1"]
    token_count = int(feature1.shape[1])
    keep_prefix = int(keep_prefix)
    patch_count = token_count - keep_prefix
    original_grid = math.isqrt(patch_count)
    if original_grid * original_grid != patch_count:
        raise ValueError(f"condition patch token count must be square after prefix: tokens={token_count} prefix={keep_prefix}")
    target_grid = min(int(patch_grid_size), original_grid)
    idx = _uniform_patch_grid_indices(
        torch_module,
        original_grid=original_grid,
        target_grid=target_grid,
        keep_prefix=keep_prefix,
        device=feature1.device,
    )
    reduced = {}
    reduced_keys = []
    skipped_keys = []
    for key, value in cond.items():
        if torch_module.is_tensor(value) and value.dim() >= 2 and int(value.shape[1]) == token_count:
            reduced[key] = value.index_select(1, idx.to(device=value.device))
            reduced_keys.append(key)
        else:
            reduced[key] = value
            skipped_keys.append(key)
    kept_tokens = int(idx.numel())
    return reduced, {
        "enabled": True,
        "mode": mode,
        "keep_prefix_tokens": keep_prefix,
        "original_patch_grid": original_grid,
        "target_patch_grid": target_grid,
        "original_condition_tokens": token_count,
        "kept_condition_tokens": kept_tokens,
        "token_reduction_ratio": kept_tokens / token_count,
        "main_sequence_length_estimate": 8192 + kept_tokens + 1,
        "index_dtype": str(idx.dtype).replace("torch.", ""),
        "reduced_keys": reduced_keys,
        "skipped_keys": skipped_keys,
        "semantics": "non-equivalent speed mode: uniformly samples condition patch tokens while preserving the first prefix/register tokens",
    }


def _round_module_floating_tensors_(module, torch_module, *, storage_dtype):
    """Round floating parameters/buffers through storage_dtype, preserving runtime dtype."""
    meta = {
        "enabled": True,
        "storage_dtype": str(storage_dtype).replace("torch.", ""),
        "parameters": {"count": 0, "elements": 0},
        "buffers": {"count": 0, "elements": 0},
    }
    with torch_module.no_grad():
        for _, param in module.named_parameters(recurse=True):
            if not param.is_floating_point():
                continue
            param.copy_(param.detach().to(dtype=storage_dtype).to(dtype=param.dtype))
            meta["parameters"]["count"] += 1
            meta["parameters"]["elements"] += int(param.numel())
        for _, buf in module.named_buffers(recurse=True):
            if not buf.is_floating_point():
                continue
            buf.copy_(buf.detach().to(dtype=storage_dtype).to(dtype=buf.dtype))
            meta["buffers"]["count"] += 1
            meta["buffers"]["elements"] += int(buf.numel())
    return meta


def _round_output_value(value, torch_module, *, storage_dtype):
    if torch_module.is_tensor(value):
        if value.is_floating_point():
            return value.to(dtype=storage_dtype).to(dtype=value.dtype)
        return value
    if isinstance(value, tuple):
        return tuple(_round_output_value(item, torch_module, storage_dtype=storage_dtype) for item in value)
    if isinstance(value, list):
        return [_round_output_value(item, torch_module, storage_dtype=storage_dtype) for item in value]
    if isinstance(value, dict):
        return {key: _round_output_value(item, torch_module, storage_dtype=storage_dtype) for key, item in value.items()}
    return value


def _cast_floating_value(value, torch_module, *, dtype):
    if torch_module.is_tensor(value):
        if value.is_floating_point():
            return value.to(dtype=dtype)
        return value
    if isinstance(value, tuple):
        return tuple(_cast_floating_value(item, torch_module, dtype=dtype) for item in value)
    if isinstance(value, list):
        return [_cast_floating_value(item, torch_module, dtype=dtype) for item in value]
    if isinstance(value, dict):
        return {key: _cast_floating_value(item, torch_module, dtype=dtype) for key, item in value.items()}
    return value


def _install_native_half_module_hooks(module, torch_module, *, include_regex=None, exclude_regex=None, output_dtype, compute_dtype="float16"):
    include_re = re.compile(include_regex) if include_regex else None
    exclude_re = re.compile(exclude_regex) if exclude_regex else None
    handles = []
    selected = []
    param_count = 0
    param_elements = 0
    output_torch_dtype = getattr(torch_module, output_dtype)
    compute_torch_dtype = getattr(torch_module, compute_dtype)

    def pre_hook(_mod, args, kwargs):
        return (
            _cast_floating_value(args, torch_module, dtype=compute_torch_dtype),
            _cast_floating_value(kwargs, torch_module, dtype=compute_torch_dtype),
        )

    def post_hook(_mod, _args, _kwargs, output):
        return _cast_floating_value(output, torch_module, dtype=output_torch_dtype)

    for name, child in module.named_modules():
        if not name:
            continue
        if include_re is not None and not include_re.search(name):
            continue
        if exclude_re is not None and exclude_re.search(name):
            continue
        child.to(dtype=compute_torch_dtype)
        setattr(child, "_native_sequence_compute_dtype", compute_torch_dtype)
        for param in child.parameters(recurse=True):
            if not param.is_floating_point():
                continue
            param_count += 1
            param_elements += int(param.numel())
        handles.append(child.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(child.register_forward_hook(post_hook, with_kwargs=True))
        selected.append(name)
    return {
        "enabled": bool(selected),
        "compute_dtype": compute_dtype,
        "output_dtype": output_dtype,
        "include_regex": include_regex,
        "exclude_regex": exclude_regex,
        "selected_count": len(selected),
        "selected_preview": selected[:32],
        "parameters": {"count": param_count, "elements": param_elements},
        "handles": handles,
    }


def _install_timestep_embedder_functional_half_patch(flow_model, torch_module, *, enabled=False, output_dtype="float32", compute_dtype="float16"):
    if not enabled:
        return {"enabled": False}
    import types
    import torch.nn.functional as F

    embedder = getattr(flow_model, "t_embedder", None)
    if embedder is None:
        return {"enabled": False, "reason": "flow_model has no t_embedder"}
    compute_torch_dtype = getattr(torch_module, compute_dtype)
    output_torch_dtype = getattr(torch_module, output_dtype)
    param_count = 0
    param_elements = 0
    for param in embedder.parameters(recurse=True):
        if not param.is_floating_point():
            continue
        param_count += 1
        param_elements += int(param.numel())

    if not hasattr(embedder, "_original_forward_functional_half"):
        embedder._original_forward_functional_half = embedder.forward

    def patched_forward(self, t):
        emb = self.timestep_embedding(t, self.frequency_embedding_size)
        h = emb.to(dtype=compute_torch_dtype)
        for layer in self.mlp:
            weight = getattr(layer, "weight", None)
            if torch_module.is_tensor(weight) and weight.is_floating_point():
                bias = getattr(layer, "bias", None)
                h = F.linear(
                    h,
                    weight.to(dtype=compute_torch_dtype),
                    None if bias is None else bias.to(dtype=compute_torch_dtype),
                )
            else:
                h = layer(h)
                if torch_module.is_tensor(h) and h.is_floating_point() and h.dtype != compute_torch_dtype:
                    h = h.to(dtype=compute_torch_dtype)
        return h.to(dtype=output_torch_dtype)

    embedder.forward = types.MethodType(patched_forward, embedder)
    return {
        "enabled": True,
        "kind": "timestep_embedder_functional_half_patch",
        "module": "t_embedder",
        "compute_dtype": compute_dtype,
        "output_dtype": output_dtype,
        "parameters": {"count": param_count, "elements": param_elements},
        "preserves_parameter_dtype": True,
        "preserves_flow_model_dtype_property": True,
        "note": "TimestepEmbedder MLP is evaluated with casted half weights/activations without changing stored parameter dtype.",
    }


def _install_module_output_rounding_hooks(module, torch_module, *, storage_dtype, include_regex=None, exclude_regex=None):
    include_re = re.compile(include_regex) if include_regex else None
    exclude_re = re.compile(exclude_regex) if exclude_regex else None
    handles = []
    selected = []

    def hook(_mod, _inputs, output):
        return _round_output_value(output, torch_module, storage_dtype=storage_dtype)

    for name, child in module.named_modules():
        if not name:
            continue
        if include_re is not None and not include_re.search(name):
            continue
        if exclude_re is not None and exclude_re.search(name):
            continue
        handles.append(child.register_forward_hook(hook))
        selected.append(name)
    return {
        "enabled": True,
        "storage_dtype": str(storage_dtype).replace("torch.", ""),
        "include_regex": include_regex,
        "exclude_regex": exclude_regex,
        "selected_count": len(selected),
        "selected_preview": selected[:32],
        "split_by_step": bool(split_by_step),
        "handles": handles,
    }


def _install_module_input_rounding_hooks(module, torch_module, *, storage_dtype, include_regex=None, exclude_regex=None):
    include_re = re.compile(include_regex) if include_regex else None
    exclude_re = re.compile(exclude_regex) if exclude_regex else None
    handles = []
    selected = []

    def hook(_mod, inputs):
        return _round_output_value(inputs, torch_module, storage_dtype=storage_dtype)

    for name, child in module.named_modules():
        if not name:
            continue
        if include_re is not None and not include_re.search(name):
            continue
        if exclude_re is not None and exclude_re.search(name):
            continue
        handles.append(child.register_forward_pre_hook(hook))
        selected.append(name)
    return {
        "enabled": True,
        "storage_dtype": str(storage_dtype).replace("torch.", ""),
        "include_regex": include_regex,
        "exclude_regex": exclude_regex,
        "selected_count": len(selected),
        "selected_preview": selected[:32],
        "handles": handles,
    }


def _manifest_hook_meta(meta: dict):
    if not meta.get("enabled"):
        return meta
    return {key: value for key, value in meta.items() if key != "handles"}


def _install_linear_input_covariance_capture(
    module,
    torch_module,
    *,
    output_npz,
    include_regex=None,
    exclude_regex=None,
    group_size: int = 16,
    split_by_step: bool = False,
    step_context: dict | None = None,
):
    if output_npz is None:
        return {"enabled": False}, None
    if int(group_size) <= 0:
        raise ValueError(f"capture group_size must be > 0, got {group_size}")
    import numpy as np
    import torch.nn as nn

    include_re = re.compile(include_regex) if include_regex else None
    exclude_re = re.compile(exclude_regex) if exclude_regex else None
    handles = []
    selected = []
    stats = {}
    group = int(group_size)

    def make_hook(name: str):
        def hook(_mod, inputs):
            if not inputs:
                return
            x = inputs[0]
            if not torch_module.is_tensor(x) or x.dim() < 2:
                return
            flat = x.detach().reshape(-1, int(x.shape[-1])).to(dtype=torch_module.float32)
            in_features = int(flat.shape[-1])
            group_count = math.ceil(in_features / group)
            entry = stats.get(name)
            if entry is None:
                entry = {
                    "count": 0,
                    "calls": 0,
                    "in_features": in_features,
                    "group_count": group_count,
                }
                if split_by_step:
                    entry["by_step"] = {}
                else:
                    entry["cov"] = torch_module.zeros((group_count, group, group), dtype=torch_module.float64, device="cpu")
                stats[name] = entry
            elif int(entry["in_features"]) != in_features:
                raise ValueError(f"input feature mismatch for {name}: {entry['in_features']} vs {in_features}")
            if split_by_step:
                step = int((step_context or {}).get("step", 0))
                by_step = entry["by_step"]
                target = by_step.get(step)
                if target is None:
                    target = {
                        "count": 0,
                        "calls": 0,
                        "cov": torch_module.zeros((group_count, group, group), dtype=torch_module.float64, device="cpu"),
                    }
                    by_step[step] = target
            else:
                target = entry
            entry["calls"] += 1
            entry["count"] += int(flat.shape[0])
            target["calls"] += 1
            target["count"] += int(flat.shape[0])
            for group_idx, start in enumerate(range(0, in_features, group)):
                end = min(start + group, in_features)
                part = flat[:, start:end]
                cov = part.transpose(0, 1).matmul(part).to(dtype=torch_module.float64).cpu()
                target["cov"][group_idx, : end - start, : end - start].add_(cov)
        return hook

    for name, child in module.named_modules():
        if not name or not isinstance(child, nn.Linear):
            continue
        if include_re is not None and not include_re.search(name):
            continue
        if exclude_re is not None and exclude_re.search(name):
            continue
        handles.append(child.register_forward_pre_hook(make_hook(name)))
        selected.append(name)

    output_npz = Path(output_npz)
    meta = {
        "enabled": True,
        "output_npz": output_npz.as_posix(),
        "include_regex": include_regex,
        "exclude_regex": exclude_regex,
        "group_size": group,
        "selected_count": len(selected),
        "selected_preview": selected[:32],
        "split_by_step": bool(split_by_step),
        "handles": handles,
    }

    def finalize():
        for handle in handles:
            handle.remove()
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        names = sorted(stats)
        if split_by_step:
            steps = sorted({int(step) for name in names for step in stats[name].get("by_step", {})})
            if names and steps:
                per_module_covariances = [
                    np.stack([
                        stats[name]["by_step"].get(step, {"cov": torch_module.zeros((int(stats[name]["group_count"]), group, group), dtype=torch_module.float64, device="cpu")})["cov"].numpy().astype("float32")
                        for step in steps
                    ], axis=0)
                    for name in names
                ]
                counts = np.array([[int(stats[name]["by_step"].get(step, {}).get("count", 0)) for step in steps] for name in names], dtype=np.int64)
                calls = np.array([[int(stats[name]["by_step"].get(step, {}).get("calls", 0)) for step in steps] for name in names], dtype=np.int64)
                in_features = np.array([int(stats[name]["in_features"]) for name in names], dtype=np.int64)
            else:
                per_module_covariances = []
                counts = np.zeros((0, 0), dtype=np.int64)
                calls = np.zeros((0, 0), dtype=np.int64)
                in_features = np.zeros((0,), dtype=np.int64)
            arrays = dict(
                module_names=np.array(names),
                steps=np.array(steps, dtype=np.int64),
                counts=counts,
                calls=calls,
                in_features=in_features,
                group_size=np.array([group], dtype=np.int64),
                split_by_step=np.array([True]),
            )
            shapes = [tuple(arr.shape) for arr in per_module_covariances]
            if not per_module_covariances or len(set(shapes)) <= 1:
                arrays["covariances"] = (
                    np.stack(per_module_covariances, axis=0)
                    if per_module_covariances
                    else np.zeros((0, 0, 0, group, group), dtype=np.float32)
                )
                arrays["covariance_format"] = np.array(["stacked"])
            else:
                keys = []
                for idx, arr in enumerate(per_module_covariances):
                    key = f"covariances_{idx:04d}"
                    arrays[key] = arr
                    keys.append(key)
                arrays["covariances"] = np.zeros((0, 0, 0, group, group), dtype=np.float32)
                arrays["covariance_format"] = np.array(["per_module_arrays"])
                arrays["covariance_keys"] = np.array(keys)
                arrays["covariance_shapes"] = np.array(shapes, dtype=np.int64)
            np.savez(output_npz, **arrays)
        else:
            if names:
                per_module_covariances = [stats[name]["cov"].numpy().astype("float32") for name in names]
                counts = np.array([int(stats[name]["count"]) for name in names], dtype=np.int64)
                calls = np.array([int(stats[name]["calls"]) for name in names], dtype=np.int64)
                in_features = np.array([int(stats[name]["in_features"]) for name in names], dtype=np.int64)
            else:
                per_module_covariances = []
                counts = np.zeros((0,), dtype=np.int64)
                calls = np.zeros((0,), dtype=np.int64)
                in_features = np.zeros((0,), dtype=np.int64)
            arrays = dict(
                module_names=np.array(names),
                counts=counts,
                calls=calls,
                in_features=in_features,
                group_size=np.array([group], dtype=np.int64),
                split_by_step=np.array([False]),
            )
            shapes = [tuple(arr.shape) for arr in per_module_covariances]
            if not per_module_covariances or len(set(shapes)) <= 1:
                arrays["covariances"] = (
                    np.stack(per_module_covariances, axis=0)
                    if per_module_covariances
                    else np.zeros((0, 0, group, group), dtype=np.float32)
                )
                arrays["covariance_format"] = np.array(["stacked"])
            else:
                keys = []
                for idx, arr in enumerate(per_module_covariances):
                    key = f"covariances_{idx:04d}"
                    arrays[key] = arr
                    keys.append(key)
                arrays["covariances"] = np.zeros((0, 0, group, group), dtype=np.float32)
                arrays["covariance_format"] = np.array(["per_module_arrays"])
                arrays["covariance_keys"] = np.array(keys)
                arrays["covariance_shapes"] = np.array(shapes, dtype=np.int64)
            np.savez(output_npz, **arrays)
        out_meta = _manifest_hook_meta(meta)
        out_meta.update(
            {
                "captured_module_count": len(names),
                "captured_modules": names[:32],
                "counts": {name: int(stats[name]["count"]) for name in names},
                "calls": {name: int(stats[name]["calls"]) for name in names},
                "split_by_step": bool(split_by_step),
                "steps": steps if split_by_step else None,
            }
        )
        return out_meta

    return meta, finalize


def _install_linear_output_residual_capture(
    module,
    torch_module,
    *,
    output_npz,
    include_regex=None,
    exclude_regex=None,
    bits: int = 8,
    mode: str = "linear_per_channel_symmetric",
    split_by_step: bool = True,
    step_context: dict | None = None,
):
    if output_npz is None:
        return {"enabled": False}, None
    if int(bits) < 2 or int(bits) > 8:
        raise ValueError(f"output residual bits must be in [2, 8], got {bits}")
    if mode != "linear_per_channel_symmetric":
        raise ValueError(f"unsupported output residual mode: {mode}")
    import numpy as np
    import torch.nn as nn

    include_re = re.compile(include_regex) if include_regex else None
    exclude_re = re.compile(exclude_regex) if exclude_regex else None
    handles = []
    selected = []
    stats = {}
    diff_cache = {}
    levels = float((1 << (int(bits) - 1)) - 1)

    def dequant_weight(weight):
        wf = weight.detach().to(dtype=torch_module.float32)
        scale = torch_module.clamp(wf.abs().amax(dim=1) / levels, min=1.0e-30)
        q = torch_module.round(wf / scale[:, None]).clamp(-levels, levels)
        return q * scale[:, None]

    def make_hook(name: str):
        def hook(mod, inputs):
            if not inputs:
                return
            x = inputs[0]
            if not torch_module.is_tensor(x) or x.dim() < 2:
                return
            flat = x.detach().reshape(-1, int(x.shape[-1])).to(dtype=torch_module.float32)
            in_features = int(flat.shape[-1])
            weight = mod.weight.detach()
            out_features = int(weight.shape[0])
            if int(weight.shape[1]) != in_features:
                raise ValueError(f"input feature mismatch for {name}: weight={int(weight.shape[1])} input={in_features}")
            diff = diff_cache.get(name)
            if diff is None:
                dequant = dequant_weight(weight)
                diff = (weight.detach().to(dtype=torch_module.float32) - dequant).contiguous()
                diff_cache[name] = diff
            residual = flat.matmul(diff.transpose(0, 1))
            step = int((step_context or {}).get("step", 0)) if split_by_step else 0
            entry = stats.get(name)
            if entry is None:
                entry = {
                    "in_features": in_features,
                    "out_features": out_features,
                    "by_step": {},
                    "weight_rmse": float(torch_module.sqrt(torch_module.mean(diff * diff)).item()),
                    "weight_max_abs": float(diff.abs().amax().item()),
                }
                stats[name] = entry
            target = entry["by_step"].get(step)
            if target is None:
                target = {
                    "count": 0,
                    "calls": 0,
                    "sumsq": torch_module.zeros((out_features,), dtype=torch_module.float64, device="cpu"),
                    "sumabs": torch_module.zeros((out_features,), dtype=torch_module.float64, device="cpu"),
                    "maxabs": torch_module.zeros((out_features,), dtype=torch_module.float32, device="cpu"),
                }
                entry["by_step"][step] = target
            abs_residual = residual.abs()
            target["count"] += int(residual.shape[0])
            target["calls"] += 1
            target["sumsq"].add_(torch_module.sum(residual * residual, dim=0).to(dtype=torch_module.float64).cpu())
            target["sumabs"].add_(torch_module.sum(abs_residual, dim=0).to(dtype=torch_module.float64).cpu())
            target["maxabs"] = torch_module.maximum(target["maxabs"], abs_residual.amax(dim=0).to(dtype=torch_module.float32).cpu())
        return hook

    for name, child in module.named_modules():
        if not name or not isinstance(child, nn.Linear):
            continue
        if include_re is not None and not include_re.search(name):
            continue
        if exclude_re is not None and exclude_re.search(name):
            continue
        handles.append(child.register_forward_pre_hook(make_hook(name)))
        selected.append(name)

    output_npz = Path(output_npz)
    meta = {
        "enabled": True,
        "output_npz": output_npz.as_posix(),
        "include_regex": include_regex,
        "exclude_regex": exclude_regex,
        "bits": int(bits),
        "mode": mode,
        "split_by_step": bool(split_by_step),
        "selected_count": len(selected),
        "selected_preview": selected[:32],
        "handles": handles,
    }

    def finalize():
        for handle in handles:
            handle.remove()
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        names = sorted(stats)
        steps = sorted({int(step) for name in names for step in stats[name].get("by_step", {})})
        if names and steps:
            out_features = np.array([int(stats[name]["out_features"]) for name in names], dtype=np.int64)
            max_out = int(max(out_features.tolist()))
            sumsq = np.zeros((len(names), len(steps), max_out), dtype=np.float64)
            sumabs = np.zeros((len(names), len(steps), max_out), dtype=np.float64)
            maxabs = np.zeros((len(names), len(steps), max_out), dtype=np.float32)
            counts = np.zeros((len(names), len(steps)), dtype=np.int64)
            calls = np.zeros((len(names), len(steps)), dtype=np.int64)
            in_features = np.array([int(stats[name]["in_features"]) for name in names], dtype=np.int64)
            weight_rmse = np.array([float(stats[name]["weight_rmse"]) for name in names], dtype=np.float32)
            weight_max_abs = np.array([float(stats[name]["weight_max_abs"]) for name in names], dtype=np.float32)
            for nidx, name in enumerate(names):
                width = int(stats[name]["out_features"])
                for sidx, step in enumerate(steps):
                    target = stats[name]["by_step"].get(step)
                    if target is None:
                        continue
                    counts[nidx, sidx] = int(target["count"])
                    calls[nidx, sidx] = int(target["calls"])
                    sumsq[nidx, sidx, :width] = target["sumsq"].numpy()
                    sumabs[nidx, sidx, :width] = target["sumabs"].numpy()
                    maxabs[nidx, sidx, :width] = target["maxabs"].numpy()
            denom = np.maximum(counts[:, :, None], 1)
            rmse = np.sqrt(sumsq / denom).astype(np.float32)
            mae = (sumabs / denom).astype(np.float32)
        else:
            in_features = np.zeros((0,), dtype=np.int64)
            out_features = np.zeros((0,), dtype=np.int64)
            weight_rmse = np.zeros((0,), dtype=np.float32)
            weight_max_abs = np.zeros((0,), dtype=np.float32)
            counts = np.zeros((0, 0), dtype=np.int64)
            calls = np.zeros((0, 0), dtype=np.int64)
            sumsq = np.zeros((0, 0, 0), dtype=np.float64)
            sumabs = np.zeros((0, 0, 0), dtype=np.float64)
            maxabs = np.zeros((0, 0, 0), dtype=np.float32)
            rmse = np.zeros((0, 0, 0), dtype=np.float32)
            mae = np.zeros((0, 0, 0), dtype=np.float32)
        np.savez(
            output_npz,
            module_names=np.array(names),
            steps=np.array(steps, dtype=np.int64),
            counts=counts,
            calls=calls,
            in_features=in_features,
            out_features=out_features,
            output_sumsq=sumsq.astype(np.float32),
            output_sumabs=sumabs.astype(np.float32),
            output_rmse=rmse,
            output_mae=mae,
            output_maxabs=maxabs,
            weight_rmse=weight_rmse,
            weight_max_abs=weight_max_abs,
            bits=np.array([int(bits)], dtype=np.int64),
            mode=np.array([mode]),
            split_by_step=np.array([bool(split_by_step)]),
        )
        out_meta = _manifest_hook_meta(meta)
        out_meta.update({
            "captured_module_count": len(names),
            "captured_modules": names[:32],
            "steps": steps,
            "counts": {name: {str(step): int(stats[name]["by_step"].get(step, {}).get("count", 0)) for step in steps} for name in names},
            "calls": {name: {str(step): int(stats[name]["by_step"].get(step, {}).get("calls", 0)) for step in steps} for name in names},
            "weight_rmse": {name: float(stats[name]["weight_rmse"]) for name in names},
        })
        return out_meta

    return meta, finalize


def _configure_torch_interop_threads(torch_module):
    raw = os.environ.get("TORCH_NUM_INTEROP_THREADS")
    if not raw:
        return None
    value = int(raw)
    torch_module.set_num_interop_threads(value)
    return value


def _parse_quantize_points(raw: str) -> tuple[str, ...]:
    if not raw:
        return ()
    valid = {"state", "condition", "prediction", "cfg_prediction", "updated_state"}
    points = tuple(part.strip() for part in raw.split(",") if part.strip())
    bad = [point for point in points if point not in valid]
    if bad:
        raise ValueError(f"unsupported quantize point(s): {bad}; valid={sorted(valid)}")
    return points


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Prepared RGB image, normally prepared_rgb.webp")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument(
        "--variants",
        help="Comma list of name:steps:guidance:shift, e.g. base:20:3:3,g2p5:20:2.5:3",
    )
    parser.add_argument("--sampler-solver", choices=["euler", "ab2"], default="euler", help="Flow ODE integrator. euler is official-compatible; ab2 is a non-equivalent NFE-reduction candidate.")
    parser.add_argument("--sampler-clone-state", action=argparse.BooleanOptionalAction, default=True, help="Clone sampler state before each model forward. Disable only for verified inference paths that do not mutate x_t.")
    parser.add_argument("--canvas-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--model-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--noise-mode", choices=["torch", "numpy"], default="numpy")
    parser.add_argument("--noise-npz", type=Path, help="Load flow initial noise from an npz file")
    parser.add_argument("--save-noise-npz", type=Path, help="Save the flow initial noise actually used")
    parser.add_argument("--condition-npz", type=Path, help="Load encoded condition feature1/feature2 from an npz file")
    parser.add_argument("--static-condition-cache", action="store_true", help="Exact optimization: precompute condition embedding/context_refiner output once and reuse it across sampler steps")
    parser.add_argument("--position-embed-cache", action="store_true", help="Exact optimization: cache the fixed pos_embedder(pos_pe) tensor across flow forwards")
    parser.add_argument("--repo-polar-cos-sin", action="store_true", help="CPU candidate: build RePo rotary complex frequencies with cos/sin instead of torch.polar")
    parser.add_argument("--cfg-deduplicate-state-forward", action="store_true", help="Exact optimization for non-split CFG: compute duplicate state/timestep-side prefix once and repeat it before main blocks")
    parser.add_argument("--cfg-deduplicate-state-assume-duplicated", action="store_true", help="Skip runtime equality guards for --cfg-deduplicate-state-forward; valid only for runner-controlled non-split CFG batches")
    parser.add_argument("--negative-condition-compression", action="store_true", help="Experimental exact-style CFG path: compress repeated zero negative condition tokens to one representative row with log(M) attention bias")
    parser.add_argument("--negative-condition-assume-identical-rows", action="store_true", help="With --negative-condition-compression, skip per-forward torch.equal verification of repeated negative condition rows; valid only for runner-controlled static zero negative condition")
    parser.add_argument("--negative-condition-logbias-lse-adjust", action="store_true", help="Exact CPU candidate for direct aten SDPA: replace single-key log(M) mask with maskless SDPA plus logsumexp probability correction")
    parser.add_argument("--negative-condition-internal-timing", action="store_true", help="Record inclusive wall-clock timers inside the patched negative-condition compression path")
    parser.add_argument("--negative-condition-full-linear-compressed-sdpa", action="store_true", help="Experimental: use full B=2 Linear/MLP in main blocks while keeping negative-branch SDPA compressed")
    parser.add_argument("--negative-condition-combine-linear", action="store_true", help="Experimental exact CPU candidate: combine positive/negative Linear/MLP rows in compressed CFG main blocks while keeping SDPA branches separate")
    parser.add_argument("--negative-condition-selective-final-block", action="store_true", help="Experimental exact final-block optimization inside negative condition compression: compute only consumed latent/camera rows in the last main block")
    parser.add_argument("--negative-condition-selective-final-positive-only", action="store_true", help="With --negative-condition-selective-final-block, only integrate the positive branch; leave the negative compact branch on its previous full final-block path")
    parser.add_argument("--negative-condition-noise-refiner-inplace-elementwise", action="store_true", help="Exact CPU candidate: apply the same order-preserving in-place modulation/residual updates to noise_refiner blocks inside negative-condition compression")
    parser.add_argument("--negative-condition-positive-compiled-realrope", action="store_true", help="CPU candidate: in negative-condition compression, run positive full main blocks through a shared torch.compile real-RoPE attention path")
    parser.add_argument("--negative-condition-positive-fullblock-compiled-realrope", action="store_true", help="CPU candidate: in negative-condition compression, run positive full main blocks through a shared torch.compile full-block real-RoPE path")
    parser.add_argument("--vae-deterministic", action="store_true")
    attention_backend_choices = ["default", "math", "flash", "chunked", "aten_flash_direct", "aten_flash_direct_scale", "aten_flash_direct_auto", "f_explicit_scale", "native_avx512_exact", "streaming", "streaming_online", "streaming_m4", "streaming_m4_d64", "streaming_m4_d64_k64", "streaming_m8"]
    parser.add_argument("--attention-backend", choices=attention_backend_choices, default="default", help="Runtime patch for TripoSplat full-attention SDPA backend.")
    parser.add_argument("--attention-compute-dtype", choices=["model", "float32"], default="model", help="Runtime patch for attention math dtype; output is cast back to the model dtype.")
    parser.add_argument("--attention-query-chunk-size", type=int, default=128, help="Query chunk size when --attention-backend=chunked")
    parser.add_argument("--attention-contiguous-qkv", action="store_true", help="Exact CPU layout optimization: make permuted [B,H,L,D] q/k/v contiguous before torch SDPA")
    parser.add_argument("--attention-module-include-regex", help="Patch only RopeMultiHeadAttention modules whose full name matches this regex")
    parser.add_argument("--attention-module-exclude-regex", help="Skip module attention patch for names matching this regex")
    parser.add_argument("--attention-module-backend", choices=attention_backend_choices, default="flash", help="SDPA backend for selected module-level attention patch")
    parser.add_argument("--attention-module-compute-dtype", choices=["model", "float32"], default="float32", help="Attention math dtype for selected module-level attention patch")
    parser.add_argument("--attention-module-linear-dtype", choices=["model", "float32"], default="model", help="Cast selected attention modules qkv/out/norm parameters to this dtype and cast inputs locally.")
    parser.add_argument("--attention-module-query-chunk-size", type=int, default=128, help="Query chunk size when --attention-module-backend=chunked")
    parser.add_argument("--selective-final-block", action="store_true", help="Patch the final main block to compute only consumed latent/camera rows exactly")
    parser.add_argument("--selective-final-block-backend", choices=attention_backend_choices, help="SDPA backend for the selective final block patch; defaults to --attention-backend")
    parser.add_argument("--selective-final-block-compute-dtype", choices=["model", "float32"], help="Attention compute dtype for selective final block; defaults to --attention-compute-dtype")
    parser.add_argument("--selective-final-block-query-chunk-size", type=int, default=128, help="Query chunk size when selective final block backend is chunked")
    parser.add_argument("--selective-final-block-round-qkv-to-fp16", action="store_true", help="Round selective final block q and kv Linear outputs through fp16 before RoPE/qk norm")
    parser.add_argument("--selective-final-block-round-v-to-fp16", action="store_true", help="Round selective final block value tensor through fp16 before attention")
    parser.add_argument("--selective-final-block-round-attn-core-to-fp16", action="store_true", help="Round selective final block attention core output through fp16 before attn.out Linear")
    parser.add_argument("--selective-final-block-half-sequence", action="store_true", help="Apply fp16-value roundtrips to norm/modulation/residual/MLP inside the selective final block patch")
    parser.add_argument("--selective-final-block-elementwise-compute-dtype", choices=["roundtrip", "float16"], default="roundtrip", help="Compute selective final block modulation/gate/residual elementwise ops in fp16 instead of only fp16 roundtripping their outputs")
    parser.add_argument("--selective-final-block-inplace-output", action="store_true", help="Inference-only exact optimization: update the final selective block input tensor in place instead of cloning the full sequence")
    parser.add_argument("--late-selective-condition-freeze-blocks", type=int, default=0, help="Non-equivalent speed mode: for the last N main blocks, update only latent/camera rows and keep condition rows frozen while preserving full K/V attention for selected rows. N=1 is equivalent to final consumed-row optimization.")
    parser.add_argument("--late-selective-condition-freeze-backend", choices=attention_backend_choices, default="default")
    parser.add_argument("--late-selective-condition-freeze-compute-dtype", choices=["model", "float32"], default="model")
    parser.add_argument("--late-selective-condition-freeze-query-chunk-size", type=int, default=128)
    parser.add_argument("--unified-block-half-sequence", action="store_true", help="Patch selected UnifiedTransformerBlock forwards with explicit fp16-value roundtrips in official operation order")
    parser.add_argument("--unified-block-half-sequence-include-regex", default=r"^(noise_refiner|context_refiner|blocks)[.][0-9]+$", help="UnifiedTransformerBlock names to patch when --unified-block-half-sequence is set")
    parser.add_argument("--unified-block-half-sequence-exclude-regex", help="UnifiedTransformerBlock names to skip for half-sequence patch")
    parser.add_argument("--unified-block-half-sequence-backend", choices=attention_backend_choices, default="default", help="Attention backend used inside the half-sequence block patch")
    parser.add_argument("--unified-block-half-sequence-compute-dtype", choices=["model", "float32"], default="model", help="Attention compute dtype inside the half-sequence block patch")
    parser.add_argument("--unified-block-half-sequence-query-chunk-size", type=int, default=128, help="Query chunk size when half-sequence block patch uses chunked attention")
    parser.add_argument("--unified-block-half-sequence-elementwise-compute-dtype", choices=["roundtrip", "float16"], default="roundtrip", help="Compute modulation/gate/residual elementwise ops in fp16 instead of only fp16 roundtripping their outputs")
    parser.add_argument("--round-flow-params-to-fp16", action="store_true", help="Round flow-model floating parameters/buffers through fp16 while keeping the runtime dtype")
    parser.add_argument("--round-runtime-state-to-fp16", action="store_true", help="Round loaded/encoded condition and initial noise through fp16 while keeping the runtime dtype")
    parser.add_argument("--round-module-outputs-to-fp16", action="store_true", help="Round selected flow-model module outputs through fp16 while keeping the runtime dtype")
    parser.add_argument("--round-module-output-include-regex", help="Only install fp16 output rounding hooks on module names matching this regex")
    parser.add_argument("--round-module-output-exclude-regex", help="Skip fp16 output rounding hooks on module names matching this regex")
    parser.add_argument("--round-module-inputs-to-fp16", action="store_true", help="Round selected flow-model module inputs through fp16 while keeping the runtime dtype")
    parser.add_argument("--round-module-input-include-regex", help="Only install fp16 input rounding hooks on module names matching this regex")
    parser.add_argument("--round-module-input-exclude-regex", help="Skip fp16 input rounding hooks on module names matching this regex")
    parser.add_argument("--native-half-module-include-regex", help="Run selected modules with native CPU float16 parameters/inputs, then cast outputs back")
    parser.add_argument("--native-half-module-exclude-regex", help="Skip native half module hook for names matching this regex")
    parser.add_argument("--native-half-module-output-dtype", choices=["float32"], default="float32")
    parser.add_argument("--native-half-module-compute-dtype", choices=["float16", "bfloat16"], default="float16", help="Compute dtype used by --native-half-module-* hooks; name kept for compatibility")
    parser.add_argument("--timestep-embedder-functional-half", action="store_true", help="Evaluate t_embedder MLP in half precision without changing parameter dtype or flow_model.dtype")
    parser.add_argument("--timestep-embedder-functional-half-output-dtype", choices=["float32"], default="float32")
    parser.add_argument("--timestep-embedder-functional-half-compute-dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--dynamic-int8-linears", action="store_true", help="Apply torch dynamic qint8 quantization to nn.Linear modules in the flow model")
    parser.add_argument("--dynamic-int8-include-regex", help="Only dynamic-quantize Linear modules whose full name matches this regex")
    parser.add_argument("--dynamic-int8-exclude-regex", help="Skip dynamic-int8 for Linear modules whose full name matches this regex")
    parser.add_argument("--dynamic-int8-qconfig", choices=["default", "per_channel"], default="default")
    parser.add_argument("--mkldnn-fused-mlp", action="store_true", help="CPU-only candidate: fuse first Linear+GELU(tanh) in FeedForwardNet MLPs through mkldnn._linear_pointwise")
    parser.add_argument("--mkldnn-fused-mlp-include-regex", default=r"^(noise_refiner|context_refiner|blocks)[.][0-9]+[.]mlp$")
    parser.add_argument("--mkldnn-fused-mlp-exclude-regex")
    parser.add_argument("--gelu-out-buffer-mlp", action="store_true", help="CPU-only exact candidate: write GELU(tanh) into the first Linear output tensor inside FeedForwardNet MLPs")
    parser.add_argument("--gelu-out-buffer-mlp-include-regex", default=r"^(noise_refiner|context_refiner|blocks)[.][0-9]+[.]mlp$")
    parser.add_argument("--gelu-out-buffer-mlp-exclude-regex")
    parser.add_argument("--chunked-mlp", action="store_true", help="CPU-only exact-style candidate: run FeedForwardNet MLPs with row-chunked addmm(out) and a reusable hidden buffer")
    parser.add_argument("--chunked-mlp-include-regex", default=r"^(noise_refiner|context_refiner|blocks)[.][0-9]+[.]mlp$")
    parser.add_argument("--chunked-mlp-exclude-regex")
    parser.add_argument("--chunked-mlp-large-chunk-rows", type=int, default=4096)
    parser.add_argument("--chunked-mlp-small-chunk-rows", type=int, default=2048)
    parser.add_argument("--chunked-mlp-large-row-threshold", type=int, default=10000)
    parser.add_argument("--chunked-mlp-min-rows", type=int, default=4096)
    parser.add_argument("--chunked-mlp-cache-weight-t", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--linear-output-buffer", action="store_true", help="CPU-only candidate: use torch.addmm(out=shared buffer) for selected large float32 Linear shapes")
    parser.add_argument("--linear-output-buffer-include-regex", help="Only patch Linear modules whose full name matches this regex")
    parser.add_argument("--linear-output-buffer-exclude-regex", help="Skip Linear modules whose full name matches this regex")
    parser.add_argument("--linear-output-buffer-ring-depth", type=int, default=4)
    parser.add_argument("--linear-output-buffer-min-rows", type=int, default=4096)
    parser.add_argument("--numpy-linear", action="store_true", help="CPU-only candidate: use NumPy BLAS for selected large float32 Linear shapes")
    parser.add_argument("--numpy-linear-include-regex", default=r"^blocks[.][0-9]+[.](attn[.](qkv|out)|mlp[.]mlp[.][02])$", help="Only patch Linear modules whose full name matches this regex")
    parser.add_argument("--numpy-linear-exclude-regex", help="Skip NumPy Linear modules whose full name matches this regex")
    parser.add_argument("--numpy-linear-min-rows", type=int, default=4096)
    parser.add_argument("--numpy-linear-patch-mlp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--native-avx512-linear", action="store_true", help="CPU-only exact candidate: replace flow-model nn.Linear calls with native float32 AVX-512/FMA")
    parser.add_argument("--native-avx512-linear-include-regex", default=r".*")
    parser.add_argument("--native-avx512-linear-exclude-regex")
    parser.add_argument("--native-avx512-linear-library", default="artifacts/backends/libtriposplat_gemm_f32_avx512.so")
    parser.add_argument("--native-avx512-linear-threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "2")))
    parser.add_argument("--native-avx512-linear-strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--native-avx512-gelu", action="store_true", help="CPU-only exact candidate: replace flow-model nn.GELU(tanh) with native float32 AVX-512")
    parser.add_argument("--native-avx512-gelu-include-regex", default=r".*")
    parser.add_argument("--native-avx512-gelu-exclude-regex")
    parser.add_argument("--native-avx512-gelu-library", default="artifacts/backends/libtriposplat_gelu_avx512.so")
    parser.add_argument("--native-avx512-gelu-threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "2")))
    parser.add_argument("--native-avx512-gelu-strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--native-avx512-norm-rope", action="store_true", help="CPU-only exact candidate: replace flow-model LayerNorm, MultiHeadRMSNorm and model RoPE with native float32 AVX-512")
    parser.add_argument("--native-avx512-norm-rope-library", default="artifacts/backends/libtriposplat_norm_rope_avx512.so")
    parser.add_argument("--native-avx512-norm-rope-threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "2")))
    parser.add_argument("--native-avx512-norm-rope-strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--native-avx512-silu", action="store_true", help="CPU-only exact candidate: replace flow-model nn.SiLU with native float32 AVX-512")
    parser.add_argument("--native-avx512-silu-library", default="artifacts/backends/libtriposplat_activations_avx512.so")
    parser.add_argument("--native-avx512-silu-threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "2")))
    parser.add_argument("--native-avx512-silu-strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--native-avx512-block-elementwise", action="store_true", help="CPU-only exact candidate: replace transformer modulation and residual updates with native float32 AVX-512")
    parser.add_argument("--native-avx512-block-elementwise-library", default="artifacts/backends/libtriposplat_block_elementwise_avx512.so")
    parser.add_argument("--native-avx512-block-elementwise-threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "2")))
    parser.add_argument("--native-avx512-block-elementwise-strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--native-avx512-repo", action="store_true", help="CPU-only exact candidate: replace remaining RePo feature multiplication and complex phasor construction with native float32 AVX-512")
    parser.add_argument("--native-avx512-repo-library", default="artifacts/backends/libtriposplat_repo_avx512.so")
    parser.add_argument("--native-avx512-repo-threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "2")))
    parser.add_argument("--native-avx512-repo-strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--native-avx512-embeddings", action="store_true", help="CPU-only exact candidate: replace fixed-position and timestep trigonometric embeddings with native float32 AVX-512")
    parser.add_argument("--native-avx512-embeddings-library", default="artifacts/backends/libtriposplat_embeddings_avx512.so")
    parser.add_argument("--native-avx512-embeddings-threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "2")))
    parser.add_argument("--native-avx512-embeddings-strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--native-avx512-sampler", action="store_true", help="CPU-only exact candidate: fuse CFG and Euler/AB2 state updates with native float32 AVX-512")
    parser.add_argument("--native-avx512-sampler-library", default="artifacts/backends/libtriposplat_sampler_avx512.so")
    parser.add_argument("--native-avx512-sampler-threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "2")))
    parser.add_argument("--native-avx512-sampler-strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--u7s8-linear", action="store_true", help="CPU-only candidate: use external AVX2 u7/s8 GEMM for selected large float32 Linear shapes")
    parser.add_argument("--u7s8-linear-include-regex", default=r"^blocks[.][0-9]+[.](attn[.](qkv|out)|mlp[.]mlp[.][02])$", help="Only patch Linear modules whose full name matches this regex")
    parser.add_argument("--u7s8-linear-exclude-regex", help="Skip u7/s8 Linear modules whose full name matches this regex")
    parser.add_argument("--u7s8-linear-min-rows", type=int, default=4096)
    parser.add_argument("--u7s8-linear-patch-mlp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--u7s8-linear-library", default="artifacts/backends/libtriposplat_gemm_i8_avx2.so")
    parser.add_argument("--u7s8-linear-threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "2")))
    parser.add_argument("--int4-nonlinear-linear", action="store_true", help="CPU candidate: replace selected Linear modules with packed int4 nonlinear codebook ctypes GEMM")
    parser.add_argument("--int4-nonlinear-linear-include-regex", default=r"^blocks[.](5)[.]attn[.]out$", help="Only patch Linear modules whose full name matches this regex")
    parser.add_argument("--int4-nonlinear-linear-exclude-regex", help="Skip int4 nonlinear Linear modules whose full name matches this regex")
    parser.add_argument("--int4-nonlinear-linear-mode", choices=["linear_symmetric", "linear_affine", "log_symmetric", "mulaw", "kmeans"], default="log_symmetric")
    parser.add_argument("--int4-nonlinear-linear-group-size", type=int, default=32)
    parser.add_argument("--int4-nonlinear-linear-percentile", type=float, default=0.999)
    parser.add_argument("--int4-nonlinear-linear-mu", type=float, default=255.0)
    parser.add_argument("--int4-nonlinear-linear-kmeans-iters", type=int, default=6)
    parser.add_argument("--int4-nonlinear-linear-min-in-features", type=int, default=1)
    parser.add_argument("--int4-nonlinear-linear-max-linears", type=int)
    parser.add_argument("--int4-nonlinear-linear-kernel", choices=["dot", "accum16", "accum16_m4", "accum16_m8", "accum16_m16", "accum16_m16_n4"], default="accum16_m4")
    parser.add_argument("--int4-nonlinear-linear-threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "2")))
    parser.add_argument("--mixed-pc8-linear", action="store_true", help="CPU candidate: use packed mixed pc8 AVX2 GEMM for selected Linear modules")
    parser.add_argument("--mixed-pc8-linear-include-regex", default=r"^blocks[.](5)[.]attn[.]out$")
    parser.add_argument("--mixed-pc8-linear-exclude-regex")
    parser.add_argument("--mixed-pc8-linear-group-mask-npz", type=Path, help="Explicit row x input-group keep mask NPZ for packed mixed pc8 Linear")
    parser.add_argument("--mixed-pc8-linear-group-size", type=int, default=32)
    parser.add_argument("--mixed-pc8-linear-library", default="artifacts/backends/libtriposplat_mixed_pc8_avx2.so")
    parser.add_argument("--mixed-pc8-linear-kernel-variant", choices=["v0_scalar_n", "n4_blocked", "n8_blocked", "n8_oneq_mask", "n8_oneq_mask_mr2", "n8_twq_mask", "n8_tile_allkeep_mr2", "n8_tile_hot_mr2", "n8_tile_hot_replace_mr2", "n8_tile_hot_replace_mr4", "n8_tile_hot_replace_mr4_mblock", "n8_tile_hot_replace_mr4_cold_mr4", "n8_tile_hot_replace_mr8", "n8_tile_hot_replace_mr8_runs", "n8_oneq_schedule"], default="n8_oneq_mask")
    parser.add_argument("--mixed-pc8-linear-min-rows", type=int, default=1)
    parser.add_argument("--mixed-pc8-linear-threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "2")))
    parser.add_argument("--mixed-pc8-linear-fallback-steps", default="", help="Comma/range list of sampler steps that should use original float Linear instead of mixed pc8, e.g. '0,3-4'")
    parser.add_argument("--mixed-pc8-linear-residual-correction-mode", choices=["none", "svd", "activation_svd"], default="none", help="Runtime: add low-rank residual correction after mixed pc8 GEMM")
    parser.add_argument("--mixed-pc8-linear-residual-correction-rank", type=int, default=0, help="Rank for --mixed-pc8-linear-residual-correction-mode")
    parser.add_argument("--mixed-pc8-linear-residual-correction-calibration-npz", type=Path, help="Activation covariance NPZ for mixed pc8 activation_svd residual correction")
    parser.add_argument("--mixed-pc8-linear-residual-correction-factors-npz", type=Path, help="Load prepacked left/right residual correction factors for mixed pc8 runtime")
    parser.add_argument("--mixed-pc8-linear-save-residual-correction-factors-npz", type=Path, help="Save generated left/right residual correction factors for later runs")
    parser.add_argument("--mixed-pc8-linear-residual-correction-gemm-library", help="Optional f32 GEMM ctypes library for mixed pc8 low-rank residual correction")
    parser.add_argument("--mixed-pc8-linear-residual-correction-gemm-symbol", default="triposplat_gemm_f32_avx2", help="ctypes symbol for --mixed-pc8-linear-residual-correction-gemm-library")
    parser.add_argument("--mixed-pc8-linear-residual-correction-fused-library", help="Optional fused AVX2 ctypes library for mixed pc8 low-rank residual correction")
    parser.add_argument("--mixed-pc8-linear-residual-correction-fused-symbol", default="triposplat_lowrank_residual_f32_avx2_add", help="ctypes symbol for --mixed-pc8-linear-residual-correction-fused-library")
    parser.add_argument("--dequant-weight-linear", action="store_true", help="Quality probe: quantize selected Linear weights once and dequantize back to float32 before normal GEMM")
    parser.add_argument("--dequant-weight-linear-include-regex", default=r"^blocks[.][0-9]+[.](attn[.](qkv|out)|mlp[.]mlp[.][02])$", help="Only dequantize Linear weights whose full name matches this regex")
    parser.add_argument("--dequant-weight-linear-exclude-regex", help="Skip dequantized-weight Linear modules whose full name matches this regex")
    parser.add_argument("--dequant-weight-linear-mode", choices=["linear_per_channel_symmetric", "linear_per_tensor_symmetric", "linear_per_channel_group_symmetric", "mulaw_per_channel_symmetric"], default="linear_per_channel_symmetric")
    parser.add_argument("--dequant-weight-linear-bits", type=int, default=8)
    parser.add_argument("--dequant-weight-linear-group-size", type=int, default=128, help="Input-channel group size for linear_per_channel_group_symmetric weight dequantization")
    parser.add_argument("--dequant-weight-linear-percentile", type=float, default=1.0)
    parser.add_argument("--dequant-weight-linear-mu", type=float, default=255.0)
    parser.add_argument("--dequant-weight-linear-external-npz", type=Path, help="Quality probe: load pre-reconstructed per-module dequant weights from NPZ instead of using the built-in quant/dequant mode")
    parser.add_argument("--dequant-weight-linear-patch-mlp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dequant-weight-linear-mixed-keep-ratio", type=float, default=0.0, help="Quality probe: keep this fraction of output rows at original float weight after quant/dequant, ranked by --dequant-weight-linear-mixed-rank")
    parser.add_argument("--dequant-weight-linear-mixed-keep-count", type=int, default=0, help="Quality probe: keep at least this many output rows at original float weight after quant/dequant")
    parser.add_argument("--dequant-weight-linear-mixed-rank", choices=["row_rmse_error", "row_max_abs_error", "row_weight_norm", "row_output_residual_error"], default="row_rmse_error")
    parser.add_argument("--dequant-weight-linear-mixed-output-residual-npz", type=Path, help="Output residual NPZ for --dequant-weight-linear-mixed-rank=row_output_residual_error")
    parser.add_argument("--dequant-weight-linear-mixed-output-residual-step-weights", default="", help="Comma weights for output residual steps, e.g. '1,1,2,4' or '1:1,2:1,3:2,4:4'")
    parser.add_argument("--dequant-weight-linear-mixed-group-size", type=int, default=16, help="Quality probe: input-channel group width for group-level mixed precision residual restore")
    parser.add_argument("--dequant-weight-linear-mixed-group-keep-ratio", type=float, default=0.0, help="Quality probe: keep this fraction of row x input-channel groups at original float weight after quant/dequant")
    parser.add_argument("--dequant-weight-linear-mixed-group-keep-count", type=int, default=0, help="Quality probe: keep at least this many row x input-channel groups at original float weight after quant/dequant")
    parser.add_argument("--dequant-weight-linear-mixed-group-rank", choices=["group_rmse_error", "group_max_abs_error", "group_weight_norm", "group_activation_error", "group_keep_mask"], default="group_rmse_error")
    parser.add_argument("--dequant-weight-linear-mixed-group-calibration-npz", type=Path, help="Activation covariance NPZ for --dequant-weight-linear-mixed-group-rank=group_activation_error")
    parser.add_argument("--dequant-weight-linear-mixed-group-mask-npz", type=Path, help="Explicit row x input-group keep mask NPZ for --dequant-weight-linear-mixed-group-rank=group_keep_mask")
    parser.add_argument("--dequant-weight-linear-residual-correction-mode", choices=["none", "svd", "activation_svd"], default="none", help="Quality probe: add a low-rank approximation of the remaining float-weight residual after quant/dequant")
    parser.add_argument("--dequant-weight-linear-residual-correction-rank", type=int, default=0, help="Rank for --dequant-weight-linear-residual-correction-mode=svd or activation_svd")
    parser.add_argument("--dequant-weight-linear-residual-correction-calibration-npz", type=Path, help="Activation covariance NPZ for --dequant-weight-linear-residual-correction-mode=activation_svd")
    parser.add_argument("--capture-linear-input-covariance-npz", type=Path, help="Write input activation covariance NPZ for selected Linear modules during sampling")
    parser.add_argument("--capture-linear-input-covariance-include-regex", default=r"^blocks[.](5)[.]attn[.]out$", help="Linear module names to capture for activation covariance")
    parser.add_argument("--capture-linear-input-covariance-exclude-regex", help="Skip covariance capture modules matching this regex")
    parser.add_argument("--capture-linear-input-covariance-group-size", type=int, default=16)
    parser.add_argument("--capture-linear-input-covariance-by-step", action="store_true", help="Split captured Linear input covariance by sampler step using the pre_forward callback step context")
    parser.add_argument("--capture-linear-output-residual-npz", type=Path, help="Write selected Linear output residual statistics for float weight minus pc8 dequantized weight during sampling")
    parser.add_argument("--capture-linear-output-residual-include-regex", default=r"^blocks[.](5)[.]attn[.]out$", help="Linear module names to capture for output residual statistics")
    parser.add_argument("--capture-linear-output-residual-exclude-regex", help="Skip output residual capture modules matching this regex")
    parser.add_argument("--capture-linear-output-residual-bits", type=int, default=8)
    parser.add_argument("--capture-linear-output-residual-mode", choices=["linear_per_channel_symmetric"], default="linear_per_channel_symmetric")
    parser.add_argument("--capture-linear-output-residual-by-step", action="store_true", help="Split captured Linear output residual statistics by sampler step using the pre_forward callback step context")
    parser.add_argument("--addcmul-elementwise", action="store_true", help="Candidate: use torch.addcmul for main-block modulation and gated residual elementwise ops")
    parser.add_argument("--addcmul-elementwise-include-regex", default=r"^blocks[.][0-9]+$", help="UnifiedTransformerBlock names to patch when --addcmul-elementwise is set")
    parser.add_argument("--addcmul-elementwise-exclude-regex", help="Skip addcmul elementwise patch for names matching this regex")
    parser.add_argument("--torch-compile-flow", action="store_true", help="Wrap the flow model with torch.compile before sampling.")
    parser.add_argument("--torch-compile-backend", default="inductor", help="torch.compile backend, e.g. inductor, eager, or aot_eager.")
    parser.add_argument("--torch-compile-mode", choices=["none", "default", "reduce-overhead", "max-autotune"], default="reduce-overhead")
    parser.add_argument("--torch-compile-fullgraph", action="store_true")
    parser.add_argument("--torch-compile-dynamic", action="store_true")
    parser.add_argument("--torch-compile-warmup", action=argparse.BooleanOptionalAction, default=True, help="Run one shape-matched warmup forward so compile time is recorded separately from sampler timing.")
    parser.add_argument("--cfg-split-forward", action="store_true", help="Run CFG cond/uncond model forwards separately instead of as a single batch=2 forward. This preserves CFG math and may help CPU cache behavior.")
    parser.add_argument("--cfg-single-state-forward", action="store_true", help="Exact batched-CFG optimization used with --cfg-deduplicate-state-forward: pass state batch N with condition/timestep batch 2N instead of materializing [state,state].")
    parser.add_argument("--negative-condition-inplace-elementwise", action="store_true", help="CPU exact candidate inside negative-condition compression: use mul_/add_ for block modulation and gated residuals while preserving operation order.")
    parser.add_argument("--negative-condition-parallel-branches", action="store_true", help="CPU multi-core candidate: run positive/negative compressed CFG branches concurrently inside main blocks. Default off; quality gate required.")
    parser.add_argument("--negative-condition-parallel-branch-workers", type=int, default=2)
    parser.add_argument("--condition-token-mode", choices=["none", "grid"], default="none", help="Non-equivalent speed mode: reduce condition tokens before flow sampling.")
    parser.add_argument("--condition-patch-grid-size", type=int, help="Target patch grid for --condition-token-mode=grid. 1024 conditions are prefix+64x64 patches; e.g. 32 keeps 1024 patch tokens.")
    parser.add_argument("--condition-prefix-tokens", type=int, default=5, help="Condition prefix/register tokens to preserve before grid token reduction.")
    parser.add_argument("--fake-quant-mode", choices=["none", "linear_symmetric", "linear_affine", "log_symmetric", "mulaw", "float16_roundtrip"], default="none")
    parser.add_argument("--fake-quant-bits", type=int, default=8)
    parser.add_argument("--fake-quant-percentile", type=float, default=0.999)
    parser.add_argument("--fake-quant-mu", type=float, default=255.0)
    parser.add_argument(
        "--fake-quant-points",
        default="state,prediction",
        help="Comma list from state,condition,prediction,cfg_prediction,updated_state. Non-linear modes are useful for state/prediction.",
    )
    parser.add_argument("--deterministic-torch", action="store_true")
    parser.add_argument("--disable-mkldnn", action="store_true")
    parser.add_argument("--trace-json", type=Path, help="Write lightweight flow step trace JSON for CPU/GPU equivalence debugging")
    parser.add_argument("--operator-audit-json", type=Path, help="Write aggregated CPU operator counts/times for native-path completeness auditing")
    parser.add_argument("--trace-max-steps", type=int, default=2)
    parser.add_argument("--trace-max-values", type=int, default=256)
    parser.add_argument("--trace-stats-sample-values", type=int, default=4096)
    parser.add_argument("--trace-events", default="", help="Comma list of trace events to keep; empty keeps all")
    parser.add_argument("--trace-store-values", action="store_true", help="Store sampled float32 values in trace JSON for sample RMSE comparison")
    parser.add_argument("--step-state-dir", type=Path, help="Write full per-step latent/camera NPZ states for CPU/GPU equivalence debugging")
    parser.add_argument("--step-state-events", default="post_update", help="Comma list from pre_forward,post_cfg_prediction,post_update")
    parser.add_argument("--step-state-max-steps", type=int, default=0, help="Maximum steps to save; 0 saves all selected events")
    parser.add_argument("--step-state-storage-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--module-trace-json", type=Path, help="Write lightweight module forward trace JSON")
    parser.add_argument("--module-trace-include-regex", default=r"^(noise_refiner|context_refiner|blocks)\.[0-9]+\.attn$")
    parser.add_argument("--module-trace-exclude-regex")
    parser.add_argument("--module-trace-max-modules", type=int, default=64)
    parser.add_argument("--module-trace-max-calls", type=int, default=1)
    parser.add_argument("--module-trace-max-values", type=int, default=128)
    parser.add_argument("--module-trace-stats-sample-values", type=int, default=1024)
    parser.add_argument("--module-trace-store-values", action="store_true")
    parser.add_argument("--module-trace-inputs", action="store_true")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True, help="Show tqdm sampler progress. Disable with --no-progress for runtime-overhead checks.")
    parser.add_argument("--flush-denormal", action=argparse.BooleanOptionalAction, default=False, help="Call torch.set_flush_denormal(True) before model execution. Exact-ish runtime knob; quality gate decides adoption.")
    parser.add_argument(
        "--float32-matmul-precision",
        choices=["highest", "high", "medium"],
        help="Call torch.set_float32_matmul_precision(...) before model execution. Exactness is validated by the quality gate.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(SCRIPT_DIR))
    sys.path.insert(0, str(REPO))

    import torch
    import triposplat
    from PIL import Image
    from triposplat import TripoSplatPipeline, load_dinov3, load_flow_model, load_vae_encoder

    from run_triposplat_encoded_external_noise import (
        build_noise,
        encode_image_controlled,
        load_condition_npz,
        resolve_dtype,
        save_noise_npz,
        save_tensor_dict_npz,
        tensor_dict_sha256,
        utc_now,
    )
    from triposplat_quantized_sampler import (
        FakeQuantConfig,
        FlowEulerCfgMultiVariantSampler,
        dynamic_quantize_linear_modules,
        parse_flow_variants,
    )
    from triposplat_attention_patch import (
        apply_triposplat_attention_patch,
        apply_triposplat_module_attention_patch,
        apply_triposplat_selective_final_block_patch,
        apply_triposplat_late_selective_condition_freeze_patch,
        apply_triposplat_cfg_duplicate_state_patch,
        apply_triposplat_negative_condition_compression_patch,
        apply_triposplat_static_condition_cache_patch,
        apply_triposplat_position_embed_cache_patch,
        apply_triposplat_repo_polar_cos_sin_patch,
        apply_triposplat_unified_block_half_sequence_patch,
        apply_triposplat_mkldnn_fused_mlp_patch,
        apply_triposplat_gelu_out_buffer_mlp_patch,
        apply_triposplat_chunked_mlp_patch,
        apply_triposplat_linear_output_buffer_patch,
        apply_triposplat_numpy_linear_patch,
        apply_triposplat_u7s8_linear_patch,
        apply_triposplat_mixed_pc8_linear_patch,
        apply_triposplat_dequantized_weight_linear_patch,
        apply_triposplat_addcmul_elementwise_patch,
        warmup_triposplat_positive_fullblock_compiled_realrope,
        warmup_triposplat_positive_final_selected_compiled_realrope,
        warmup_triposplat_negative_fullblock_compiled_realrope,
        make_triposplat_cached_condition,
    )
    from native_linear_avx512_range_patch import apply_triposplat_native_linear_avx512_patch
    from native_gelu_avx512_patch import apply_triposplat_native_gelu_avx512_patch
    from native_norm_rope_avx512_patch import apply_triposplat_native_norm_rope_avx512_patch
    from native_silu_avx512_patch import apply_triposplat_native_silu_avx512_patch
    from native_block_elementwise_avx512_patch import apply_triposplat_native_block_elementwise_avx512_patch
    from native_repo_avx512_patch import apply_triposplat_native_repo_avx512_patch
    from native_embeddings_avx512_patch import apply_triposplat_native_embeddings_avx512_patch
    from native_sampler_avx512_backend import create_triposplat_native_sampler_avx512_backend
    from flow_trace_utils import FlowTraceRecorder
    from module_trace_utils import ModuleTraceRecorder
    from step_state_recorder import StepStateRecorder

    if args.deterministic_torch:
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
        torch.use_deterministic_algorithms(True, warn_only=True)
    if args.disable_mkldnn and hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.dynamic_int8_linears and args.device != "cpu":
        raise ValueError("--dynamic-int8-linears is CPU-only in this runner")
    if args.mkldnn_fused_mlp and args.device != "cpu":
        raise ValueError("--mkldnn-fused-mlp is CPU-only")
    if args.mkldnn_fused_mlp and args.dynamic_int8_linears:
        raise ValueError("--mkldnn-fused-mlp is not combined with --dynamic-int8-linears in this runner")
    if args.gelu_out_buffer_mlp and args.device != "cpu":
        raise ValueError("--gelu-out-buffer-mlp is CPU-only")
    if args.gelu_out_buffer_mlp and args.dynamic_int8_linears:
        raise ValueError("--gelu-out-buffer-mlp requires non-quantized nn.Linear modules")
    if args.gelu_out_buffer_mlp and args.mkldnn_fused_mlp:
        raise ValueError("--gelu-out-buffer-mlp is not combined with --mkldnn-fused-mlp in this runner")
    if args.chunked_mlp and args.device != "cpu":
        raise ValueError("--chunked-mlp is CPU-only")
    if args.chunked_mlp and args.dynamic_int8_linears:
        raise ValueError("--chunked-mlp requires non-quantized nn.Linear modules")
    if args.chunked_mlp and args.mkldnn_fused_mlp:
        raise ValueError("--chunked-mlp is not combined with --mkldnn-fused-mlp in this runner")
    if args.chunked_mlp and args.gelu_out_buffer_mlp:
        raise ValueError("--chunked-mlp is not combined with --gelu-out-buffer-mlp in this runner")
    if args.chunked_mlp and args.numpy_linear:
        raise ValueError("--chunked-mlp is not combined with --numpy-linear in this runner")
    if args.linear_output_buffer and args.device != "cpu":
        raise ValueError("--linear-output-buffer is CPU-only")
    if args.linear_output_buffer and args.dynamic_int8_linears:
        raise ValueError("--linear-output-buffer requires non-quantized nn.Linear modules")
    if args.linear_output_buffer and args.mkldnn_fused_mlp:
        raise ValueError("--linear-output-buffer is not combined with --mkldnn-fused-mlp in this runner")
    if args.numpy_linear and args.device != "cpu":
        raise ValueError("--numpy-linear is CPU-only")
    if args.numpy_linear and args.dynamic_int8_linears:
        raise ValueError("--numpy-linear is not combined with --dynamic-int8-linears in this runner")
    if args.numpy_linear and args.linear_output_buffer:
        raise ValueError("--numpy-linear is not combined with --linear-output-buffer in this runner")
    if args.numpy_linear and args.mkldnn_fused_mlp:
        raise ValueError("--numpy-linear is not combined with --mkldnn-fused-mlp in this runner")
    if args.u7s8_linear and args.device != "cpu":
        raise ValueError("--u7s8-linear is CPU-only")
    if args.u7s8_linear and args.dynamic_int8_linears:
        raise ValueError("--u7s8-linear is not combined with --dynamic-int8-linears in this runner")
    if args.u7s8_linear and args.linear_output_buffer:
        raise ValueError("--u7s8-linear is not combined with --linear-output-buffer in this runner")
    if args.u7s8_linear and args.numpy_linear:
        raise ValueError("--u7s8-linear is not combined with --numpy-linear in this runner")
    if args.u7s8_linear and args.mkldnn_fused_mlp:
        raise ValueError("--u7s8-linear is not combined with --mkldnn-fused-mlp in this runner")
    if args.int4_nonlinear_linear and args.device != "cpu":
        raise ValueError("--int4-nonlinear-linear is CPU-only")
    if args.int4_nonlinear_linear and args.dynamic_int8_linears:
        raise ValueError("--int4-nonlinear-linear requires non-quantized nn.Linear modules")
    if args.int4_nonlinear_linear and args.dequant_weight_linear:
        raise ValueError("--int4-nonlinear-linear is not combined with --dequant-weight-linear; compare them in separate runs")
    if args.int4_nonlinear_linear and args.mixed_pc8_linear:
        raise ValueError("--int4-nonlinear-linear is not combined with --mixed-pc8-linear; compare them in separate runs")
    if args.int4_nonlinear_linear and args.u7s8_linear:
        raise ValueError("--int4-nonlinear-linear is not combined with --u7s8-linear")
    if args.int4_nonlinear_linear and args.numpy_linear:
        raise ValueError("--int4-nonlinear-linear is not combined with --numpy-linear")
    if args.int4_nonlinear_linear and args.linear_output_buffer:
        raise ValueError("--int4-nonlinear-linear is not combined with --linear-output-buffer")
    if args.mixed_pc8_linear and args.device != "cpu":
        raise ValueError("--mixed-pc8-linear is CPU-only")
    if args.mixed_pc8_linear and args.dynamic_int8_linears:
        raise ValueError("--mixed-pc8-linear requires non-quantized nn.Linear modules")
    if args.mixed_pc8_linear and args.dequant_weight_linear:
        raise ValueError("--mixed-pc8-linear is not combined with --dequant-weight-linear; compare them in separate runs")
    if args.mixed_pc8_linear and args.u7s8_linear:
        raise ValueError("--mixed-pc8-linear is not combined with --u7s8-linear")
    if args.mixed_pc8_linear and args.numpy_linear:
        raise ValueError("--mixed-pc8-linear is not combined with --numpy-linear")
    if args.mixed_pc8_linear and args.linear_output_buffer:
        raise ValueError("--mixed-pc8-linear is not combined with --linear-output-buffer")
    if args.mixed_pc8_linear and args.mixed_pc8_linear_group_mask_npz is None:
        raise ValueError("--mixed-pc8-linear requires --mixed-pc8-linear-group-mask-npz")
    if args.dequant_weight_linear and args.dynamic_int8_linears:
        raise ValueError("--dequant-weight-linear requires non-quantized nn.Linear modules")
    if args.dequant_weight_linear and args.device != "cpu":
        raise ValueError("--dequant-weight-linear is currently a CPU calibration probe")
    if args.selective_final_block and args.dynamic_int8_linears:
        include = args.dynamic_int8_include_regex or ""
        if r"mlp\.mlp" not in include:
            raise ValueError("--selective-final-block with --dynamic-int8-linears is only enabled for explicit MLP-only include regexes")
    if int(args.late_selective_condition_freeze_blocks) > 0 and args.selective_final_block:
        raise ValueError("--late-selective-condition-freeze-blocks and --selective-final-block both patch main blocks; use one or the other")
    if int(args.late_selective_condition_freeze_blocks) > 0 and args.dynamic_int8_linears:
        raise ValueError("--late-selective-condition-freeze-blocks currently requires non-quantized nn.Linear modules")
    if args.round_flow_params_to_fp16 and args.dynamic_int8_linears:
        raise ValueError("--round-flow-params-to-fp16 must run before/without dynamic-int8 replacement")
    if args.round_module_outputs_to_fp16 and args.dynamic_int8_linears:
        raise ValueError("--round-module-outputs-to-fp16 currently requires non-quantized nn.Module outputs")
    if args.round_module_inputs_to_fp16 and args.dynamic_int8_linears:
        raise ValueError("--round-module-inputs-to-fp16 currently requires non-quantized nn.Module inputs")
    interop_threads_requested = _configure_torch_interop_threads(torch)
    if args.device == "cpu":
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    flush_denormal_meta = {"enabled": bool(args.flush_denormal), "applied": None}
    if args.flush_denormal:
        flush_denormal_meta["applied"] = bool(torch.set_flush_denormal(True))

    float32_matmul_precision_meta = {
        "enabled": args.float32_matmul_precision is not None,
        "requested": args.float32_matmul_precision,
        "previous": None,
        "effective": None,
    }
    if args.float32_matmul_precision is not None:
        if hasattr(torch, "get_float32_matmul_precision"):
            float32_matmul_precision_meta["previous"] = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision(args.float32_matmul_precision)
        if hasattr(torch, "get_float32_matmul_precision"):
            float32_matmul_precision_meta["effective"] = torch.get_float32_matmul_precision()

    variants = parse_flow_variants(args.variants, args.steps, args.guidance_scale, args.shift)
    quantize_points = _parse_quantize_points(args.fake_quant_points)
    if args.static_condition_cache and args.fake_quant_mode != "none" and "condition" in quantize_points:
        raise ValueError("--static-condition-cache cannot be combined with fake quantization at the condition boundary")
    if args.cfg_deduplicate_state_forward and args.cfg_split_forward:
        raise ValueError("--cfg-deduplicate-state-forward requires batched CFG; do not combine it with --cfg-split-forward")
    if args.cfg_single_state_forward and args.cfg_split_forward:
        raise ValueError("--cfg-single-state-forward requires non-split batched CFG")
    if args.cfg_single_state_forward and not args.cfg_deduplicate_state_forward:
        raise ValueError("--cfg-single-state-forward requires --cfg-deduplicate-state-forward")
    if args.negative_condition_compression and not args.cfg_deduplicate_state_forward:
        raise ValueError("--negative-condition-compression requires --cfg-deduplicate-state-forward")
    if args.negative_condition_compression and not args.static_condition_cache:
        raise ValueError("--negative-condition-compression requires --static-condition-cache")
    if args.negative_condition_compression and args.cfg_split_forward:
        raise ValueError("--negative-condition-compression requires non-split batched CFG")
    if args.negative_condition_logbias_lse_adjust and not args.negative_condition_compression:
        raise ValueError("--negative-condition-logbias-lse-adjust requires --negative-condition-compression")
    if args.negative_condition_logbias_lse_adjust and not args.attention_backend.startswith("aten_flash_direct"):
        raise ValueError("--negative-condition-logbias-lse-adjust requires an aten_flash_direct attention backend")
    if args.negative_condition_full_linear_compressed_sdpa and not args.negative_condition_compression:
        raise ValueError("--negative-condition-full-linear-compressed-sdpa requires --negative-condition-compression")
    if args.negative_condition_combine_linear and not args.negative_condition_compression:
        raise ValueError("--negative-condition-combine-linear requires --negative-condition-compression")
    if args.negative_condition_selective_final_block and not args.negative_condition_compression:
        raise ValueError("--negative-condition-selective-final-block requires --negative-condition-compression")
    if args.negative_condition_selective_final_positive_only and not args.negative_condition_selective_final_block:
        raise ValueError("--negative-condition-selective-final-positive-only requires --negative-condition-selective-final-block")
    if args.negative_condition_compression and args.cfg_single_state_forward:
        raise ValueError("--negative-condition-compression currently supports duplicated [state,state] CFG, not single-state CFG")
    if args.negative_condition_parallel_branches and not args.negative_condition_compression:
        raise ValueError("--negative-condition-parallel-branches requires --negative-condition-compression")
    if int(args.negative_condition_parallel_branch_workers) < 2:
        raise ValueError("--negative-condition-parallel-branch-workers must be >= 2")
    if args.cfg_deduplicate_state_assume_duplicated and not args.cfg_deduplicate_state_forward:
        raise ValueError("--cfg-deduplicate-state-assume-duplicated requires --cfg-deduplicate-state-forward")
    fake_quant = FakeQuantConfig(
        mode=args.fake_quant_mode,
        bits=int(args.fake_quant_bits),
        percentile=float(args.fake_quant_percentile),
        mu=float(args.fake_quant_mu),
    )

    triposplat._CANVAS_SIZE = int(args.canvas_size)
    global_attention_patch_meta = apply_triposplat_attention_patch(
        backend=args.attention_backend,
        compute_dtype=args.attention_compute_dtype,
        query_chunk_size=int(args.attention_query_chunk_size),
        contiguous_qkv=bool(args.attention_contiguous_qkv),
    )
    attention_patch_meta = {"global": global_attention_patch_meta}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = resolve_dtype(args.model_dtype, device.type)
    t0 = time.time()

    pipe = TripoSplatPipeline.__new__(TripoSplatPipeline)
    pipe._device = device
    pipe.rmbg = None
    pipe.decoder = None
    pipe.dinov3 = None
    pipe.vae_encoder = None
    if args.condition_npz is None:
        pipe.dinov3 = load_dinov3(str(CKPTS / "clip_vision/dino_v3_vit_h.safetensors"), device=device, dtype=dtype)
        pipe.vae_encoder = load_vae_encoder(str(CKPTS / "vae/flux2-vae.safetensors"), device=device, dtype=dtype)
    pipe.flow_model = load_flow_model(str(CKPTS / "diffusion_models/triposplat_fp16.safetensors"), device=device, dtype=dtype)
    pipe.flow_model.eval()
    flow_param_rounding_meta = {"enabled": False}
    if args.round_flow_params_to_fp16:
        flow_param_rounding_meta = _round_module_floating_tensors_(
            pipe.flow_model,
            torch,
            storage_dtype=torch.float16,
        )
    module_attention_patch_meta = apply_triposplat_module_attention_patch(
        pipe.flow_model,
        include_regex=args.attention_module_include_regex,
        exclude_regex=args.attention_module_exclude_regex,
        backend=args.attention_module_backend,
        compute_dtype=args.attention_module_compute_dtype,
        query_chunk_size=int(args.attention_module_query_chunk_size),
        linear_dtype=args.attention_module_linear_dtype,
    )
    attention_patch_meta = {
        "global": global_attention_patch_meta,
        "module": module_attention_patch_meta,
    }

    quantized_model, dynamic_quant_meta = dynamic_quantize_linear_modules(
        pipe.flow_model,
        bool(args.dynamic_int8_linears),
        include_regex=args.dynamic_int8_include_regex,
        exclude_regex=args.dynamic_int8_exclude_regex,
        qconfig_name=args.dynamic_int8_qconfig,
    )
    pipe.flow_model = quantized_model.to(device) if not args.dynamic_int8_linears else quantized_model
    mkldnn_fused_mlp_meta = apply_triposplat_mkldnn_fused_mlp_patch(
        pipe.flow_model,
        enabled=bool(args.mkldnn_fused_mlp),
        include_regex=args.mkldnn_fused_mlp_include_regex,
        exclude_regex=args.mkldnn_fused_mlp_exclude_regex,
    )
    gelu_out_buffer_mlp_meta = apply_triposplat_gelu_out_buffer_mlp_patch(
        pipe.flow_model,
        enabled=bool(args.gelu_out_buffer_mlp),
        include_regex=args.gelu_out_buffer_mlp_include_regex,
        exclude_regex=args.gelu_out_buffer_mlp_exclude_regex,
    )
    chunked_mlp_meta = apply_triposplat_chunked_mlp_patch(
        pipe.flow_model,
        enabled=bool(args.chunked_mlp),
        include_regex=args.chunked_mlp_include_regex,
        exclude_regex=args.chunked_mlp_exclude_regex,
        large_chunk_rows=int(args.chunked_mlp_large_chunk_rows),
        small_chunk_rows=int(args.chunked_mlp_small_chunk_rows),
        large_row_threshold=int(args.chunked_mlp_large_row_threshold),
        min_rows=int(args.chunked_mlp_min_rows),
        cache_weight_t=bool(args.chunked_mlp_cache_weight_t),
    )
    linear_output_buffer_meta = apply_triposplat_linear_output_buffer_patch(
        pipe.flow_model,
        enabled=bool(args.linear_output_buffer),
        include_regex=args.linear_output_buffer_include_regex,
        exclude_regex=args.linear_output_buffer_exclude_regex,
        ring_depth=int(args.linear_output_buffer_ring_depth),
        min_rows=int(args.linear_output_buffer_min_rows),
    )
    numpy_linear_meta = apply_triposplat_numpy_linear_patch(
        pipe.flow_model,
        enabled=bool(args.numpy_linear),
        include_regex=args.numpy_linear_include_regex,
        exclude_regex=args.numpy_linear_exclude_regex,
        min_rows=int(args.numpy_linear_min_rows),
        patch_mlp=bool(args.numpy_linear_patch_mlp),
    )
    dequant_weight_linear_meta = apply_triposplat_dequantized_weight_linear_patch(
        pipe.flow_model,
        enabled=bool(args.dequant_weight_linear),
        include_regex=args.dequant_weight_linear_include_regex,
        exclude_regex=args.dequant_weight_linear_exclude_regex,
        bits=int(args.dequant_weight_linear_bits),
        mode=args.dequant_weight_linear_mode,
        percentile=float(args.dequant_weight_linear_percentile),
        mu=float(args.dequant_weight_linear_mu),
        patch_mlp=bool(args.dequant_weight_linear_patch_mlp),
        group_size=int(args.dequant_weight_linear_group_size),
        external_weight_npz=(
            args.dequant_weight_linear_external_npz.as_posix()
            if args.dequant_weight_linear_external_npz is not None
            else None
        ),
        mixed_keep_ratio=float(args.dequant_weight_linear_mixed_keep_ratio),
        mixed_keep_count=int(args.dequant_weight_linear_mixed_keep_count),
        mixed_rank=args.dequant_weight_linear_mixed_rank,
        mixed_output_residual_npz=(
            args.dequant_weight_linear_mixed_output_residual_npz.as_posix()
            if args.dequant_weight_linear_mixed_output_residual_npz is not None
            else None
        ),
        mixed_output_residual_step_weights=args.dequant_weight_linear_mixed_output_residual_step_weights,
        mixed_group_size=int(args.dequant_weight_linear_mixed_group_size),
        mixed_group_keep_ratio=float(args.dequant_weight_linear_mixed_group_keep_ratio),
        mixed_group_keep_count=int(args.dequant_weight_linear_mixed_group_keep_count),
        mixed_group_rank=args.dequant_weight_linear_mixed_group_rank,
        mixed_group_calibration_npz=(
            args.dequant_weight_linear_mixed_group_calibration_npz.as_posix()
            if args.dequant_weight_linear_mixed_group_calibration_npz is not None
            else None
        ),
        mixed_group_mask_npz=(
            args.dequant_weight_linear_mixed_group_mask_npz.as_posix()
            if args.dequant_weight_linear_mixed_group_mask_npz is not None
            else None
        ),
        residual_correction_rank=int(args.dequant_weight_linear_residual_correction_rank),
        residual_correction_mode=args.dequant_weight_linear_residual_correction_mode,
        residual_correction_calibration_npz=(
            args.dequant_weight_linear_residual_correction_calibration_npz.as_posix()
            if args.dequant_weight_linear_residual_correction_calibration_npz is not None
            else None
        ),
    )
    int4_nonlinear_linear_meta = {"enabled": False}
    collect_int4_nonlinear_runtime_stats = None
    if args.int4_nonlinear_linear:
        from int4_nonlinear_gemm import Int4QuantConfig
        from int4_nonlinear_gemm_ctypes import (
            collect_int4_nonlinear_runtime_stats,
            replace_linear_modules_ctypes,
        )

        os.environ["INT4_GEMM_KERNEL"] = args.int4_nonlinear_linear_kernel
        os.environ["INT4_GEMM_THREADS"] = str(int(args.int4_nonlinear_linear_threads))
        int4_cfg = Int4QuantConfig(
            mode=args.int4_nonlinear_linear_mode,
            group_size=int(args.int4_nonlinear_linear_group_size),
            percentile=float(args.int4_nonlinear_linear_percentile),
            mu=float(args.int4_nonlinear_linear_mu),
            kmeans_iters=int(args.int4_nonlinear_linear_kmeans_iters),
            backend="ctypes",
        )
        pipe.flow_model, int4_nonlinear_linear_meta = replace_linear_modules_ctypes(
            pipe.flow_model,
            int4_cfg,
            include_regex=args.int4_nonlinear_linear_include_regex,
            exclude_regex=args.int4_nonlinear_linear_exclude_regex,
            min_in_features=int(args.int4_nonlinear_linear_min_in_features),
            max_linears=args.int4_nonlinear_linear_max_linears,
        )
    mixed_pc8_linear_step_context = {"step": 0}
    mixed_pc8_linear_meta = apply_triposplat_mixed_pc8_linear_patch(
        pipe.flow_model,
        enabled=bool(args.mixed_pc8_linear),
        include_regex=args.mixed_pc8_linear_include_regex,
        exclude_regex=args.mixed_pc8_linear_exclude_regex,
        group_mask_npz=(
            args.mixed_pc8_linear_group_mask_npz.as_posix()
            if args.mixed_pc8_linear_group_mask_npz is not None
            else None
        ),
        group_size=int(args.mixed_pc8_linear_group_size),
        library_path=args.mixed_pc8_linear_library,
        kernel_variant=args.mixed_pc8_linear_kernel_variant,
        min_rows=int(args.mixed_pc8_linear_min_rows),
        threads=int(args.mixed_pc8_linear_threads),
        step_context=mixed_pc8_linear_step_context,
        fallback_steps=args.mixed_pc8_linear_fallback_steps,
        residual_correction_rank=int(args.mixed_pc8_linear_residual_correction_rank),
        residual_correction_mode=args.mixed_pc8_linear_residual_correction_mode,
        residual_correction_calibration_npz=(
            args.mixed_pc8_linear_residual_correction_calibration_npz.as_posix()
            if args.mixed_pc8_linear_residual_correction_calibration_npz is not None
            else None
        ),
        residual_correction_factors_npz=(
            args.mixed_pc8_linear_residual_correction_factors_npz.as_posix()
            if args.mixed_pc8_linear_residual_correction_factors_npz is not None
            else None
        ),
        residual_correction_save_factors_npz=(
            args.mixed_pc8_linear_save_residual_correction_factors_npz.as_posix()
            if args.mixed_pc8_linear_save_residual_correction_factors_npz is not None
            else None
        ),
        residual_correction_gemm_library=args.mixed_pc8_linear_residual_correction_gemm_library,
        residual_correction_gemm_symbol=args.mixed_pc8_linear_residual_correction_gemm_symbol,
        residual_correction_fused_library=args.mixed_pc8_linear_residual_correction_fused_library,
        residual_correction_fused_symbol=args.mixed_pc8_linear_residual_correction_fused_symbol,
    )
    u7s8_linear_meta = apply_triposplat_u7s8_linear_patch(
        pipe.flow_model,
        enabled=bool(args.u7s8_linear),
        include_regex=args.u7s8_linear_include_regex,
        exclude_regex=args.u7s8_linear_exclude_regex,
        min_rows=int(args.u7s8_linear_min_rows),
        patch_mlp=bool(args.u7s8_linear_patch_mlp),
        library_path=args.u7s8_linear_library,
        threads=int(args.u7s8_linear_threads),
    )
    native_avx512_linear_meta = apply_triposplat_native_linear_avx512_patch(
        pipe.flow_model,
        enabled=bool(args.native_avx512_linear),
        include_regex=args.native_avx512_linear_include_regex,
        exclude_regex=args.native_avx512_linear_exclude_regex,
        library_path=args.native_avx512_linear_library,
        threads=int(args.native_avx512_linear_threads),
        strict=bool(args.native_avx512_linear_strict),
    )
    native_avx512_gelu_meta = apply_triposplat_native_gelu_avx512_patch(
        pipe.flow_model,
        enabled=bool(args.native_avx512_gelu),
        include_regex=args.native_avx512_gelu_include_regex,
        exclude_regex=args.native_avx512_gelu_exclude_regex,
        library_path=args.native_avx512_gelu_library,
        threads=int(args.native_avx512_gelu_threads),
        strict=bool(args.native_avx512_gelu_strict),
    )
    native_avx512_norm_rope_meta = apply_triposplat_native_norm_rope_avx512_patch(
        pipe.flow_model,
        enabled=bool(args.native_avx512_norm_rope),
        library_path=args.native_avx512_norm_rope_library,
        threads=int(args.native_avx512_norm_rope_threads),
        strict=bool(args.native_avx512_norm_rope_strict),
    )
    native_avx512_silu_meta = apply_triposplat_native_silu_avx512_patch(
        pipe.flow_model, enabled=bool(args.native_avx512_silu),
        library_path=args.native_avx512_silu_library,
        threads=int(args.native_avx512_silu_threads), strict=bool(args.native_avx512_silu_strict),
    )
    native_avx512_block_elementwise_meta = apply_triposplat_native_block_elementwise_avx512_patch(
        pipe.flow_model, enabled=bool(args.native_avx512_block_elementwise),
        library_path=args.native_avx512_block_elementwise_library,
        threads=int(args.native_avx512_block_elementwise_threads), strict=bool(args.native_avx512_block_elementwise_strict),
    )
    native_avx512_repo_meta = apply_triposplat_native_repo_avx512_patch(
        pipe.flow_model, enabled=bool(args.native_avx512_repo),
        library_path=args.native_avx512_repo_library,
        threads=int(args.native_avx512_repo_threads), strict=bool(args.native_avx512_repo_strict),
    )
    native_avx512_embeddings_meta = apply_triposplat_native_embeddings_avx512_patch(
        pipe.flow_model, enabled=bool(args.native_avx512_embeddings),
        library_path=args.native_avx512_embeddings_library,
        threads=int(args.native_avx512_embeddings_threads), strict=bool(args.native_avx512_embeddings_strict),
    )
    native_avx512_sampler_backend, native_avx512_sampler_meta = create_triposplat_native_sampler_avx512_backend(
        enabled=bool(args.native_avx512_sampler),
        library_path=args.native_avx512_sampler_library,
        threads=int(args.native_avx512_sampler_threads), strict=bool(args.native_avx512_sampler_strict),
    )
    selective_final_block_patch_meta = apply_triposplat_selective_final_block_patch(
        pipe.flow_model,
        enabled=bool(args.selective_final_block),
        backend=args.selective_final_block_backend or args.attention_backend,
        compute_dtype=args.selective_final_block_compute_dtype or args.attention_compute_dtype,
        query_chunk_size=int(args.selective_final_block_query_chunk_size),
        round_qkv_to_fp16=bool(args.selective_final_block_round_qkv_to_fp16),
        round_v_to_fp16=bool(args.selective_final_block_round_v_to_fp16),
        round_attn_core_to_fp16=bool(args.selective_final_block_round_attn_core_to_fp16),
        half_sequence=bool(args.selective_final_block_half_sequence),
        elementwise_compute_dtype=args.selective_final_block_elementwise_compute_dtype,
        inplace_output=bool(args.selective_final_block_inplace_output),
    )
    late_selective_condition_freeze_meta = apply_triposplat_late_selective_condition_freeze_patch(
        pipe.flow_model,
        enabled=int(args.late_selective_condition_freeze_blocks) > 0,
        block_count=int(args.late_selective_condition_freeze_blocks),
        backend=args.late_selective_condition_freeze_backend,
        compute_dtype=args.late_selective_condition_freeze_compute_dtype,
        query_chunk_size=int(args.late_selective_condition_freeze_query_chunk_size),
    )
    attention_patch_meta = {
        "global": global_attention_patch_meta,
        "module": module_attention_patch_meta,
        "selective_final_block": selective_final_block_patch_meta,
        "late_selective_condition_freeze": late_selective_condition_freeze_meta,
    }
    unified_block_half_sequence_exclude_regex = args.unified_block_half_sequence_exclude_regex
    if args.unified_block_half_sequence and args.selective_final_block and hasattr(pipe.flow_model, "blocks") and len(pipe.flow_model.blocks):
        final_block_regex = rf"^blocks[.]{len(pipe.flow_model.blocks) - 1}$"
        if unified_block_half_sequence_exclude_regex:
            unified_block_half_sequence_exclude_regex = f"(?:{unified_block_half_sequence_exclude_regex})|(?:{final_block_regex})"
        else:
            unified_block_half_sequence_exclude_regex = final_block_regex
    static_condition_cache_patch_meta = apply_triposplat_static_condition_cache_patch(
        pipe.flow_model,
        enabled=bool(args.static_condition_cache),
    )
    static_condition_cache_meta = {"enabled": bool(args.static_condition_cache), "patch": static_condition_cache_patch_meta}
    position_embed_cache_meta = apply_triposplat_position_embed_cache_patch(
        pipe.flow_model,
        enabled=bool(args.position_embed_cache),
    )
    repo_polar_cos_sin_meta = apply_triposplat_repo_polar_cos_sin_patch(
        pipe.flow_model,
        enabled=bool(args.repo_polar_cos_sin),
    )
    cfg_duplicate_state_patch_meta = apply_triposplat_cfg_duplicate_state_patch(
        pipe.flow_model,
        enabled=bool(args.cfg_deduplicate_state_forward),
        assume_duplicated=bool(args.cfg_deduplicate_state_assume_duplicated),
    )
    negative_condition_compression_meta = apply_triposplat_negative_condition_compression_patch(
        pipe.flow_model,
        enabled=bool(args.negative_condition_compression),
        verify_negative_rows_identical=not bool(args.negative_condition_assume_identical_rows),
        combine_linear_blocks=bool(args.negative_condition_combine_linear),
        full_linear_compressed_sdpa_blocks=bool(args.negative_condition_full_linear_compressed_sdpa),
        selective_final_block=bool(args.negative_condition_selective_final_block),
        selective_final_negative_branch=not bool(args.negative_condition_selective_final_positive_only),
        addcmul_elementwise=bool(args.addcmul_elementwise),
        inplace_elementwise=bool(args.negative_condition_inplace_elementwise),
        noise_refiner_inplace_elementwise=bool(args.negative_condition_noise_refiner_inplace_elementwise),
        positive_compiled_realrope=bool(args.negative_condition_positive_compiled_realrope),
        positive_fullblock_compiled_realrope=bool(args.negative_condition_positive_fullblock_compiled_realrope),
        parallel_branches=bool(args.negative_condition_parallel_branches),
        parallel_branch_workers=int(args.negative_condition_parallel_branch_workers),
        attention_backend=args.attention_backend,
        logbias_lse_adjust=bool(args.negative_condition_logbias_lse_adjust),
        internal_timing=bool(args.negative_condition_internal_timing),
    )
    addcmul_elementwise_exclude_regex = args.addcmul_elementwise_exclude_regex
    if args.addcmul_elementwise and args.selective_final_block and hasattr(pipe.flow_model, "blocks") and len(pipe.flow_model.blocks):
        final_block_regex = rf"^blocks[.]{len(pipe.flow_model.blocks) - 1}$"
        if addcmul_elementwise_exclude_regex:
            addcmul_elementwise_exclude_regex = f"(?:{addcmul_elementwise_exclude_regex})|(?:{final_block_regex})"
        else:
            addcmul_elementwise_exclude_regex = final_block_regex
    addcmul_elementwise_meta = apply_triposplat_addcmul_elementwise_patch(
        pipe.flow_model,
        enabled=bool(args.addcmul_elementwise),
        include_regex=args.addcmul_elementwise_include_regex,
        exclude_regex=addcmul_elementwise_exclude_regex,
    )
    unified_block_half_sequence_meta = apply_triposplat_unified_block_half_sequence_patch(
        pipe.flow_model,
        enabled=bool(args.unified_block_half_sequence),
        include_regex=args.unified_block_half_sequence_include_regex,
        exclude_regex=unified_block_half_sequence_exclude_regex,
        backend=args.unified_block_half_sequence_backend,
        compute_dtype=args.unified_block_half_sequence_compute_dtype,
        query_chunk_size=int(args.unified_block_half_sequence_query_chunk_size),
        elementwise_compute_dtype=args.unified_block_half_sequence_elementwise_compute_dtype,
    )
    module_output_rounding_meta = {"enabled": False}
    if args.round_module_outputs_to_fp16:
        module_output_rounding_meta = _install_module_output_rounding_hooks(
            pipe.flow_model,
            torch,
            storage_dtype=torch.float16,
            include_regex=args.round_module_output_include_regex,
            exclude_regex=args.round_module_output_exclude_regex,
        )
    module_input_rounding_meta = {"enabled": False}
    if args.round_module_inputs_to_fp16:
        module_input_rounding_meta = _install_module_input_rounding_hooks(
            pipe.flow_model,
            torch,
            storage_dtype=torch.float16,
            include_regex=args.round_module_input_include_regex,
            exclude_regex=args.round_module_input_exclude_regex,
        )
    native_half_module_meta = {"enabled": False}
    if args.native_half_module_include_regex:
        native_half_module_meta = _install_native_half_module_hooks(
            pipe.flow_model,
            torch,
            include_regex=args.native_half_module_include_regex,
            exclude_regex=args.native_half_module_exclude_regex,
            output_dtype=args.native_half_module_output_dtype,
            compute_dtype=args.native_half_module_compute_dtype,
        )
    linear_input_covariance_output_meta = {"enabled": False}
    linear_input_covariance_step_context = {"step": 0}
    linear_input_covariance_meta, linear_input_covariance_finalize = _install_linear_input_covariance_capture(
        pipe.flow_model,
        torch,
        output_npz=args.capture_linear_input_covariance_npz,
        include_regex=args.capture_linear_input_covariance_include_regex,
        exclude_regex=args.capture_linear_input_covariance_exclude_regex,
        group_size=int(args.capture_linear_input_covariance_group_size),
        split_by_step=bool(args.capture_linear_input_covariance_by_step),
        step_context=linear_input_covariance_step_context,
    )
    linear_input_covariance_output_meta = _manifest_hook_meta(linear_input_covariance_meta)
    linear_output_residual_output_meta = {"enabled": False}
    linear_output_residual_step_context = {"step": 0}
    linear_output_residual_meta, linear_output_residual_finalize = _install_linear_output_residual_capture(
        pipe.flow_model,
        torch,
        output_npz=args.capture_linear_output_residual_npz,
        include_regex=args.capture_linear_output_residual_include_regex,
        exclude_regex=args.capture_linear_output_residual_exclude_regex,
        bits=int(args.capture_linear_output_residual_bits),
        mode=args.capture_linear_output_residual_mode,
        split_by_step=bool(args.capture_linear_output_residual_by_step),
        step_context=linear_output_residual_step_context,
    )
    linear_output_residual_output_meta = _manifest_hook_meta(linear_output_residual_meta)
    timestep_embedder_functional_half_meta = _install_timestep_embedder_functional_half_patch(
        pipe.flow_model,
        torch,
        enabled=bool(args.timestep_embedder_functional_half),
        output_dtype=args.timestep_embedder_functional_half_output_dtype,
        compute_dtype=args.timestep_embedder_functional_half_compute_dtype,
    )
    model_runtime_dtype = torch.float32 if args.dynamic_int8_linears else dtype
    runtime_state_rounding_meta = {
        "enabled": bool(args.round_runtime_state_to_fp16),
        "storage_dtype": "float16" if args.round_runtime_state_to_fp16 else None,
        "runtime_dtype": str(model_runtime_dtype).replace("torch.", ""),
    }
    condition_token_reduction_meta = {"enabled": False, "mode": args.condition_token_mode}
    static_condition_cache_meta = {"enabled": False}
    flow_compile_meta = {"enabled": False}
    positive_fullblock_compile_warmup_meta = {"enabled": bool(args.negative_condition_positive_fullblock_compiled_realrope), "warmup": bool(args.torch_compile_warmup), "ran": False}
    positive_final_selected_compile_warmup_meta = {"enabled": os.environ.get("TRIPOSPLAT_POS_FINAL_SELECTED_COMPILED_REALROPE", "").strip().lower() in {"1", "true", "yes", "on"}, "warmup": bool(args.torch_compile_warmup), "ran": False}
    negative_fullblock_compile_warmup_meta = {"enabled": os.environ.get("TRIPOSPLAT_NEG_FULLBLOCK_COMPILED_REALROPE", "").strip().lower() in {"1", "true", "yes", "on"}, "warmup": bool(args.torch_compile_warmup), "ran": False}

    prepared = Image.open(args.input).convert("RGB")
    gen = torch.Generator(device=device).manual_seed(int(args.seed))
    with torch.inference_mode():
        if args.condition_npz:
            cond, loaded_condition_meta = load_condition_npz(args.condition_npz, device=device, model_dtype=model_runtime_dtype)
            condition_meta = {
                "source": "loaded_npz",
                "condition_npz": args.condition_npz.as_posix(),
                "loaded_metadata": loaded_condition_meta,
            }
        else:
            cond = encode_image_controlled(pipe, prepared, generator=gen, vae_deterministic=bool(args.vae_deterministic))
            condition_meta = {"source": "encoded", "vae_deterministic": bool(args.vae_deterministic)}
        noise, noise_meta, noise_np = build_noise(pipe.flow_model, args, device, gen)
        cond = _state_to_dtype(cond, model_runtime_dtype)
        noise = _state_to_dtype(noise, model_runtime_dtype)
        if args.round_runtime_state_to_fp16:
            cond = _state_roundtrip(cond, torch.float16, model_runtime_dtype)
            noise = _state_roundtrip(noise, torch.float16, model_runtime_dtype)
        cond, condition_token_reduction_meta = _reduce_condition_tokens(
            cond,
            torch,
            mode=args.condition_token_mode,
            patch_grid_size=args.condition_patch_grid_size,
            keep_prefix=int(args.condition_prefix_tokens),
        )
        neg_cond = {key: torch.zeros_like(value) for key, value in cond.items()}
        cond_sha256 = tensor_dict_sha256(cond)
        noise_sha256 = tensor_dict_sha256(noise)
        if args.static_condition_cache:
            cond, cond_cache_meta = make_triposplat_cached_condition(pipe.flow_model, cond)
            neg_cond, neg_cache_meta = make_triposplat_cached_condition(pipe.flow_model, neg_cond)
            static_condition_cache_meta = {
                "enabled": True,
                "patch": static_condition_cache_patch_meta,
                "condition": cond_cache_meta,
                "negative_condition": neg_cache_meta,
            }
        if args.negative_condition_positive_fullblock_compiled_realrope and args.torch_compile_warmup:
            cached_cond = cond.get("_triposplat_cached_h_cond") if isinstance(cond, dict) else None
            cond_tokens = int(cached_cond.shape[1]) if cached_cond is not None else int(cond["feature1"].shape[1])
            latent_tokens = int(noise["latent"].shape[1])
            cam_tokens = int(noise["camera"].shape[1]) if "camera" in noise else 0
            positive_fullblock_compile_warmup_meta = warmup_triposplat_positive_fullblock_compiled_realrope(
                pipe.flow_model,
                torch,
                sequence_length=latent_tokens + cond_tokens + cam_tokens,
                dtype=model_runtime_dtype,
                device=device,
            )
            positive_fullblock_compile_warmup_meta["warmup"] = True
            positive_fullblock_compile_warmup_meta["ran"] = True
            positive_fullblock_compile_warmup_meta["latent_tokens"] = latent_tokens
            positive_fullblock_compile_warmup_meta["condition_tokens"] = cond_tokens
            positive_fullblock_compile_warmup_meta["camera_tokens"] = cam_tokens
        if os.environ.get("TRIPOSPLAT_POS_FINAL_SELECTED_COMPILED_REALROPE", "").strip().lower() in {"1", "true", "yes", "on"} and args.torch_compile_warmup:
            cached_cond = cond.get("_triposplat_cached_h_cond") if isinstance(cond, dict) else None
            cond_tokens = int(cached_cond.shape[1]) if cached_cond is not None else int(cond["feature1"].shape[1])
            latent_tokens = int(noise["latent"].shape[1])
            cam_tokens = int(noise["camera"].shape[1]) if "camera" in noise else 0
            positive_final_selected_compile_warmup_meta = warmup_triposplat_positive_final_selected_compiled_realrope(
                pipe.flow_model,
                torch,
                sequence_length=latent_tokens + cond_tokens + cam_tokens,
                latent_tokens=latent_tokens,
                camera_tokens=cam_tokens,
                dtype=model_runtime_dtype,
                device=device,
            )
            positive_final_selected_compile_warmup_meta["warmup"] = True
            positive_final_selected_compile_warmup_meta["ran"] = True
        if os.environ.get("TRIPOSPLAT_NEG_FULLBLOCK_COMPILED_REALROPE", "").strip().lower() in {"1", "true", "yes", "on"} and args.torch_compile_warmup:
            cached_cond = cond.get("_triposplat_cached_h_cond") if isinstance(cond, dict) else None
            cond_tokens = int(cached_cond.shape[1]) if cached_cond is not None else int(cond["feature1"].shape[1])
            latent_tokens = int(noise["latent"].shape[1])
            cam_tokens = int(noise["camera"].shape[1]) if "camera" in noise else 0
            negative_fullblock_compile_warmup_meta = warmup_triposplat_negative_fullblock_compiled_realrope(
                pipe.flow_model,
                torch,
                sequence_length=latent_tokens + 1 + cam_tokens,
                condition_tokens=cond_tokens,
                negative_condition_index=latent_tokens,
                dtype=model_runtime_dtype,
                device=device,
            )
            negative_fullblock_compile_warmup_meta["warmup"] = True
            negative_fullblock_compile_warmup_meta["ran"] = True
        pipe.flow_model, flow_compile_meta = _compile_flow_model_if_requested(
            pipe.flow_model,
            torch,
            enabled=bool(args.torch_compile_flow),
            backend=args.torch_compile_backend,
            mode=args.torch_compile_mode,
            fullgraph=bool(args.torch_compile_fullgraph),
            dynamic=bool(args.torch_compile_dynamic),
            warmup=bool(args.torch_compile_warmup),
            cfg_split_forward=bool(args.cfg_split_forward),
            cfg_single_state_forward=bool(args.cfg_single_state_forward),
            variants=variants,
            noise=noise,
            cond=cond,
            neg_cond=neg_cond,
        )
        if args.save_noise_npz and noise_np is not None:
            save_noise_npz(args.save_noise_npz, noise_np, noise_meta)
        sampler = FlowEulerCfgMultiVariantSampler(
            fake_quant=fake_quant,
            quantize_points=quantize_points,
            cfg_split_forward=bool(args.cfg_split_forward),
            cfg_single_state_forward=bool(args.cfg_single_state_forward),
            solver=args.sampler_solver,
            clone_state_each_step=bool(args.sampler_clone_state),
            native_sampler_backend=native_avx512_sampler_backend,
        )
        module_trace_recorder = None
        module_trace_meta = None
        if args.module_trace_json:
            module_trace_recorder = ModuleTraceRecorder(
                include_regex=args.module_trace_include_regex,
                exclude_regex=args.module_trace_exclude_regex,
                max_modules=int(args.module_trace_max_modules),
                max_calls_per_module=int(args.module_trace_max_calls),
                max_values=int(args.module_trace_max_values),
                stats_sample_values=int(args.module_trace_stats_sample_values),
                store_values=bool(args.module_trace_store_values),
                trace_inputs=bool(args.module_trace_inputs),
            )
            module_trace_meta = module_trace_recorder.install(pipe.flow_model)
        trace_recorder = None
        if args.trace_json:
            trace_events = {part.strip() for part in args.trace_events.split(",") if part.strip()}
            trace_recorder = FlowTraceRecorder(
                max_steps=int(args.trace_max_steps),
                max_values=int(args.trace_max_values),
                stats_sample_values=int(args.trace_stats_sample_values),
                events=trace_events or None,
                store_values=bool(args.trace_store_values),
            )
        step_state_recorder = None
        if args.step_state_dir:
            step_state_events = {part.strip() for part in args.step_state_events.split(",") if part.strip()}
            valid_step_state_events = {"pre_forward", "post_cfg_prediction", "post_update"}
            bad_step_state_events = sorted(step_state_events - valid_step_state_events)
            if bad_step_state_events:
                raise ValueError(f"unsupported step-state event(s): {bad_step_state_events}; valid={sorted(valid_step_state_events)}")
            step_state_recorder = StepStateRecorder(
                args.step_state_dir,
                variants=variants,
                events=step_state_events or {"post_update"},
                max_steps=int(args.step_state_max_steps),
                storage_dtype=args.step_state_storage_dtype,
            )
        callbacks = []
        if args.mixed_pc8_linear and args.mixed_pc8_linear_fallback_steps.strip():
            def mixed_pc8_linear_step_callback(event, payload):
                if event == "pre_forward":
                    mixed_pc8_linear_step_context["step"] = int(payload.get("step", 0))
            callbacks.append(mixed_pc8_linear_step_callback)
        if args.capture_linear_input_covariance_by_step:
            def linear_input_covariance_step_callback(event, payload):
                if event == "pre_forward":
                    linear_input_covariance_step_context["step"] = int(payload.get("step", 0))
            callbacks.append(linear_input_covariance_step_callback)
        if args.capture_linear_output_residual_by_step:
            def linear_output_residual_step_callback(event, payload):
                if event == "pre_forward":
                    linear_output_residual_step_context["step"] = int(payload.get("step", 0))
            callbacks.append(linear_output_residual_step_callback)
        if trace_recorder is not None:
            callbacks.append(trace_recorder.callback)
        if step_state_recorder is not None:
            callbacks.append(step_state_recorder.callback)

        def combined_trace_callback(event, payload):
            for callback in callbacks:
                callback(event, payload)

        operator_profiler_context = (
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=True,
                profile_memory=False,
                with_stack=True,
            )
            if args.operator_audit_json
            else contextlib.nullcontext()
        )
        operator_profiler = None
        try:
            with operator_profiler_context as operator_profiler:
                latents, sampler_meta = sampler.sample(
                    pipe.flow_model,
                    noise,
                    cond,
                    neg_cond,
                    variants,
                    show_progress=bool(args.progress),
                    trace_callback=None if not callbacks else combined_trace_callback,
                )
        finally:
            if module_trace_recorder is not None:
                module_trace_recorder.remove()
            if linear_input_covariance_finalize is not None:
                linear_input_covariance_output_meta = linear_input_covariance_finalize()
            if linear_output_residual_finalize is not None:
                linear_output_residual_output_meta = linear_output_residual_finalize()

    if int4_nonlinear_linear_meta.get("enabled") and collect_int4_nonlinear_runtime_stats is not None:
        int4_nonlinear_linear_meta = dict(int4_nonlinear_linear_meta)
        int4_nonlinear_linear_meta["runtime_stats"] = collect_int4_nonlinear_runtime_stats(pipe.flow_model)

    trace_meta = None
    module_trace_output_meta = None
    step_state_output_meta = None
    operator_audit_meta = {"enabled": False}
    if args.operator_audit_json and operator_profiler is not None:
        operator_rows = []
        for event in operator_profiler.key_averages(group_by_input_shape=True, group_by_stack_n=8):
            operator_rows.append({
                "operator": str(event.key),
                "calls": int(event.count),
                "self_cpu_time_us": float(event.self_cpu_time_total),
                "total_cpu_time_us": float(event.cpu_time_total),
                "input_shapes": event.input_shapes,
                "stack": list(event.stack[:8]),
            })
        operator_rows.sort(key=lambda row: row["self_cpu_time_us"], reverse=True)
        arithmetic_fragments = (
            "add", "sub", "mul", "div", "mm", "linear", "softmax", "exp", "sin", "cos",
            "norm", "gelu", "silu", "sqrt", "pow", "clamp", "sum", "mean", "var",
        )
        arithmetic_candidates = [
            row for row in operator_rows
            if row["operator"].startswith("aten::")
            and any(fragment in row["operator"].lower() for fragment in arithmetic_fragments)
        ]
        operator_audit_payload = {
            "created_at": utc_now(),
            "scope": "FlowEulerCfgMultiVariantSampler.sample only",
            "record_shapes": True,
            "with_stack": True,
            "profile_memory": False,
            "operators": operator_rows,
            "arithmetic_candidates": arithmetic_candidates,
        }
        args.operator_audit_json.parent.mkdir(parents=True, exist_ok=True)
        args.operator_audit_json.write_text(json.dumps(operator_audit_payload, indent=2) + "\n")
        operator_audit_meta = {
            "enabled": True,
            "path": args.operator_audit_json.as_posix(),
            "operator_count": len(operator_rows),
            "arithmetic_candidate_count": len(arithmetic_candidates),
            "scope": operator_audit_payload["scope"],
        }
    if args.module_trace_json and module_trace_recorder is not None:
        module_trace_output_meta = {
            "module_trace_json": args.module_trace_json.as_posix(),
            "record_count": len(module_trace_recorder.records),
            "install": module_trace_meta,
        }
        module_trace_recorder.write_json(
            args.module_trace_json,
            metadata={
                "input": args.input.as_posix(),
                "output_dir": args.output_dir.as_posix(),
                "variants": [variant.__dict__ for variant in variants],
                "canvas_size": int(args.canvas_size),
                "seed": int(args.seed),
                "device": str(device),
                "model_dtype": str(dtype).replace("torch.", ""),
                "runtime_dtype": str(model_runtime_dtype).replace("torch.", ""),
                "attention_patch": attention_patch_meta,
                "fake_quant": fake_quant.__dict__,
                "fake_quant_points": list(quantize_points),
                "condition_token_reduction": condition_token_reduction_meta,
                "static_condition_cache": static_condition_cache_meta,
                "position_embed_cache": position_embed_cache_meta,
                "repo_polar_cos_sin": repo_polar_cos_sin_meta,
                "cfg_duplicate_state_forward": cfg_duplicate_state_patch_meta,
                "negative_condition_compression": negative_condition_compression_meta,
                "module_input_rounding": _manifest_hook_meta(module_input_rounding_meta),
                "native_half_module": _manifest_hook_meta(native_half_module_meta),
                "linear_input_covariance": linear_input_covariance_output_meta,
                "linear_output_residual": linear_output_residual_output_meta,
                "timestep_embedder_functional_half": timestep_embedder_functional_half_meta,
                "unified_block_half_sequence": unified_block_half_sequence_meta,
                "gelu_out_buffer_mlp": gelu_out_buffer_mlp_meta,
                "chunked_mlp": chunked_mlp_meta,
                "linear_output_buffer": linear_output_buffer_meta,
                "native_avx512_linear": native_avx512_linear_meta,
                "native_avx512_gelu": native_avx512_gelu_meta,
                "native_avx512_norm_rope": native_avx512_norm_rope_meta,
                "native_avx512_silu": native_avx512_silu_meta,
                "native_avx512_block_elementwise": native_avx512_block_elementwise_meta,
                "native_avx512_repo": native_avx512_repo_meta,
                "native_avx512_embeddings": native_avx512_embeddings_meta,
                "native_avx512_sampler": native_avx512_sampler_meta,
                "dequant_weight_linear": dequant_weight_linear_meta,
                "int4_nonlinear_linear": int4_nonlinear_linear_meta,
                "mixed_pc8_linear": mixed_pc8_linear_meta,
                "addcmul_elementwise": addcmul_elementwise_meta,
                "torch_compile": flow_compile_meta,
            },
        )

    if args.trace_json and trace_recorder is not None:
        trace_meta = {
            "trace_json": args.trace_json.as_posix(),
            "record_count": len(trace_recorder.records),
            "max_steps": int(args.trace_max_steps),
            "max_values": int(args.trace_max_values),
            "stats_sample_values": int(args.trace_stats_sample_values),
            "events": sorted({part.strip() for part in args.trace_events.split(",") if part.strip()}),
            "store_values": bool(args.trace_store_values),
        }
        trace_recorder.write_json(
            args.trace_json,
            metadata={
                "input": args.input.as_posix(),
                "output_dir": args.output_dir.as_posix(),
                "variants": [variant.__dict__ for variant in variants],
                "canvas_size": int(args.canvas_size),
                "seed": int(args.seed),
                "device": str(device),
                "model_dtype": str(dtype).replace("torch.", ""),
                "runtime_dtype": str(model_runtime_dtype).replace("torch.", ""),
                "attention_patch": attention_patch_meta,
                "fake_quant": fake_quant.__dict__,
                "fake_quant_points": list(quantize_points),
                "condition_token_reduction": condition_token_reduction_meta,
                "static_condition_cache": static_condition_cache_meta,
                "cfg_duplicate_state_forward": cfg_duplicate_state_patch_meta,
                "negative_condition_compression": negative_condition_compression_meta,
                "module_input_rounding": _manifest_hook_meta(module_input_rounding_meta),
                "native_half_module": _manifest_hook_meta(native_half_module_meta),
                "linear_input_covariance": linear_input_covariance_output_meta,
                "linear_output_residual": linear_output_residual_output_meta,
                "timestep_embedder_functional_half": timestep_embedder_functional_half_meta,
                "unified_block_half_sequence": unified_block_half_sequence_meta,
                "addcmul_elementwise": addcmul_elementwise_meta,
                "torch_compile": flow_compile_meta,
            },
        )

    if args.step_state_dir and step_state_recorder is not None:
        step_state_manifest = args.step_state_dir / "manifest.json"
        step_state_output_meta = {
            "step_state_dir": args.step_state_dir.as_posix(),
            "manifest": step_state_manifest.as_posix(),
            "record_count": len(step_state_recorder.records),
            "events": sorted({part.strip() for part in args.step_state_events.split(",") if part.strip()}),
            "max_steps": int(args.step_state_max_steps),
            "storage_dtype": args.step_state_storage_dtype,
        }
        step_state_recorder.write_manifest(
            step_state_manifest,
            metadata={
                "input": args.input.as_posix(),
                "output_dir": args.output_dir.as_posix(),
                "variants": [variant.__dict__ for variant in variants],
                "canvas_size": int(args.canvas_size),
                "seed": int(args.seed),
                "device": str(device),
                "model_dtype": str(dtype).replace("torch.", ""),
                "runtime_dtype": str(model_runtime_dtype).replace("torch.", ""),
                "attention_patch": attention_patch_meta,
                "fake_quant": fake_quant.__dict__,
                "fake_quant_points": list(quantize_points),
                "condition_token_reduction": condition_token_reduction_meta,
                "static_condition_cache": static_condition_cache_meta,
                "cfg_duplicate_state_forward": cfg_duplicate_state_patch_meta,
                "negative_condition_compression": negative_condition_compression_meta,
                "module_input_rounding": _manifest_hook_meta(module_input_rounding_meta),
                "native_half_module": _manifest_hook_meta(native_half_module_meta),
                "linear_input_covariance": linear_input_covariance_output_meta,
                "linear_output_residual": linear_output_residual_output_meta,
                "timestep_embedder_functional_half": timestep_embedder_functional_half_meta,
                "unified_block_half_sequence": unified_block_half_sequence_meta,
                "addcmul_elementwise": addcmul_elementwise_meta,
                "torch_compile": flow_compile_meta,
            },
        )

    latent_files = {}
    latent_sha256 = {}
    for variant in variants:
        latent = latents[variant.name]
        latent_meta = {
            "created_at": utc_now(),
            "source": "quantized_param_batch",
            "variant": variant.__dict__,
            "fake_quant": fake_quant.__dict__,
            "dynamic_quantization": dynamic_quant_meta,
            "flow_param_rounding": flow_param_rounding_meta,
            "runtime_state_rounding": runtime_state_rounding_meta,
            "flush_denormal": flush_denormal_meta,
            "float32_matmul_precision": float32_matmul_precision_meta,
            "condition_token_reduction": condition_token_reduction_meta,
            "static_condition_cache": static_condition_cache_meta,
            "position_embed_cache": position_embed_cache_meta,
            "repo_polar_cos_sin": repo_polar_cos_sin_meta,
            "cfg_duplicate_state_forward": cfg_duplicate_state_patch_meta,
            "negative_condition_compression": negative_condition_compression_meta,
            "module_output_rounding": _manifest_hook_meta(module_output_rounding_meta),
            "module_input_rounding": _manifest_hook_meta(module_input_rounding_meta),
            "native_half_module": _manifest_hook_meta(native_half_module_meta),
            "linear_input_covariance": linear_input_covariance_output_meta,
            "linear_output_residual": linear_output_residual_output_meta,
            "timestep_embedder_functional_half": timestep_embedder_functional_half_meta,
            "unified_block_half_sequence": unified_block_half_sequence_meta,
            "gelu_out_buffer_mlp": gelu_out_buffer_mlp_meta,
            "chunked_mlp": chunked_mlp_meta,
            "linear_output_buffer": linear_output_buffer_meta,
            "native_avx512_linear": native_avx512_linear_meta,
            "native_avx512_gelu": native_avx512_gelu_meta,
            "native_avx512_norm_rope": native_avx512_norm_rope_meta,
            "native_avx512_silu": native_avx512_silu_meta,
            "native_avx512_block_elementwise": native_avx512_block_elementwise_meta,
            "native_avx512_repo": native_avx512_repo_meta,
            "native_avx512_embeddings": native_avx512_embeddings_meta,
            "native_avx512_sampler": native_avx512_sampler_meta,
            "dequant_weight_linear": dequant_weight_linear_meta,
            "int4_nonlinear_linear": int4_nonlinear_linear_meta,
            "mixed_pc8_linear": mixed_pc8_linear_meta,
            "addcmul_elementwise": addcmul_elementwise_meta,
            "torch_compile": flow_compile_meta,
            "attention_patch": attention_patch_meta,
            "module_trace": module_trace_output_meta,
            "step_state": step_state_output_meta,
            "operator_audit": operator_audit_meta,
            "canvas_size": int(args.canvas_size),
            "seed": int(args.seed),
            "device": str(device),
            "dtype": str(model_runtime_dtype).replace("torch.", ""),
        }
        path = args.output_dir / f"{variant.name}_latent.npz"
        save_tensor_dict_npz(path, latent, latent_meta)
        latent_files[variant.name] = path.as_posix()
        latent_sha256[variant.name] = tensor_dict_sha256(latent)

    elapsed = time.time() - t0
    manifest = {
        "created_at": utc_now(),
        "algorithm": "TripoSplat batched flow parameter sampler with quantization options",
        "input": args.input.as_posix(),
        "output_dir": args.output_dir.as_posix(),
        "variants": [variant.__dict__ for variant in variants],
        "canvas_size": int(args.canvas_size),
        "seed": int(args.seed),
        "device": str(device),
        "model_dtype": str(dtype).replace("torch.", ""),
        "runtime_dtype": str(model_runtime_dtype).replace("torch.", ""),
        "dynamic_quantization": dynamic_quant_meta,
        "mkldnn_fused_mlp": mkldnn_fused_mlp_meta,
        "flow_param_rounding": flow_param_rounding_meta,
        "runtime_state_rounding": runtime_state_rounding_meta,
        "flush_denormal": flush_denormal_meta,
        "float32_matmul_precision": float32_matmul_precision_meta,
        "condition_token_reduction": condition_token_reduction_meta,
        "static_condition_cache": static_condition_cache_meta,
        "position_embed_cache": position_embed_cache_meta,
        "repo_polar_cos_sin": repo_polar_cos_sin_meta,
        "cfg_duplicate_state_forward": cfg_duplicate_state_patch_meta,
        "negative_condition_compression": negative_condition_compression_meta,
        "module_output_rounding": _manifest_hook_meta(module_output_rounding_meta),
        "module_input_rounding": _manifest_hook_meta(module_input_rounding_meta),
        "native_half_module": _manifest_hook_meta(native_half_module_meta),
        "linear_input_covariance": linear_input_covariance_output_meta,
        "linear_output_residual": linear_output_residual_output_meta,
        "timestep_embedder_functional_half": timestep_embedder_functional_half_meta,
        "unified_block_half_sequence": unified_block_half_sequence_meta,
        "gelu_out_buffer_mlp": gelu_out_buffer_mlp_meta,
        "chunked_mlp": chunked_mlp_meta,
        "linear_output_buffer": linear_output_buffer_meta,
        "native_avx512_linear": native_avx512_linear_meta,
        "native_avx512_gelu": native_avx512_gelu_meta,
        "native_avx512_norm_rope": native_avx512_norm_rope_meta,
        "native_avx512_silu": native_avx512_silu_meta,
        "native_avx512_block_elementwise": native_avx512_block_elementwise_meta,
        "native_avx512_repo": native_avx512_repo_meta,
        "native_avx512_embeddings": native_avx512_embeddings_meta,
        "native_avx512_sampler": native_avx512_sampler_meta,
        "numpy_linear": numpy_linear_meta,
        "dequant_weight_linear": dequant_weight_linear_meta,
        "int4_nonlinear_linear": int4_nonlinear_linear_meta,
        "mixed_pc8_linear": mixed_pc8_linear_meta,
        "u7s8_linear": u7s8_linear_meta,
        "torch_compile": flow_compile_meta,
        "positive_fullblock_compile_warmup": positive_fullblock_compile_warmup_meta,
        "positive_final_selected_compile_warmup": positive_final_selected_compile_warmup_meta,
        "negative_fullblock_compile_warmup": negative_fullblock_compile_warmup_meta,
        "attention_patch": attention_patch_meta,
        "fake_quant": fake_quant.__dict__,
        "fake_quant_points": list(quantize_points),
        "cfg_split_forward": bool(args.cfg_split_forward),
        "progress": bool(args.progress),
        "noise": noise_meta,
        "condition": condition_meta,
        "condition_sha256": cond_sha256,
        "noise_sha256": noise_sha256,
        "latent_sha256": latent_sha256,
        "latent_files": latent_files,
        "sampler": sampler_meta,
        "trace": trace_meta,
        "module_trace": module_trace_output_meta,
        "step_state": step_state_output_meta,
        "operator_audit": operator_audit_meta,
        "elapsed_sec": elapsed,
        "torch_threads": {
            "num_threads": int(torch.get_num_threads()),
            "num_interop_threads": int(torch.get_num_interop_threads()),
            "requested_interop_threads": interop_threads_requested,
            "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads_env": os.environ.get("MKL_NUM_THREADS"),
        },
        "noise_npz_saved": args.save_noise_npz.as_posix() if args.save_noise_npz else None,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
