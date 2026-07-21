#!/usr/bin/env python3
"""Quantized and batched flow samplers for TripoSplat experiments.

This module is deliberately separated from the equivalence runner. Quantization
changes numeric behavior, so these helpers are for speed/quality exploration,
not for proving byte-equivalence to the official pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math
import re
import time
from typing import Iterable

import numpy as np
import torch
from tqdm.auto import tqdm


@dataclass(frozen=True)
class FlowVariant:
    name: str
    steps: int
    guidance_scale: float
    shift: float


@dataclass(frozen=True)
class FakeQuantConfig:
    mode: str = "none"
    bits: int = 8
    percentile: float = 0.999
    eps: float = 1e-8
    mu: float = 255.0


def parse_flow_variants(spec: str | None, default_steps: int, default_guidance: float, default_shift: float) -> list[FlowVariant]:
    """Parse a compact variant list.

    Format:
        name:steps:guidance:shift,name2:steps:guidance:shift

    Empty names are generated from the numeric fields.
    """
    if not spec:
        return [FlowVariant("base", int(default_steps), float(default_guidance), float(default_shift))]
    variants = []
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) == 3:
            name = ""
            steps_s, guidance_s, shift_s = parts
        elif len(parts) == 4:
            name, steps_s, guidance_s, shift_s = parts
        else:
            raise ValueError(f"invalid variant {item!r}; expected name:steps:guidance:shift")
        steps = int(steps_s)
        guidance = float(guidance_s)
        shift = float(shift_s)
        if not name:
            name = f"s{steps}_g{guidance:g}_shift{shift:g}".replace(".", "p")
        variants.append(FlowVariant(name, steps, guidance, shift))
    if not variants:
        raise ValueError("no valid variants were provided")
    return variants


def group_variants_by_steps(variants: Iterable[FlowVariant]) -> dict[int, list[FlowVariant]]:
    out: dict[int, list[FlowVariant]] = {}
    for variant in variants:
        out.setdefault(int(variant.steps), []).append(variant)
    return out


def _quantile_abs(x: torch.Tensor, q: float) -> torch.Tensor:
    flat = x.detach().float().abs().flatten()
    if flat.numel() == 0:
        return torch.tensor(1.0, device=x.device, dtype=torch.float32)
    if q >= 1:
        return flat.max()
    return torch.quantile(flat, float(q))


def _linear_symmetric_fake_quant(x: torch.Tensor, cfg: FakeQuantConfig) -> torch.Tensor:
    if cfg.bits <= 1:
        raise ValueError("linear symmetric quantization needs at least 2 bits")
    qmax = float((1 << (cfg.bits - 1)) - 1)
    amax = _quantile_abs(x, cfg.percentile).clamp_min(cfg.eps)
    scale = amax / qmax
    return (torch.clamp(torch.round(x.float() / scale), -qmax, qmax) * scale).to(dtype=x.dtype)


def _linear_affine_fake_quant(x: torch.Tensor, cfg: FakeQuantConfig) -> torch.Tensor:
    if cfg.bits <= 1:
        raise ValueError("linear affine quantization needs at least 2 bits")
    xf = x.float()
    lo = torch.quantile(xf.flatten(), 1.0 - float(cfg.percentile)) if cfg.percentile < 1 else xf.min()
    hi = torch.quantile(xf.flatten(), float(cfg.percentile)) if cfg.percentile < 1 else xf.max()
    if torch.isclose(hi, lo):
        return x
    qmin = 0.0
    qmax = float((1 << cfg.bits) - 1)
    scale = (hi - lo).clamp_min(cfg.eps) / (qmax - qmin)
    zero_point = torch.round(qmin - lo / scale).clamp(qmin, qmax)
    q = torch.clamp(torch.round(xf / scale + zero_point), qmin, qmax)
    return ((q - zero_point) * scale).to(dtype=x.dtype)


def _log_symmetric_fake_quant(x: torch.Tensor, cfg: FakeQuantConfig) -> torch.Tensor:
    """Non-linear sign/log-magnitude fake quantization.

    This keeps more relative precision near zero than linear quantization. It is
    useful for flow states whose magnitudes span a wide range.
    """
    if cfg.bits <= 2:
        raise ValueError("log symmetric quantization needs at least 3 bits")
    xf = x.float()
    sign = torch.sign(xf)
    mag = xf.abs()
    amax = _quantile_abs(xf, cfg.percentile).clamp_min(cfg.eps)
    nonzero = mag[mag > cfg.eps]
    amin = torch.quantile(nonzero, 1.0 - float(cfg.percentile)).clamp_min(cfg.eps) if nonzero.numel() else torch.tensor(cfg.eps, device=x.device)
    levels = float((1 << (cfg.bits - 1)) - 1)
    log_min = torch.log2(amin)
    log_max = torch.log2(amax)
    denom = (log_max - log_min).clamp_min(cfg.eps)
    q = torch.clamp(torch.round((torch.log2(mag.clamp_min(cfg.eps)) - log_min) / denom * levels), 0, levels)
    restored = torch.pow(2.0, q / levels * denom + log_min)
    restored = torch.where(mag <= cfg.eps, torch.zeros_like(restored), restored)
    return (sign * restored).to(dtype=x.dtype)


def _mulaw_fake_quant(x: torch.Tensor, cfg: FakeQuantConfig) -> torch.Tensor:
    """Non-linear mu-law companding fake quantization."""
    if cfg.bits <= 1:
        raise ValueError("mu-law quantization needs at least 2 bits")
    xf = x.float()
    amax = _quantile_abs(xf, cfg.percentile).clamp_min(cfg.eps)
    clipped = torch.clamp(xf / amax, -1.0, 1.0)
    mu = torch.tensor(float(cfg.mu), device=x.device, dtype=torch.float32)
    companded = torch.sign(clipped) * torch.log1p(mu * clipped.abs()) / torch.log1p(mu)
    qmax = float((1 << (cfg.bits - 1)) - 1)
    q = torch.clamp(torch.round(companded * qmax), -qmax, qmax) / qmax
    restored = torch.sign(q) * (torch.pow(1.0 + mu, q.abs()) - 1.0) / mu
    return (restored * amax).to(dtype=x.dtype)


def fake_quant_tensor(x: torch.Tensor, cfg: FakeQuantConfig) -> torch.Tensor:
    if cfg.mode == "none" or cfg.bits <= 0:
        return x
    if not torch.is_floating_point(x):
        return x
    if cfg.mode == "float16_roundtrip":
        return x.to(dtype=torch.float16).to(dtype=x.dtype)
    if cfg.mode == "linear_symmetric":
        return _linear_symmetric_fake_quant(x, cfg)
    if cfg.mode == "linear_affine":
        return _linear_affine_fake_quant(x, cfg)
    if cfg.mode == "log_symmetric":
        return _log_symmetric_fake_quant(x, cfg)
    if cfg.mode == "mulaw":
        return _mulaw_fake_quant(x, cfg)
    raise ValueError(f"unsupported fake quantization mode: {cfg.mode}")


def fake_quant_state(values, cfg: FakeQuantConfig):
    if cfg.mode == "none":
        return values
    if isinstance(values, dict):
        return {key: fake_quant_tensor(value, cfg) for key, value in values.items()}
    return fake_quant_tensor(values, cfg)


def tensor_error_stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    ref = reference.detach().float()
    cand = candidate.detach().float()
    diff = cand - ref
    denom = ref.abs().clamp_min(1e-8)
    return {
        "shape": list(ref.shape),
        "mse": float(torch.mean(diff * diff).cpu()),
        "mae": float(torch.mean(diff.abs()).cpu()),
        "max_abs": float(torch.max(diff.abs()).cpu()),
        "mean_relative_abs": float(torch.mean(diff.abs() / denom).cpu()),
    }


def state_error_stats(reference, candidate) -> dict:
    if isinstance(reference, dict):
        return {key: tensor_error_stats(reference[key], candidate[key]) for key in sorted(reference)}
    return {"value": tensor_error_stats(reference, candidate)}


def _state_cat(states: list[dict]) -> dict:
    return {key: torch.cat([state[key] for state in states], dim=0) for key in states[0]}


def _state_split(state: dict, count: int) -> list[dict]:
    chunks = {key: value.chunk(count, dim=0) for key, value in state.items()}
    return [{key: chunks[key][i].contiguous() for key in chunks} for i in range(count)]


def _state_repeat(value: dict, count: int) -> dict:
    return {key: tensor.repeat(count, *([1] * (tensor.dim() - 1))) for key, tensor in value.items()}


def _state_clone(value: dict) -> dict:
    return {key: tensor.clone() for key, tensor in value.items()}


def _state_update(sample: dict, pred: dict, dt: torch.Tensor) -> dict:
    out = {}
    for key, value in sample.items():
        shape = [dt.shape[0]] + [1] * (value.dim() - 1)
        out[key] = value - pred[key] * dt.reshape(shape).to(device=value.device, dtype=pred[key].dtype)
    return out


def _cat_cond_neg(cond: dict, neg_cond: dict) -> dict:
    return {key: torch.cat([cond[key], neg_cond[key]], dim=0) for key in cond}


def _cat_state2(state: dict) -> dict:
    return {key: torch.cat([value, value], dim=0) for key, value in state.items()}


def _split_pred2(pred: dict) -> tuple[dict, dict]:
    cond = {}
    neg = {}
    for key, value in pred.items():
        cond[key], neg[key] = value.chunk(2, dim=0)
    return cond, neg


def _model_forward(model, state: dict, t_scaled: torch.Tensor, cond: dict) -> dict:
    return model(state, t_scaled, cond)



class DynamicQuantizedLinearCompat(torch.nn.Module):
    """Compatibility wrapper for TripoSplat modules that inspect .weight.dtype.

    torch dynamic quantized Linear exposes weight as a method, while parts of
    TripoSplat use module.weight.dtype to cast inputs. The quantized op still
    expects float activations, so a tiny float32 proxy preserves that contract.
    """

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner
        self.register_buffer("_weight_proxy", torch.empty(0, dtype=torch.float32), persistent=False)

    @property
    def weight(self) -> torch.Tensor:
        return self._weight_proxy

    @property
    def bias(self):
        bias = getattr(self.inner, "bias", None)
        if callable(bias):
            return bias()
        return bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)

    def extra_repr(self) -> str:
        return f"inner={self.inner.__class__.__module__}.{self.inner.__class__.__name__}"


def _wrap_dynamic_quantized_linears(module: torch.nn.Module) -> int:
    replaced = 0
    for child_name, child in list(module.named_children()):
        if "quantized.dynamic" in child.__class__.__module__ and child.__class__.__name__ == "Linear":
            setattr(module, child_name, DynamicQuantizedLinearCompat(child))
            replaced += 1
        else:
            replaced += _wrap_dynamic_quantized_linears(child)
    return replaced

def dynamic_quantize_linear_modules(
    model: torch.nn.Module,
    enabled: bool,
    include_regex: str | None = None,
    exclude_regex: str | None = None,
    qconfig_name: str = "default",
) -> tuple[torch.nn.Module, dict]:
    """Apply PyTorch dynamic quantization to Linear modules on CPU.

    This targets the speed bottleneck in transformer MLP/projection layers. It
    does not quantize attention softmax, layer norms, activations, or custom
    non-linear code paths.
    """
    if not enabled:
        return model, {"enabled": False}
    import torch.ao.quantization as tq

    started = time.time()
    model_fp32 = copy.deepcopy(model).cpu().float().eval()
    include = re.compile(include_regex) if include_regex else None
    exclude = re.compile(exclude_regex) if exclude_regex else None

    selected_names = []
    skipped_names = []
    for name, module in model_fp32.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        selected = True
        if include is not None and include.search(name) is None:
            selected = False
        if exclude is not None and exclude.search(name) is not None:
            selected = False
        if selected:
            selected_names.append(name)
        else:
            skipped_names.append(name)

    if qconfig_name == "default":
        qconfig = tq.default_dynamic_qconfig
    elif qconfig_name == "per_channel":
        qconfig = tq.per_channel_dynamic_qconfig
    else:
        raise ValueError(f"unsupported dynamic int8 qconfig: {qconfig_name}")

    if include is None and exclude is None:
        qconfig_spec = {torch.nn.Linear: qconfig}
    else:
        qconfig_spec = {name: qconfig for name in selected_names}

    quantized = tq.quantize_dynamic(model_fp32, qconfig_spec, dtype=torch.qint8, inplace=False)
    quantized_linears = sum(1 for module in quantized.modules() if "quantized.dynamic" in module.__class__.__module__)
    compat_wrapped_linears = _wrap_dynamic_quantized_linears(quantized)
    return quantized, {
        "enabled": True,
        "kind": "torch_dynamic_qint8_linear",
        "include_regex": include_regex,
        "exclude_regex": exclude_regex,
        "qconfig": qconfig_name,
        "selected_linear_count": int(len(selected_names)),
        "selected_linear_sample": selected_names[:50],
        "skipped_linear_count": int(len(skipped_names)),
        "skipped_linear_sample": skipped_names[:50],
        "quantized_dynamic_modules": int(quantized_linears),
        "compat_wrapped_linears": int(compat_wrapped_linears),
        "elapsed_sec": time.time() - started,
        "note": "Only selected nn.Linear modules are int8 dynamic; non-linear ops remain float. Compat wrappers expose .weight.dtype for TripoSplat.",
    }


class FlowEulerCfgMultiVariantSampler:
    """Batch several guidance/shift variants into the same flow forward calls."""

    def __init__(
        self,
        fake_quant: FakeQuantConfig | None = None,
        quantize_points: tuple[str, ...] = ("state", "prediction"),
        cfg_split_forward: bool = False,
        cfg_single_state_forward: bool = False,
        solver: str = "euler",
        clone_state_each_step: bool = True,
        native_sampler_backend=None,
    ):
        if solver not in {"euler", "ab2"}:
            raise ValueError(f"unsupported flow sampler solver={solver!r}; expected euler or ab2")
        self.fake_quant = fake_quant or FakeQuantConfig()
        self.quantize_points = set(quantize_points)
        self.cfg_split_forward = bool(cfg_split_forward)
        self.cfg_single_state_forward = bool(cfg_single_state_forward)
        if self.cfg_split_forward and self.cfg_single_state_forward:
            raise ValueError("cfg_single_state_forward is only valid for non-split batched CFG")
        self.solver = solver
        self.clone_state_each_step = bool(clone_state_each_step)
        self.native_sampler_backend = native_sampler_backend

    def _maybe_quant(self, name: str, value):
        if name not in self.quantize_points:
            return value
        return fake_quant_state(value, self.fake_quant)

    @torch.no_grad()
    def sample_group(self, model, noise: dict, cond: dict, neg_cond: dict, variants: list[FlowVariant], show_progress: bool = False, trace_callback=None):
        if not variants:
            return {}, {}
        steps = int(variants[0].steps)
        if any(int(v.steps) != steps for v in variants):
            raise ValueError("sample_group requires variants with identical steps")
        batch = len(variants)
        sample = _state_cat([_state_clone(noise) for _ in variants])
        cond_b = _state_repeat(cond, batch)
        neg_b = _state_repeat(neg_cond, batch)
        guidance = torch.tensor([float(v.guidance_scale) for v in variants], dtype=torch.float32, device=next(iter(sample.values())).device)
        shifts = np.array([float(v.shift) for v in variants], dtype=np.float64)
        lin = np.linspace(1.0, 0.0, steps + 1, dtype=np.float64)[None, :]
        t_seq = shifts[:, None] * lin / (1.0 + (shifts[:, None] - 1.0) * lin)
        iterator = range(steps)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Sampling {batch} quantized variants", total=steps)
        timings = []
        quant_error = []
        prev_cfg_pred = None
        prev_dt = None
        static_condition_ok = self.fake_quant.mode == "none" or "condition" not in self.quantize_points
        static_cond_in = self._maybe_quant("condition", cond_b) if static_condition_ok else None
        static_neg_in = self._maybe_quant("condition", neg_b) if static_condition_ok else None
        static_cond_pair = _cat_cond_neg(static_cond_in, static_neg_in) if static_condition_ok else None
        for i in iterator:
            t = torch.tensor(t_seq[:, i] * 1000.0, dtype=torch.float32, device=guidance.device)
            dt = torch.tensor(t_seq[:, i] - t_seq[:, i + 1], dtype=torch.float32, device=guidance.device)
            state_source = _state_clone(sample) if self.clone_state_each_step else sample
            state_in = self._maybe_quant("state", state_source)
            if "state" in self.quantize_points and self.fake_quant.mode != "none":
                quant_error.append({"step": i + 1, "point": "state", "error": state_error_stats(sample, state_in)})
            if static_condition_ok:
                cond_in = static_cond_in
                neg_in = static_neg_in
                cond_pair = static_cond_pair
            else:
                cond_in = self._maybe_quant("condition", cond_b)
                neg_in = self._maybe_quant("condition", neg_b)
                cond_pair = _cat_cond_neg(cond_in, neg_in)
            if trace_callback is not None:
                trace_callback("pre_forward", {"step": i + 1, "state": state_in, "t": t, "cond": cond_in, "neg_cond": neg_in})
            started = time.time()
            if self.cfg_split_forward:
                pred_v_raw = _model_forward(model, state_in, t, cond_in)
                neg_pred_v_raw = _model_forward(model, state_in, t, neg_in)
                pred2 = _state_cat([pred_v_raw, neg_pred_v_raw])
            else:
                t_pair = torch.cat([t, t], dim=0)
                if self.cfg_single_state_forward:
                    pred2 = _model_forward(model, state_in, t_pair, cond_pair)
                else:
                    pred2 = _model_forward(model, _cat_state2(state_in), t_pair, cond_pair)
            elapsed = time.time() - started
            if trace_callback is not None:
                trace_callback("post_forward_raw", {"step": i + 1, "prediction": pred2})
            pred_v, neg_pred_v = _split_pred2(pred2)
            pred_v = self._maybe_quant("prediction", pred_v)
            neg_pred_v = self._maybe_quant("prediction", neg_pred_v)
            if "prediction" in self.quantize_points and self.fake_quant.mode != "none":
                # Store only the first two steps to keep manifests small.
                if i < 2:
                    pred_ref, neg_ref = _split_pred2(pred2)
                    quant_error.append({"step": i + 1, "point": "prediction", "error": {"cond": state_error_stats(pred_ref, pred_v), "neg": state_error_stats(neg_ref, neg_pred_v)}})
            native_cfg = False
            if self.native_sampler_backend is not None:
                native_cfg = self.native_sampler_backend.cfg_combine_inplace(pred_v, neg_pred_v, guidance)
            if not native_cfg:
                for key in pred_v:
                    shape = [batch] + [1] * (pred_v[key].dim() - 1)
                    g = guidance.reshape(shape).to(dtype=pred_v[key].dtype, device=pred_v[key].device)
                    pred_v[key] = g * pred_v[key] - (g - 1.0) * neg_pred_v[key]
            cfg_pred = self._maybe_quant("cfg_prediction", pred_v)
            if "cfg_prediction" in self.quantize_points and self.fake_quant.mode != "none" and i < 2:
                quant_error.append({"step": i + 1, "point": "cfg_prediction", "error": state_error_stats(pred_v, cfg_pred)})
            pred_v = cfg_pred
            if trace_callback is not None:
                trace_callback("post_cfg_prediction", {"step": i + 1, "prediction": pred_v, "negative_prediction": neg_pred_v, "dt": dt})
            native_update = False
            if self.native_sampler_backend is not None:
                if self.solver == "ab2" and prev_cfg_pred is not None and prev_dt is not None:
                    native_update = self.native_sampler_backend.ab2_update_inplace(
                        sample, pred_v, prev_cfg_pred, dt, prev_dt
                    )
                else:
                    native_update = self.native_sampler_backend.euler_update_inplace(sample, pred_v, dt)
            if not native_update:
                solver_pred = pred_v
                if self.solver == "ab2" and prev_cfg_pred is not None and prev_dt is not None:
                    # Variable-step Adams-Bashforth 2 for x_{i+1}=x_i-dt*f_i.
                    # With r=dt_i/dt_{i-1}, f_eff=(1+r/2)f_i-(r/2)f_{i-1}.
                    solver_pred = {}
                    ratio = dt / prev_dt.clamp_min(1e-12)
                    for key, value in pred_v.items():
                        shape = [batch] + [1] * (value.dim() - 1)
                        r = ratio.reshape(shape).to(device=value.device, dtype=value.dtype)
                        solver_pred[key] = (1.0 + 0.5 * r) * value - (0.5 * r) * prev_cfg_pred[key].to(device=value.device, dtype=value.dtype)
                sample = _state_update(sample, solver_pred, dt)
            if self.solver == "ab2":
                prev_cfg_pred = _state_clone(pred_v)
                prev_dt = dt.clone()
            updated_sample = self._maybe_quant("updated_state", sample)
            if "updated_state" in self.quantize_points and self.fake_quant.mode != "none" and i < 2:
                quant_error.append({"step": i + 1, "point": "updated_state", "error": state_error_stats(sample, updated_sample)})
            sample = updated_sample
            if trace_callback is not None:
                trace_callback("post_update", {"step": i + 1, "sample": sample})
            timings.append({"step": i + 1, "forward_elapsed_sec": elapsed})
        split = _state_split(sample, batch)
        outputs = {variant.name: split[i] for i, variant in enumerate(variants)}
        metadata = {
            "steps": steps,
            "batch_variants": batch,
            "variants": [variant.__dict__ for variant in variants],
            "fake_quant": self.fake_quant.__dict__,
            "quantize_points": sorted(self.quantize_points),
            "cfg_split_forward": bool(self.cfg_split_forward),
            "cfg_single_state_forward": bool(self.cfg_single_state_forward),
            "solver": self.solver,
            "clone_state_each_step": bool(self.clone_state_each_step),
            "solver_first_step": "euler" if self.solver == "ab2" else self.solver,
            "solver_formula": "variable_step_ab2: f_eff=(1+r/2)f_i-(r/2)f_{i-1}, r=dt_i/dt_{i-1}" if self.solver == "ab2" else "euler",
            "native_avx512_sampler": (
                self.native_sampler_backend.metadata()
                if self.native_sampler_backend is not None
                else {"enabled": False}
            ),
            "static_condition_reuse": bool(static_condition_ok),
            "timings": timings,
            "quantization_error_samples": quant_error,
        }
        return outputs, metadata

    @torch.no_grad()
    def sample(self, model, noise: dict, cond: dict, neg_cond: dict, variants: list[FlowVariant], show_progress: bool = False, trace_callback=None):
        outputs = {}
        groups_meta = []
        for steps, group in group_variants_by_steps(variants).items():
            group_outputs, meta = self.sample_group(model, noise, cond, neg_cond, group, show_progress=show_progress, trace_callback=trace_callback)
            outputs.update(group_outputs)
            groups_meta.append({"steps": int(steps), **meta})
        return outputs, {"groups": groups_meta}


def _toy_self_test() -> None:
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

        def forward(self, x_t, t, cond):
            t_term = t.reshape(-1, 1, 1).to(x_t["latent"].dtype) / 1000.0
            return {"latent": self.lin(x_t["latent"] + cond["feature"] * 0.1) * (0.5 + t_term)}

    torch.manual_seed(7)
    model = ToyModel().eval()
    noise = {"latent": torch.randn(1, 3, 4)}
    cond = {"feature": torch.randn(1, 3, 4)}
    neg = {"feature": torch.zeros_like(cond["feature"])}
    variants = parse_flow_variants("a:2:3:3,b:2:2:2", 2, 3, 3)
    sampler = FlowEulerCfgMultiVariantSampler(FakeQuantConfig(mode="none"))
    out, meta = sampler.sample(model, noise, cond, neg, variants)
    assert sorted(out) == ["a", "b"]
    assert out["a"]["latent"].shape == noise["latent"].shape
    assert meta["groups"][0]["batch_variants"] == 2
    class SingleStateAwareToyModel(ToyModel):
        def forward(self, x_t, t, cond):
            latent = x_t["latent"]
            feature = cond["feature"]
            if feature.shape[0] == latent.shape[0] * 2:
                latent = latent.repeat(2, *([1] * (latent.dim() - 1)))
            t_term = t.reshape(-1, 1, 1).to(latent.dtype) / 1000.0
            return {"latent": self.lin(latent + feature * 0.1) * (0.5 + t_term)}

    single_model = SingleStateAwareToyModel().eval()
    single_model.load_state_dict(model.state_dict())
    regular_sampler = FlowEulerCfgMultiVariantSampler(FakeQuantConfig(mode="none"), cfg_single_state_forward=False)
    single_sampler = FlowEulerCfgMultiVariantSampler(FakeQuantConfig(mode="none"), cfg_single_state_forward=True)
    out_regular, _ = regular_sampler.sample(single_model, noise, cond, neg, parse_flow_variants("single:2:3:3", 2, 3, 3))
    out_single, meta_single = single_sampler.sample(single_model, noise, cond, neg, parse_flow_variants("single:2:3:3", 2, 3, 3))
    assert meta_single["groups"][0]["cfg_single_state_forward"] is True
    assert torch.equal(out_regular["single"]["latent"], out_single["single"]["latent"])

    sampler_ab2 = FlowEulerCfgMultiVariantSampler(FakeQuantConfig(mode="none"), solver="ab2")
    out_ab2, meta_ab2 = sampler_ab2.sample(model, noise, cond, neg, parse_flow_variants("ab2:3:3:3", 3, 3, 3))
    assert out_ab2["ab2"]["latent"].shape == noise["latent"].shape
    assert torch.isfinite(out_ab2["ab2"]["latent"]).all()
    assert meta_ab2["groups"][0]["solver"] == "ab2"
    q = fake_quant_tensor(torch.tensor([-10.0, -1.0, -0.01, 0.0, 0.01, 1.0, 10.0]), FakeQuantConfig(mode="log_symmetric", bits=6))
    assert torch.isfinite(q).all()
    q = fake_quant_tensor(torch.linspace(-1, 1, 17), FakeQuantConfig(mode="mulaw", bits=6))
    assert torch.isfinite(q).all()
    q_model, q_meta = dynamic_quantize_linear_modules(model, True)
    assert q_meta["enabled"]
    y = q_model({"latent": torch.randn(1, 3, 4)}, torch.ones(1), {"feature": torch.zeros(1, 3, 4)})
    assert "latent" in y


if __name__ == "__main__":
    _toy_self_test()
    print("triposplat_quantized_sampler self-test passed")
