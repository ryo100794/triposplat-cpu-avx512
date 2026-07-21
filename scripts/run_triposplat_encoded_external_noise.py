#!/usr/bin/env python3
"""Run TripoSplat from a prepared RGB image with explicit flow noise control.

This runner keeps the official TripoSplat encode -> sample -> decode pipeline,
but exposes the initial flow noise so CPU/GPU and low-resource sampler variants
can be compared from the same state.  It does not change the TripoSplat model;
it changes only the implementation boundary used for evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import hashlib

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get("GS_PROJECT_ROOT", SCRIPT_DIR.parent)).resolve()
BASE = Path(os.environ.get("GS_BASE", PROJECT.parent)).resolve()
REPO = Path(os.environ.get("TRIPOSPLAT_REPO", PROJECT / "vendor" / "TripoSplat")).resolve()
CKPTS = Path(os.environ.get("TRIPOSPLAT_CKPTS", PROJECT / "models" / "TripoSplat" / "ckpts")).resolve()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def cast_gaussian_export_tensors(gaussian) -> None:
    import torch

    for key, value in list(gaussian._storage.items()):
        if isinstance(value, torch.Tensor) and value.dtype in (torch.bfloat16, torch.float16):
            gaussian._storage[key] = value.float()
    for name in ("aabb", "scale_bias", "rots_bias", "opacity_bias_val"):
        value = getattr(gaussian, name, None)
        if isinstance(value, torch.Tensor) and value.dtype in (torch.bfloat16, torch.float16):
            setattr(gaussian, name, value.float())


def resolve_dtype(name: str, device_type: str):
    import torch

    if name == "auto":
        return torch.float16 if device_type == "cuda" else torch.bfloat16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def encode_image_controlled(pipe, image, generator, vae_deterministic: bool):
    """Equivalent to triposplat.encode_image, with optional deterministic VAE."""
    if not vae_deterministic:
        return pipe.encode_image(image, generator=generator)

    import torch
    import torch.nn.functional as F
    import triposplat
    from torchvision import transforms

    device = next(pipe.dinov3.parameters()).device
    img_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device=device, dtype=torch.float32)
    img_normed = triposplat._DINOV3_NORMALIZE(img_tensor)
    dinov3_dtype = next(pipe.dinov3.parameters()).dtype
    vae_dtype = next(pipe.vae_encoder.parameters()).dtype
    dinov3_feat = pipe.dinov3(pixel_values=img_normed.to(dinov3_dtype))
    dinov3_feat = F.layer_norm(dinov3_feat.float(), dinov3_feat.shape[-1:])
    vae_feat = pipe.vae_encoder.encode(img_tensor.to(vae_dtype) * 2 - 1, deterministic=True, generator=None)
    zero_reg = torch.zeros(vae_feat.shape[0], 5, vae_feat.shape[2], dtype=vae_feat.dtype, device=vae_feat.device)
    vae_feat = torch.cat([zero_reg, vae_feat], dim=1)
    return {"feature1": dinov3_feat, "feature2": vae_feat}


def expected_noise_shapes(flow_model) -> dict[str, tuple[int, ...]]:
    shapes = {"latent": (1, int(flow_model.q_token_length), int(flow_model.in_channels))}
    if flow_model.cam_channels is not None:
        shapes["camera"] = (1, 1, int(flow_model.cam_channels))
    return shapes


def numpy_noise(shapes: dict[str, tuple[int, ...]], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    return {key: rng.standard_normal(shape).astype(np.float32) for key, shape in shapes.items()}


def torch_noise(shapes: dict[str, tuple[int, ...]], device, generator) -> dict:
    import torch

    return {key: torch.randn(*shape, device=device, generator=generator) for key, shape in shapes.items()}


def load_noise_npz(path: Path, shapes: dict[str, tuple[int, ...]]) -> dict[str, np.ndarray]:
    data = np.load(path)
    out = {}
    for key, shape in shapes.items():
        if key not in data:
            raise ValueError(f"noise npz is missing {key!r}: {path}")
        arr = np.asarray(data[key], dtype=np.float32)
        if tuple(arr.shape) != tuple(shape):
            raise ValueError(f"noise {key!r} shape mismatch: expected {shape}, got {arr.shape}")
        out[key] = arr
    return out


def tensors_from_numpy(noise_np: dict[str, np.ndarray], device) -> dict:
    import torch

    return {key: torch.from_numpy(arr).to(device=device) for key, arr in noise_np.items()}


def save_noise_npz(path: Path, noise_np: dict[str, np.ndarray], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **noise_np, metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)))


def save_tensor_dict_npz(path: Path, values: dict, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: value.detach().float().cpu().numpy() for key, value in values.items()}
    np.savez_compressed(path, **arrays, metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)))


def load_tensor_dict_npz(path: Path, device) -> tuple[dict, dict]:
    import torch

    data = np.load(path)
    metadata = {}
    if "metadata_json" in data:
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    out = {}
    for key in data.files:
        if key == "metadata_json":
            continue
        out[key] = torch.from_numpy(np.asarray(data[key], dtype=np.float32)).to(device=device)
    if "latent" not in out:
        raise ValueError(f"latent npz is missing 'latent': {path}")
    return out, metadata


def save_condition_npz(path: Path, cond: dict, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: value.detach().float().cpu().numpy() for key, value in cond.items()}
    tensor_meta = {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
            "sha256_float32": tensor_sha256(value),
        }
        for key, value in cond.items()
    }
    payload_meta = {**metadata, "tensors": tensor_meta, "storage_dtype": "float32"}
    np.savez_compressed(path, **arrays, metadata_json=np.array(json.dumps(payload_meta, ensure_ascii=False)))


def load_condition_npz(path: Path, device, model_dtype) -> tuple[dict, dict]:
    import torch

    data = np.load(path)
    metadata = {}
    if "metadata_json" in data:
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    missing = [key for key in ("feature1", "feature2") if key not in data]
    if missing:
        raise ValueError(f"condition npz is missing {missing}: {path}")
    cond = {
        "feature1": torch.from_numpy(np.asarray(data["feature1"], dtype=np.float32)).to(device=device, dtype=torch.float32),
        "feature2": torch.from_numpy(np.asarray(data["feature2"], dtype=np.float32)).to(device=device, dtype=model_dtype),
    }
    return cond, metadata


def tensor_sha256(tensor) -> str:
    arr = tensor.detach().float().cpu().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def tensor_dict_sha256(values: dict) -> dict[str, str]:
    return {key: tensor_sha256(value) for key, value in values.items()}


def tensor_fingerprint(tensor) -> dict:
    arr = tensor.detach().float().cpu().contiguous().numpy()
    finite = np.isfinite(arr)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "sha256_float32": hashlib.sha256(arr.tobytes()).hexdigest(),
        "finite_count": int(finite.sum()),
        "nan_count": int(np.isnan(arr).sum()),
        "posinf_count": int(np.isposinf(arr).sum()),
        "neginf_count": int(np.isneginf(arr).sum()),
    }


def tensor_dict_fingerprint(values: dict) -> dict[str, dict]:
    return {key: tensor_fingerprint(value) for key, value in values.items()}


def gaussian_fingerprint(gaussian) -> dict[str, dict]:
    import torch

    out = {}
    for key in sorted(gaussian._storage.keys()):
        value = gaussian._storage[key]
        if isinstance(value, torch.Tensor):
            out[f"storage.{key}"] = tensor_fingerprint(value)
    for name in ("aabb", "scale_bias", "rots_bias", "opacity_bias_val"):
        value = getattr(gaussian, name, None)
        if isinstance(value, torch.Tensor):
            out[f"attr.{name}"] = tensor_fingerprint(value)
    return out


class DecoderRandomProvider:
    def __init__(self, mode: str, seed: int, npz_path: Path | None = None):
        self.mode = "npz" if npz_path else mode
        self.seed = int(seed)
        self.npz_path = npz_path
        self.rng = np.random.default_rng(self.seed)
        self.loaded = np.load(npz_path) if npz_path else None
        self.counts = {}
        self.records = []
        self.arrays = {}

    def rand(self, shape: tuple[int, ...], device, kind: str, context: dict) -> object:
        import torch

        idx = self.counts.get(kind, 0)
        self.counts[kind] = idx + 1
        key = f"{kind}_{idx:03d}"
        if self.loaded is not None:
            if key not in self.loaded:
                raise ValueError(f"decoder random npz is missing {key!r}: {self.npz_path}")
            arr = np.asarray(self.loaded[key], dtype=np.float32)
            if tuple(arr.shape) != tuple(shape):
                raise ValueError(f"decoder random {key!r} shape mismatch: expected {shape}, got {arr.shape}")
            source = "loaded_npz"
        else:
            arr = self.rng.random(shape, dtype=np.float32)
            source = "numpy_pcg64_random"
        sha = hashlib.sha256(arr.tobytes()).hexdigest()
        self.arrays[key] = arr
        self.records.append({
            "key": key,
            "kind": kind,
            "shape": list(shape),
            "sha256": sha,
            "source": source,
            **context,
        })
        return torch.from_numpy(arr).to(device=device)

    def save_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = self.manifest(save_path=path)
        np.savez_compressed(path, **self.arrays, metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)))

    def manifest(self, save_path: Path | None = None) -> dict:
        return {
            "mode": self.mode,
            "seed": self.seed,
            "npz": self.npz_path.as_posix() if self.npz_path else None,
            "saved_npz": save_path.as_posix() if save_path else None,
            "records": self.records,
        }


def sample_probs_external(probs, counts, provider: DecoderRandomProvider, level: int):
    import torch

    batch_shape = counts.shape
    batch = counts.numel()
    points = probs.size(-1)
    device = probs.device
    probs = probs.view(batch, points)
    counts = counts.view(batch)

    probs = probs.to(torch.float32).clamp_min_(0)
    row_sums = probs.sum(1, keepdim=True)
    zero_mask = row_sums.eq(0)
    probs = probs / row_sums.clamp_min_(1)
    if zero_mask.any():
        probs = probs.clone()
        probs[zero_mask.expand_as(probs)] = 1.0 / points

    counts = counts.to(device=device, dtype=torch.long)
    out = torch.zeros(batch, points, dtype=torch.long, device=device)
    cdf = probs.cumsum(dim=1).clamp(max=1.0 - 1e-12)
    unique_n, inv = counts.unique(sorted=False, return_inverse=True)
    for group_index, n in enumerate(unique_n.tolist()):
        if n == 0:
            continue
        rows = (inv == group_index).nonzero(as_tuple=False).squeeze(1)
        row_count = rows.numel()
        u0 = provider.rand(
            (row_count, 1),
            device=device,
            kind="u0",
            context={"level": int(level), "count": int(n), "rows": int(row_count)},
        ) / float(n)
        grid = torch.arange(n, device=device, dtype=torch.float32)[None, :] / float(n)
        us = (u0 + grid).clamp(max=1.0 - 1e-12)
        cdf_rows = cdf.index_select(0, rows)
        idx = torch.searchsorted(cdf_rows, us).clamp_max(probs.size(1) - 1)
        buf = torch.zeros(row_count, points, dtype=torch.float32, device=device)
        buf.scatter_add_(1, idx, torch.ones_like(idx, dtype=buf.dtype))
        out.index_copy_(0, rows, buf.to(torch.long))

    return out.view(*batch_shape, points)


def sample_octree_external(model, cond, num_points, level, provider: DecoderRandomProvider, temperature=1.0, algo="systematic"):
    import torch

    if algo != "systematic":
        raise ValueError(f"external decoder random currently supports systematic only, got {algo!r}")
    batch = cond.shape[0]
    device = cond.device
    child_offset = torch.tensor([[i, j, k] for k in [0, 1] for j in [0, 1] for i in [0, 1]],
                                dtype=torch.long, device=device)
    prev_coords_int = torch.zeros(batch, 1, 3, dtype=torch.long, device=device)
    prev_counts = torch.full((batch, 1), int(num_points), dtype=torch.long, device=device)
    prev_log_probs = torch.zeros(batch, 1, dtype=torch.float32, device=device)
    batch_indices_range = torch.arange(batch, device=device).unsqueeze(1)
    num_tensor = torch.full((batch,), int(num_points), dtype=torch.long, device=device)

    for lv in range(1, int(level) + 1):
        res_p = 1 << (lv - 1)
        res = 1 << lv
        parent_coords_norm = (prev_coords_int.to(torch.float32) + 0.5) / res_p
        res_tensor = torch.full((batch,), res, dtype=torch.long, device=device)
        pred_logits = model(parent_coords_norm, res_tensor, cond, num_tensor)["logits"] / temperature
        pred_probs = torch.softmax(pred_logits, dim=-1)
        pred_log_probs = torch.log_softmax(pred_logits, dim=-1)
        sampled = sample_probs_external(pred_probs, prev_counts, provider=provider, level=lv).flatten(1, 2)
        pred_log_probs = pred_log_probs.flatten(1, 2)
        prev_log_probs_expanded = prev_log_probs.repeat_interleave(8, dim=1)
        child_coords_int = (prev_coords_int[:, :, None, :] * 2 + child_offset[None, None, :, :]).flatten(1, 2)
        mask = sampled > 0
        max_valid = mask.sum(dim=1).max().item()
        scatter_indices = mask.cumsum(dim=1) - 1
        valid_scatter_indices = scatter_indices[mask]
        valid_batch_indices = batch_indices_range.expand_as(mask)[mask]
        next_prev_coords_int = torch.zeros(batch, max_valid, 3, dtype=child_coords_int.dtype, device=device)
        next_prev_coords_int[valid_batch_indices, valid_scatter_indices] = child_coords_int[mask]
        next_prev_counts = torch.zeros(batch, max_valid, dtype=sampled.dtype, device=device)
        next_prev_counts[valid_batch_indices, valid_scatter_indices] = sampled[mask]
        next_prev_log_probs = torch.zeros(batch, max_valid, dtype=prev_log_probs.dtype, device=device)
        next_prev_log_probs[valid_batch_indices, valid_scatter_indices] = (prev_log_probs_expanded + pred_log_probs)[mask]
        prev_coords_int = next_prev_coords_int
        prev_counts = next_prev_counts
        prev_log_probs = next_prev_log_probs

    res = 1 << int(level)
    prev_log_probs = torch.repeat_interleave(prev_log_probs.flatten(0, 1), prev_counts.flatten(0, 1), dim=0).reshape(batch, num_points)
    coords_int = torch.repeat_interleave(prev_coords_int.flatten(0, 1), prev_counts.flatten(0, 1), dim=0).reshape(batch, num_points, -1)
    jitter = provider.rand(
        tuple(coords_int.shape),
        device=device,
        kind="jitter",
        context={"level": int(level), "num_points": int(num_points), "res": int(res)},
    )
    coords_norm = (coords_int.to(torch.float32) + jitter) / res
    return {"points": coords_norm, "log_probs": prev_log_probs}


def make_external_decoder_sample(provider: DecoderRandomProvider):
    def sample(model, cond, num_points, level, temperature=1.0, algo="systematic"):
        return sample_octree_external(
            model, cond, num_points=num_points, level=level,
            provider=provider, temperature=temperature, algo=algo,
        )
    return sample


def build_noise(flow_model, args, device, generator) -> tuple[dict, dict, dict[str, np.ndarray] | None]:
    shapes = expected_noise_shapes(flow_model)
    metadata = {
        "created_at": utc_now(),
        "shape": {key: list(value) for key, value in shapes.items()},
        "seed": int(args.seed),
        "noise_mode": args.noise_mode,
        "noise_npz": args.noise_npz.as_posix() if args.noise_npz else None,
    }

    if args.noise_npz:
        noise_np = load_noise_npz(args.noise_npz, shapes)
        noise = tensors_from_numpy(noise_np, device)
        metadata["source"] = "loaded_npz"
        return noise, metadata, noise_np

    if args.noise_mode == "numpy":
        noise_np = numpy_noise(shapes, int(args.seed))
        noise = tensors_from_numpy(noise_np, device)
        metadata["source"] = "numpy_pcg64_standard_normal"
        return noise, metadata, noise_np

    noise = torch_noise(shapes, device, generator)
    noise_np = {key: value.detach().float().cpu().numpy() for key, value in noise.items()}
    metadata["source"] = f"torch_randn_{device.type}"
    return noise, metadata, noise_np


def sample_latent_from_noise(flow_model, noise, cond, steps, guidance_scale, shift, show_progress, use_lowmem, separate_cfg):
    import torch
    import triposplat

    neg_cond = {key: torch.zeros_like(value) for key, value in cond.items()}
    if use_lowmem:
        from triposplat_lowmem_sampler import FlowEulerCfgSamplerLowmem

        sampler = FlowEulerCfgSamplerLowmem(batched_cfg=not separate_cfg)
        sampler_name = "lowmem_separate_cfg" if separate_cfg else "lowmem_batched_cfg"
    else:
        sampler = triposplat.FlowEulerCfgSampler()
        sampler_name = "official"
    out = sampler.sample(
        flow_model,
        {key: value.clone() for key, value in noise.items()},
        cond=cond,
        neg_cond=neg_cond,
        steps=int(steps),
        guidance_scale=float(guidance_scale),
        shift=float(shift),
        show_progress=show_progress,
    )
    return out, sampler_name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Preprocessed RGB image, normally prepared_rgb.webp")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-gaussians", type=int, default=32768)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--canvas-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--model-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--decoder-dtype", choices=["same", "auto", "bfloat16", "float16", "float32"], default="same", help="Override decoder dtype while keeping other models at --model-dtype")
    parser.add_argument("--decoder-random-mode", choices=["torch", "numpy"], default="torch", help="Control decoder octree/jitter random source; torch preserves official behavior")
    parser.add_argument("--decoder-random-seed", type=int, help="Seed for --decoder-random-mode numpy; defaults to --seed")
    parser.add_argument("--decoder-random-npz", type=Path, help="Load decoder octree/jitter random values from an npz trace")
    parser.add_argument("--save-decoder-random-npz", type=Path, help="Save decoder octree/jitter random values actually used")
    parser.add_argument("--noise-mode", choices=["torch", "numpy"], default="numpy")
    parser.add_argument("--noise-npz", type=Path, help="Load flow initial noise from an npz file")
    parser.add_argument("--save-noise-npz", type=Path, help="Save the flow initial noise actually used")
    parser.add_argument("--latent-npz", type=Path, help="Load sampled flow latent from an npz file and skip encode/sample")
    parser.add_argument("--save-latent-npz", type=Path, help="Save sampled flow latent for decoder-only sweeps")
    parser.add_argument("--condition-npz", type=Path, help="Load encoded condition features from an npz file and skip DINO/VAE encode")
    parser.add_argument("--save-condition-npz", type=Path, help="Save encoded condition features for flow-only equivalence sweeps")
    parser.add_argument("--vae-deterministic", action="store_true", help="Use deterministic VAE encode for equivalence experiments")
    parser.add_argument("--deterministic-torch", action="store_true", help="Enable torch deterministic mode for reproducibility experiments")
    parser.add_argument("--disable-mkldnn", action="store_true", help="Disable MKLDNN to reduce CPU nondeterminism in reproducibility experiments")
    parser.add_argument("--lowmem-export", action="store_true", help="stream PLY/SPLAT export without final full output buffers")
    parser.add_argument("--export-chunk-size", type=int, default=32768)
    parser.add_argument("--lowmem-sampler", action="store_true", help="use result-equivalent low-resource sampler implementation")
    parser.add_argument("--separate-cfg", action="store_true", help="with --lowmem-sampler, keep official two-call CFG instead of batch-2 CFG")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO))
    import torch
    import triposplat
    from PIL import Image
    from triposplat import TripoSplatPipeline, load_decoder, load_dinov3, load_flow_model, load_vae_encoder

    if args.deterministic_torch:
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
        torch.use_deterministic_algorithms(True, warn_only=True)
    if args.disable_mkldnn and hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))

    triposplat._CANVAS_SIZE = int(args.canvas_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = resolve_dtype(args.model_dtype, device.type)
    decoder_dtype = dtype if args.decoder_dtype == "same" else resolve_dtype(args.decoder_dtype, device.type)
    t0 = time.time()

    pipe = TripoSplatPipeline.__new__(TripoSplatPipeline)
    pipe._device = device
    pipe.rmbg = None
    pipe.dinov3 = None
    pipe.vae_encoder = None
    pipe.flow_model = None
    if not args.latent_npz:
        if not args.condition_npz:
            pipe.dinov3 = load_dinov3(str(CKPTS / "clip_vision/dino_v3_vit_h.safetensors"), device=device, dtype=dtype)
            pipe.vae_encoder = load_vae_encoder(str(CKPTS / "vae/flux2-vae.safetensors"), device=device, dtype=dtype)
        pipe.flow_model = load_flow_model(str(CKPTS / "diffusion_models/triposplat_fp16.safetensors"), device=device, dtype=dtype)
    pipe.decoder = load_decoder(str(CKPTS / "vae/triposplat_vae_decoder_fp16.safetensors"), device=device, dtype=decoder_dtype)

    prepared = Image.open(args.input).convert("RGB")
    gen = torch.Generator(device=device).manual_seed(int(args.seed))
    cond_sha256 = {}
    noise_sha256 = {}
    latent_sha256 = {}
    condition_fingerprint = {}
    noise_fingerprint = {}
    latent_fingerprint = {}
    noise_meta = None
    condition_meta = None
    latent_meta = None
    decoder_random_provider = None
    decoder_random_meta = {
        "mode": "torch_global_rng",
        "seed": int(args.seed),
        "note": "official decoder random behavior",
    }
    with torch.inference_mode():
        if args.latent_npz:
            out, loaded_latent_meta = load_tensor_dict_npz(args.latent_npz, device=device)
            sampler_name = "loaded_latent_npz"
            latent_meta = {
                "source": "loaded_npz",
                "latent_npz": args.latent_npz.as_posix(),
                "loaded_metadata": loaded_latent_meta,
            }
        else:
            if args.condition_npz:
                cond, loaded_condition_meta = load_condition_npz(args.condition_npz, device=device, model_dtype=dtype)
                condition_meta = {
                    "source": "loaded_npz",
                    "condition_npz": args.condition_npz.as_posix(),
                    "loaded_metadata": loaded_condition_meta,
                    "load_cast": {"feature1": "float32", "feature2": str(dtype).replace("torch.", "")},
                }
            else:
                cond = encode_image_controlled(pipe, prepared, generator=gen, vae_deterministic=bool(args.vae_deterministic))
                condition_meta = {
                    "source": "encoded",
                    "vae_deterministic": bool(args.vae_deterministic),
                    "model_dtype": str(dtype).replace("torch.", ""),
                }
            cond_sha256 = tensor_dict_sha256(cond)
            condition_fingerprint = tensor_dict_fingerprint(cond)
            if args.save_condition_npz:
                save_condition_npz(args.save_condition_npz, cond, {**(condition_meta or {}), "created_at": utc_now()})
                condition_meta = {**(condition_meta or {}), "saved_npz": args.save_condition_npz.as_posix()}
            noise, noise_meta, noise_np = build_noise(pipe.flow_model, args, device, gen)
            noise_sha256 = tensor_dict_sha256(noise)
            noise_fingerprint = tensor_dict_fingerprint(noise)
            if args.save_noise_npz and noise_np is not None:
                save_noise_npz(args.save_noise_npz, noise_np, noise_meta)
            out, sampler_name = sample_latent_from_noise(
                pipe.flow_model,
                noise,
                cond,
                steps=int(args.steps),
                guidance_scale=float(args.guidance_scale),
                shift=float(args.shift),
                show_progress=True,
                use_lowmem=bool(args.lowmem_sampler),
                separate_cfg=bool(args.separate_cfg),
            )
            latent_meta = {
                "source": "sampled",
                "steps": int(args.steps),
                "guidance_scale": float(args.guidance_scale),
                "shift": float(args.shift),
                "sampler_mode": sampler_name,
            }
        latent_sha256 = tensor_dict_sha256(out)
        latent_fingerprint = tensor_dict_fingerprint(out)
        if args.save_latent_npz:
            save_tensor_dict_npz(args.save_latent_npz, out, {**(latent_meta or {}), "created_at": utc_now()})
            latent_meta = {**(latent_meta or {}), "saved_npz": args.save_latent_npz.as_posix()}
        if args.decoder_random_mode == "numpy" or args.decoder_random_npz:
            from model import OctreeProbabilityFixedlenDecoder

            decoder_random_seed = int(args.decoder_random_seed if args.decoder_random_seed is not None else args.seed)
            decoder_random_provider = DecoderRandomProvider(
                mode=args.decoder_random_mode,
                seed=decoder_random_seed,
                npz_path=args.decoder_random_npz,
            )
            original_sample = OctreeProbabilityFixedlenDecoder.sample
            OctreeProbabilityFixedlenDecoder.sample = staticmethod(make_external_decoder_sample(decoder_random_provider))
            try:
                gaussian = pipe.decode_latent(out["latent"], num_gaussians=int(args.num_gaussians))
            finally:
                OctreeProbabilityFixedlenDecoder.sample = original_sample
            if args.save_decoder_random_npz:
                decoder_random_provider.save_npz(args.save_decoder_random_npz)
            decoder_random_meta = decoder_random_provider.manifest(save_path=args.save_decoder_random_npz)
        else:
            gaussian = pipe.decode_latent(out["latent"], num_gaussians=int(args.num_gaussians))
    gaussian_sha256_before_export_cast = gaussian_fingerprint(gaussian)
    cast_gaussian_export_tensors(gaussian)
    gaussian_sha256_after_export_cast = gaussian_fingerprint(gaussian)

    prepared_path = args.output_dir / "preprocessed_image.webp"
    ply_path = args.output_dir / "output.ply"
    splat_path = args.output_dir / "output.splat"
    prepared.save(prepared_path)
    if args.lowmem_export:
        from triposplat_lowmem_export import save_ply_lowmem, save_splat_lowmem

        save_ply_lowmem(gaussian, ply_path, chunk_size=int(args.export_chunk_size))
        save_splat_lowmem(gaussian, splat_path, chunk_size=int(args.export_chunk_size))
        export_mode = "lowmem_streaming"
    else:
        gaussian.save_ply(str(ply_path))
        gaussian.save_splat(str(splat_path))
        export_mode = "official_materialized"
    elapsed = time.time() - t0

    manifest = {
        "created_at": utc_now(),
        "algorithm": "TripoSplat encoded prepared RGB with external flow noise control",
        "implementation_goal": "low-resource/equivalence TripoSplat inference boundary",
        "ref_usage": "none during inference; ref is only a reference for bounded-memory export/evaluation design",
        "input": args.input.as_posix(),
        "output_dir": args.output_dir.as_posix(),
        "num_gaussians": int(args.num_gaussians),
        "steps": int(args.steps),
        "guidance_scale": float(args.guidance_scale),
        "shift": float(args.shift),
        "canvas_size": int(args.canvas_size),
        "seed": int(args.seed),
        "device": str(device),
        "dtype": f"{str(dtype).replace('torch.', '')}_models_float32_export",
        "flow_model_dtype": str(dtype).replace("torch.", ""),
        "decoder_dtype": str(decoder_dtype).replace("torch.", ""),
        "flow_forward_cast": "LatentSeqMMFlowModel.forward casts latent, camera, feature1, and feature2 to self.dtype before Linear layers",
        "vae_deterministic": bool(args.vae_deterministic),
        "deterministic_torch": bool(args.deterministic_torch),
        "disable_mkldnn": bool(args.disable_mkldnn),
        "condition_sha256": cond_sha256,
        "condition_fingerprint": condition_fingerprint,
        "condition": condition_meta,
        "condition_npz_saved": args.save_condition_npz.as_posix() if args.save_condition_npz else None,
        "noise_sha256": noise_sha256,
        "noise_fingerprint": noise_fingerprint,
        "latent_sha256": latent_sha256,
        "latent_fingerprint": latent_fingerprint,
        "gaussian_sha256_before_export_cast": gaussian_sha256_before_export_cast,
        "gaussian_sha256_after_export_cast": gaussian_sha256_after_export_cast,
        "noise": noise_meta,
        "latent": latent_meta,
        "decoder_random": decoder_random_meta,
        "export_mode": export_mode,
        "export_chunk_size": int(args.export_chunk_size),
        "sampler_mode": sampler_name,
        "preprocess_handling": "prepared RGB encoded directly; no preprocess_image call",
        "elapsed_sec": elapsed,
        "preprocessed_image": prepared_path.as_posix(),
        "noise_npz_saved": args.save_noise_npz.as_posix() if args.save_noise_npz else None,
        "ply": ply_path.as_posix(),
        "splat": splat_path.as_posix(),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
