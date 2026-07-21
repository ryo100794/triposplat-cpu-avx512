#!/usr/bin/env python3
"""Full step-state NPZ recorder for TripoSplat flow equivalence debugging."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("._") or "unnamed"


def _event_state_key(event: str) -> str | None:
    return {
        "pre_forward": "state",
        "post_cfg_prediction": "prediction",
        "post_update": "sample",
    }.get(event)


def _to_numpy(value: torch.Tensor, storage_dtype: str) -> np.ndarray:
    array = value.detach().to(device="cpu")
    if storage_dtype == "float16":
        return array.to(dtype=torch.float16).numpy()
    if storage_dtype == "float32":
        return array.to(dtype=torch.float32).numpy()
    raise ValueError(f"unsupported step-state storage dtype: {storage_dtype}")


def _tensor_fingerprint(array: np.ndarray) -> dict[str, Any]:
    f32 = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(f32)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256_float32": hashlib.sha256(f32.tobytes()).hexdigest(),
        "finite_count": int(finite.sum()),
        "nan_count": int(np.isnan(f32).sum()),
        "posinf_count": int(np.isposinf(f32).sum()),
        "neginf_count": int(np.isneginf(f32).sum()),
    }


class StepStateRecorder:
    """Save full latent/camera tensors at selected sampler events.

    The recorder is intentionally a trace_callback-compatible object so it can
    be composed with existing lightweight JSON trace recorders without changing
    sampler math.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        variants: list[Any],
        events: set[str] | None = None,
        max_steps: int = 0,
        storage_dtype: str = "float32",
    ):
        self.output_dir = Path(output_dir)
        self.variants = list(variants)
        self.variant_names = [_safe_name(str(getattr(variant, "name", f"variant{i}"))) for i, variant in enumerate(self.variants)]
        self.events = set(events or {"post_update"})
        self.max_steps = int(max_steps)
        self.storage_dtype = storage_dtype
        self.records: list[dict[str, Any]] = []

    def callback(self, event: str, payload: dict[str, Any]) -> None:
        if event not in self.events:
            return
        step = int(payload.get("step", 0))
        if self.max_steps > 0 and step > self.max_steps:
            return
        state_key = _event_state_key(event)
        if state_key is None:
            return
        values = payload.get(state_key)
        if not isinstance(values, dict) or not values or not all(torch.is_tensor(v) for v in values.values()):
            return
        batch = next(iter(values.values())).shape[0]
        if batch != len(self.variant_names):
            raise ValueError(
                f"step-state recorder expected batch={len(self.variant_names)} for event {event}, got {batch}"
            )
        for index, variant_name in enumerate(self.variant_names):
            tensors = {
                key: _to_numpy(value[index : index + 1].contiguous(), self.storage_dtype)
                for key, value in sorted(values.items())
            }
            fingerprints = {key: _tensor_fingerprint(array) for key, array in tensors.items()}
            metadata = {
                "created_at": utc_now(),
                "event": event,
                "step": step,
                "variant": variant_name,
                "source_key": state_key,
                "storage_dtype": self.storage_dtype,
                "tensors": fingerprints,
            }
            if torch.is_tensor(payload.get("t")):
                metadata["t"] = [float(v) for v in payload["t"].detach().cpu().reshape(-1).tolist()]
            if torch.is_tensor(payload.get("dt")):
                metadata["dt"] = [float(v) for v in payload["dt"].detach().cpu().reshape(-1).tolist()]
            out_path = self.output_dir / f"{_safe_name(event)}_step{step:04d}_{variant_name}.npz"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_path, **tensors, metadata_json=json.dumps(metadata, ensure_ascii=False))
            record = {
                "event": event,
                "step": step,
                "variant": variant_name,
                "path": out_path.as_posix(),
                "source_key": state_key,
                "storage_dtype": self.storage_dtype,
                "tensors": fingerprints,
            }
            self.records.append(record)

    def to_dict(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "created_at": utc_now(),
            "metadata": metadata or {},
            "config": {
                "events": sorted(self.events),
                "max_steps": self.max_steps,
                "storage_dtype": self.storage_dtype,
                "variants": self.variant_names,
            },
            "record_count": len(self.records),
            "records": self.records,
        }

    def write_manifest(self, path: Path, metadata: dict[str, Any] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(metadata), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
