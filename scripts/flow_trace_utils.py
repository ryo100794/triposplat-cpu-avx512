#!/usr/bin/env python3
"""Small tensor trace helpers for TripoSplat flow equivalence checks."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sample_tensor(x: torch.Tensor, max_values: int) -> torch.Tensor:
    flat = x.detach().reshape(-1)
    if flat.numel() <= max_values:
        return flat
    # Fixed index sampling is deterministic across CPU/GPU and avoids storing
    # full activations. The indices are generated on CPU to avoid device noise.
    idx = torch.linspace(0, flat.numel() - 1, steps=max_values, dtype=torch.long)
    return flat.cpu()[idx]


def tensor_trace(
    x: torch.Tensor,
    max_values: int = 256,
    stats_sample_values: int = 4096,
    store_values: bool = False,
) -> dict[str, Any]:
    sample = _sample_tensor(x, int(max_values)).to(dtype=torch.float32, device="cpu").contiguous()
    stat_sample = _sample_tensor(x, int(stats_sample_values)).to(dtype=torch.float32, device="cpu")
    finite = torch.isfinite(stat_sample)
    if finite.any():
        finite_values = stat_sample[finite]
        mean = float(finite_values.mean().item())
        std = float(finite_values.std(unbiased=False).item())
        min_value = float(finite_values.min().item())
        max_value = float(finite_values.max().item())
    else:
        mean = std = min_value = max_value = None
    out = {
        "shape": list(x.shape),
        "dtype": str(x.dtype).replace("torch.", ""),
        "device": str(x.device),
        "numel": int(x.numel()),
        "sample_count": int(sample.numel()),
        "sample_float32_sha256": hashlib.sha256(sample.numpy().tobytes()).hexdigest(),
        "stats_sample_count": int(stat_sample.numel()),
        "finite_count": int(finite.sum().item()),
        "mean": mean,
        "std": std,
        "min": min_value,
        "max": max_value,
    }
    if store_values:
        out["sample_values_float32"] = [float(v) for v in sample.tolist()]
    return out


def state_trace(
    values: dict[str, torch.Tensor],
    max_values: int = 256,
    stats_sample_values: int = 4096,
    store_values: bool = False,
) -> dict[str, Any]:
    return {
        key: tensor_trace(
            value,
            max_values=max_values,
            stats_sample_values=stats_sample_values,
            store_values=store_values,
        )
        for key, value in sorted(values.items())
    }


class FlowTraceRecorder:
    def __init__(
        self,
        *,
        max_steps: int = 2,
        max_values: int = 256,
        stats_sample_values: int = 4096,
        events: set[str] | None = None,
        store_values: bool = False,
    ):
        self.max_steps = int(max_steps)
        self.max_values = int(max_values)
        self.stats_sample_values = int(stats_sample_values)
        self.events = set(events or [])
        self.store_values = bool(store_values)
        self.records: list[dict[str, Any]] = []

    def callback(self, event: str, payload: dict[str, Any]) -> None:
        step = int(payload.get("step", 0))
        if self.max_steps > 0 and step > self.max_steps:
            return
        if self.events and event not in self.events:
            return
        record = {
            "event": event,
            "step": step,
            "created_at": utc_now(),
        }
        for key, value in payload.items():
            if isinstance(value, torch.Tensor):
                record[key] = tensor_trace(
                    value,
                    max_values=self.max_values,
                    stats_sample_values=self.stats_sample_values,
                    store_values=self.store_values,
                )
            elif isinstance(value, dict) and value and all(isinstance(v, torch.Tensor) for v in value.values()):
                record[key] = state_trace(
                    value,
                    max_values=self.max_values,
                    stats_sample_values=self.stats_sample_values,
                    store_values=self.store_values,
                )
            elif key != "step":
                record[key] = value
        self.records.append(record)

    def to_dict(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "created_at": utc_now(),
            "metadata": metadata or {},
            "config": {
                "max_steps": self.max_steps,
                "max_values": self.max_values,
                "stats_sample_values": self.stats_sample_values,
                "events": sorted(self.events),
                "store_values": self.store_values,
            },
            "record_count": len(self.records),
            "records": self.records,
        }

    def write_json(self, path: Path, metadata: dict[str, Any] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(metadata), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
