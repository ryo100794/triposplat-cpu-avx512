#!/usr/bin/env python3
"""Module forward trace hooks for TripoSplat flow debugging."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from flow_trace_utils import state_trace, tensor_trace


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _shape_summary(value: Any) -> dict[str, Any] | None:
    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
            "device": str(value.device),
            "numel": int(value.numel()),
        }
    if isinstance(value, dict):
        out = {str(key): _shape_summary(item) for key, item in value.items()}
        return {key: item for key, item in out.items() if item is not None} or None
    if isinstance(value, (tuple, list)):
        out = {str(index): _shape_summary(item) for index, item in enumerate(value)}
        return {key: item for key, item in out.items() if item is not None} or None
    if value is None:
        return {"kind": "none"}
    return {"kind": type(value).__name__}


def _trace_value(
    value: Any,
    *,
    max_values: int,
    stats_sample_values: int,
    store_values: bool,
) -> dict[str, Any] | None:
    if isinstance(value, torch.Tensor):
        return tensor_trace(
            value,
            max_values=max_values,
            stats_sample_values=stats_sample_values,
            store_values=store_values,
        )
    if isinstance(value, dict) and value and all(isinstance(v, torch.Tensor) for v in value.values()):
        return state_trace(
            value,
            max_values=max_values,
            stats_sample_values=stats_sample_values,
            store_values=store_values,
        )
    if isinstance(value, (tuple, list)):
        out = {}
        for index, item in enumerate(value):
            traced = _trace_value(
                item,
                max_values=max_values,
                stats_sample_values=stats_sample_values,
                store_values=store_values,
            )
            if traced is not None:
                out[str(index)] = traced
        return out or None
    return None


class ModuleTraceRecorder:
    def __init__(
        self,
        *,
        include_regex: str,
        exclude_regex: str | None = None,
        max_modules: int = 64,
        max_calls_per_module: int = 1,
        max_values: int = 128,
        stats_sample_values: int = 1024,
        store_values: bool = False,
        trace_inputs: bool = False,
    ):
        self.include_regex = include_regex
        self.exclude_regex = exclude_regex
        self.max_modules = int(max_modules)
        self.max_calls_per_module = int(max_calls_per_module)
        self.max_values = int(max_values)
        self.stats_sample_values = int(stats_sample_values)
        self.store_values = bool(store_values)
        self.trace_inputs = bool(trace_inputs)
        self.records: list[dict[str, Any]] = []
        self.selected: list[str] = []
        self._handles = []
        self._call_counts: dict[str, int] = {}
        self._start_times: dict[str, list[float]] = {}
        self._global_call_index = 0

    def install(self, model: torch.nn.Module) -> dict[str, Any]:
        include = re.compile(self.include_regex)
        exclude = re.compile(self.exclude_regex) if self.exclude_regex else None
        for name, module in model.named_modules():
            if not name:
                continue
            if not include.search(name):
                continue
            if exclude is not None and exclude.search(name):
                continue
            if len(self.selected) >= self.max_modules:
                break
            self.selected.append(name)
            self._handles.append(module.register_forward_pre_hook(self._make_pre_hook(name)))
            self._handles.append(module.register_forward_hook(self._make_hook(name)))
        return {
            "enabled": True,
            "kind": "module_forward_trace",
            "include_regex": self.include_regex,
            "exclude_regex": self.exclude_regex,
            "max_modules": self.max_modules,
            "max_calls_per_module": self.max_calls_per_module,
            "max_values": self.max_values,
            "stats_sample_values": self.stats_sample_values,
            "store_values": self.store_values,
            "trace_inputs": self.trace_inputs,
            "selected_count": len(self.selected),
            "selected": self.selected,
        }

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_pre_hook(self, name: str):
        def pre_hook(module, inputs):
            self._start_times.setdefault(name, []).append(time.perf_counter())

        return pre_hook

    def _make_hook(self, name: str):
        def hook(module, inputs, output):
            stack = self._start_times.get(name)
            started_at = stack.pop() if stack else None
            count = self._call_counts.get(name, 0)
            self._call_counts[name] = count + 1
            if self.max_calls_per_module > 0 and count >= self.max_calls_per_module:
                return
            call_index = self._global_call_index
            self._global_call_index += 1
            elapsed_sec = None if started_at is None else time.perf_counter() - started_at
            record: dict[str, Any] = {
                "event": f"module:{name}",
                "step": int(call_index),
                "created_at": utc_now(),
                "module": name,
                "module_class": module.__class__.__name__,
                "module_call_index": int(count),
                "elapsed_sec": elapsed_sec,
                "input_summary": _shape_summary(inputs),
                "output_summary": _shape_summary(output),
            }
            if self.trace_inputs:
                traced_inputs = _trace_value(
                    inputs,
                    max_values=self.max_values,
                    stats_sample_values=self.stats_sample_values,
                    store_values=self.store_values,
                )
                if traced_inputs is not None:
                    record["input"] = traced_inputs
            traced_output = _trace_value(
                output,
                max_values=self.max_values,
                stats_sample_values=self.stats_sample_values,
                store_values=self.store_values,
            )
            if traced_output is not None:
                record["output"] = traced_output
            self.records.append(record)

        return hook

    def to_dict(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "created_at": utc_now(),
            "metadata": metadata or {},
            "config": {
                "include_regex": self.include_regex,
                "exclude_regex": self.exclude_regex,
                "max_modules": self.max_modules,
                "max_calls_per_module": self.max_calls_per_module,
                "max_values": self.max_values,
                "stats_sample_values": self.stats_sample_values,
                "store_values": self.store_values,
                "trace_inputs": self.trace_inputs,
                "selected": self.selected,
                "records_include_elapsed_sec": True,
                "records_include_shape_summary": True,
            },
            "record_count": len(self.records),
            "records": self.records,
        }

    def write_json(self, path: Path, metadata: dict[str, Any] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(metadata), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
