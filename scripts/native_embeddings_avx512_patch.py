from __future__ import annotations

import ctypes
import time
import types
from pathlib import Path
from typing import Any


def apply_triposplat_native_embeddings_avx512_patch(
    flow_model,
    *,
    enabled: bool = False,
    library_path: str = "artifacts/backends/libtriposplat_embeddings_avx512.so",
    threads: int = 2,
    strict: bool = True,
) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False}

    import numpy as np
    import torch

    path = Path(library_path)
    if not path.exists():
        raise FileNotFoundError(path)
    lib = ctypes.CDLL(path.as_posix())
    position_kernel = lib.triposplat_pcd_position_embedding_f32_avx512
    position_kernel.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    position_kernel.restype = ctypes.c_int
    timestep_kernel = lib.triposplat_timestep_embedding_f32_avx512
    timestep_kernel.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    timestep_kernel.restype = ctypes.c_int

    runtime = {
        "position": {"calls": 0, "rows": 0, "elements": 0, "seconds": 0.0, "fallbacks": 0},
        "timestep": {"calls": 0, "samples": 0, "elements": 0, "seconds": 0.0, "fallbacks": 0},
    }
    position_selected = []
    timestep_selected = []

    def valid_f32(tensor):
        return (
            torch.is_tensor(tensor)
            and tensor.device.type == "cpu"
            and tensor.dtype == torch.float32
            and tensor.is_contiguous()
        )

    def make_position_forward(name):
        def forward(self, x):
            if not valid_f32(x) or x.ndim < 2 or int(x.shape[-1]) != int(self.in_channels):
                runtime["position"]["fallbacks"] += 1
                if strict:
                    raise RuntimeError(f"native position embedding strict violation: {name}")
                return self._original_forward_native_embeddings(x)
            rows = int(x.numel() // int(self.in_channels))
            output = torch.empty((*x.shape[:-1], int(self.channels)), dtype=torch.float32, device=x.device)
            started = time.perf_counter()
            status = int(position_kernel(
                x.data_ptr(), self._native_position_frequencies.data_ptr(), output.data_ptr(),
                rows, int(self.in_channels), int(self.freq_dim), int(self.channels), int(self._native_position_double_pi), int(threads),
            ))
            runtime["position"]["seconds"] += time.perf_counter() - started
            if status:
                raise RuntimeError(f"native position embedding returned {status}: {name}")
            runtime["position"]["calls"] += 1
            runtime["position"]["rows"] += rows
            runtime["position"]["elements"] += int(output.numel())
            return output

        return forward

    def make_timestep_forward(name):
        def forward(self, t):
            source = t.float().contiguous()
            if not valid_f32(source) or source.ndim != 1:
                runtime["timestep"]["fallbacks"] += 1
                if strict:
                    raise RuntimeError(f"native timestep embedding strict violation: {name}")
                return self._original_forward_native_embeddings(t)
            dim = int(self.frequency_embedding_size)
            half = dim // 2
            embedding = torch.empty((int(source.shape[0]), dim), dtype=torch.float32, device=source.device)
            started = time.perf_counter()
            status = int(timestep_kernel(
                source.data_ptr(), self._native_timestep_frequencies.data_ptr(), embedding.data_ptr(),
                int(source.shape[0]), half, dim, int(threads),
            ))
            runtime["timestep"]["seconds"] += time.perf_counter() - started
            if status:
                raise RuntimeError(f"native timestep embedding returned {status}: {name}")
            runtime["timestep"]["calls"] += 1
            runtime["timestep"]["samples"] += int(source.shape[0])
            runtime["timestep"]["elements"] += int(embedding.numel())
            return self.mlp(embedding)

        return forward

    for name, module in flow_model.named_modules():
        class_name = module.__class__.__name__
        if class_name in {"PcdAbsolutePositionEmbedder", "PcdAbsolutePositionEmbedderV2"}:
            module._native_position_frequencies = module._freqs(torch.device("cpu")).detach().float().contiguous()
            module._native_position_double_pi = class_name == "PcdAbsolutePositionEmbedder"
            module._original_forward_native_embeddings = module.forward
            module.forward = types.MethodType(make_position_forward(name), module)
            position_selected.append(name)
        elif class_name == "TimestepEmbedder":
            dim = int(module.frequency_embedding_size)
            half = dim // 2
            module._native_timestep_frequencies = torch.exp(
                -float(np.log(10000)) * torch.arange(0, half, dtype=torch.float32) / half
            ).contiguous()
            module._original_forward_native_embeddings = module.forward
            module.forward = types.MethodType(make_timestep_forward(name), module)
            timestep_selected.append(name)

    if strict and (len(position_selected) != 1 or len(timestep_selected) != 1):
        raise RuntimeError(
            f"native embedding selection mismatch: position={position_selected} timestep={timestep_selected}"
        )
    return {
        "enabled": True,
        "kind": "native_f32_avx512_position_timestep_embeddings_patch",
        "library_path": path.as_posix(),
        "symbols": [
            "triposplat_pcd_position_embedding_f32_avx512",
            "triposplat_timestep_embedding_f32_avx512",
        ],
        "threads": int(threads),
        "strict": bool(strict),
        "position_selected": position_selected,
        "timestep_selected": timestep_selected,
        "runtime": runtime,
    }
