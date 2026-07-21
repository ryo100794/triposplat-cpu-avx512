from __future__ import annotations

import ctypes
import time
from pathlib import Path
from typing import Any


class NativeSamplerAVX512Backend:
    def __init__(self, *, library_path: str, threads: int = 2, strict: bool = True):
        import torch

        self.torch = torch
        self.threads = int(threads)
        self.strict = bool(strict)
        self.path = Path(library_path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.lib = ctypes.CDLL(self.path.as_posix())
        self.cfg_kernel = self.lib.triposplat_cfg_combine_inplace_f32_avx512
        self.cfg_kernel.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_int, ctypes.c_int64, ctypes.c_int]
        self.cfg_kernel.restype = ctypes.c_int
        self.euler_kernel = self.lib.triposplat_euler_update_inplace_f32_avx512
        self.euler_kernel.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                      ctypes.c_int, ctypes.c_int64, ctypes.c_int]
        self.euler_kernel.restype = ctypes.c_int
        self.ab2_kernel = self.lib.triposplat_ab2_update_inplace_f32_avx512
        self.ab2_kernel.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_int, ctypes.c_int64, ctypes.c_int]
        self.ab2_kernel.restype = ctypes.c_int
        self.runtime = {
            "cfg": {"calls": 0, "tensors": 0, "elements": 0, "seconds": 0.0, "fallbacks": 0},
            "euler": {"calls": 0, "tensors": 0, "elements": 0, "seconds": 0.0, "fallbacks": 0},
            "ab2": {"calls": 0, "tensors": 0, "elements": 0, "seconds": 0.0, "fallbacks": 0},
        }

    def _valid_vector(self, value, batch: int) -> bool:
        return (
            self.torch.is_tensor(value)
            and value.device.type == "cpu"
            and value.dtype == self.torch.float32
            and value.is_contiguous()
            and value.ndim == 1
            and int(value.numel()) == int(batch)
        )

    def _valid_state_pair(self, left: dict, right: dict, batch: int) -> bool:
        if left.keys() != right.keys():
            return False
        for key, value in left.items():
            other = right[key]
            if not (
                self.torch.is_tensor(value)
                and self.torch.is_tensor(other)
                and value.device.type == "cpu"
                and other.device.type == "cpu"
                and value.dtype == self.torch.float32
                and other.dtype == self.torch.float32
                and value.is_contiguous()
                and other.is_contiguous()
                and value.shape == other.shape
                and value.ndim >= 1
                and int(value.shape[0]) == int(batch)
            ):
                return False
        return True

    def _reject(self, section: str, reason: str) -> bool:
        self.runtime[section]["fallbacks"] += 1
        if self.strict:
            raise RuntimeError(f"native AVX-512 sampler strict violation ({section}): {reason}")
        return False

    def cfg_combine_inplace(self, positive: dict, negative: dict, guidance) -> bool:
        batch = int(guidance.numel()) if self.torch.is_tensor(guidance) else -1
        if not self._valid_vector(guidance, batch):
            return self._reject("cfg", "guidance must be contiguous CPU float32 [batch]")
        if not self._valid_state_pair(positive, negative, batch):
            return self._reject("cfg", "prediction states must be matching contiguous CPU float32 tensors")
        started = time.perf_counter()
        elements = 0
        for key, value in positive.items():
            per_batch = int(value.numel() // batch)
            status = int(self.cfg_kernel(
                value.data_ptr(), negative[key].data_ptr(), guidance.data_ptr(),
                batch, per_batch, self.threads,
            ))
            if status:
                raise RuntimeError(f"native AVX-512 CFG kernel returned {status} for {key}")
            elements += int(value.numel())
        self.runtime["cfg"]["seconds"] += time.perf_counter() - started
        self.runtime["cfg"]["calls"] += 1
        self.runtime["cfg"]["tensors"] += len(positive)
        self.runtime["cfg"]["elements"] += elements
        return True

    def euler_update_inplace(self, sample: dict, prediction: dict, dt) -> bool:
        batch = int(dt.numel()) if self.torch.is_tensor(dt) else -1
        if not self._valid_vector(dt, batch):
            return self._reject("euler", "dt must be contiguous CPU float32 [batch]")
        if not self._valid_state_pair(sample, prediction, batch):
            return self._reject("euler", "state and prediction must be matching contiguous CPU float32 tensors")
        started = time.perf_counter()
        elements = 0
        for key, value in sample.items():
            per_batch = int(value.numel() // batch)
            status = int(self.euler_kernel(
                value.data_ptr(), prediction[key].data_ptr(), dt.data_ptr(),
                batch, per_batch, self.threads,
            ))
            if status:
                raise RuntimeError(f"native AVX-512 Euler kernel returned {status} for {key}")
            elements += int(value.numel())
        self.runtime["euler"]["seconds"] += time.perf_counter() - started
        self.runtime["euler"]["calls"] += 1
        self.runtime["euler"]["tensors"] += len(sample)
        self.runtime["euler"]["elements"] += elements
        return True

    def ab2_update_inplace(self, sample: dict, prediction: dict, previous_prediction: dict, dt, previous_dt) -> bool:
        batch = int(dt.numel()) if self.torch.is_tensor(dt) else -1
        valid = self._valid_vector(dt, batch) and self._valid_vector(previous_dt, batch)
        valid = valid and self._valid_state_pair(sample, prediction, batch)
        valid = valid and self._valid_state_pair(sample, previous_prediction, batch)
        if not valid:
            return self._reject("ab2", "AB2 inputs must be matching contiguous CPU float32 tensors")
        started = time.perf_counter()
        elements = 0
        for key, value in sample.items():
            per_batch = int(value.numel() // batch)
            status = int(self.ab2_kernel(
                value.data_ptr(), prediction[key].data_ptr(), previous_prediction[key].data_ptr(),
                dt.data_ptr(), previous_dt.data_ptr(), batch, per_batch, self.threads,
            ))
            if status:
                raise RuntimeError(f"native AVX-512 AB2 kernel returned {status} for {key}")
            elements += int(value.numel())
        self.runtime["ab2"]["seconds"] += time.perf_counter() - started
        self.runtime["ab2"]["calls"] += 1
        self.runtime["ab2"]["tensors"] += len(sample)
        self.runtime["ab2"]["elements"] += elements
        return True

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "kind": "native_f32_avx512_fused_cfg_euler_ab2_sampler",
            "library_path": self.path.as_posix(),
            "symbols": [
                "triposplat_cfg_combine_inplace_f32_avx512",
                "triposplat_euler_update_inplace_f32_avx512",
                "triposplat_ab2_update_inplace_f32_avx512",
            ],
            "threads": self.threads,
            "strict": self.strict,
            "inplace_cfg": True,
            "inplace_state_update": True,
            "runtime": self.runtime,
        }


def create_triposplat_native_sampler_avx512_backend(
    *, enabled: bool = False,
    library_path: str = "artifacts/backends/libtriposplat_sampler_avx512.so",
    threads: int = 2,
    strict: bool = True,
):
    if not enabled:
        return None, {"enabled": False}
    backend = NativeSamplerAVX512Backend(
        library_path=library_path,
        threads=threads,
        strict=strict,
    )
    return backend, backend.metadata()
