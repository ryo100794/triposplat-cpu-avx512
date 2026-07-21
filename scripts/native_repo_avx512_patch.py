from __future__ import annotations

import ctypes
import time
import types
from pathlib import Path
from typing import Any


def apply_triposplat_native_repo_avx512_patch(
    flow_model,
    *,
    enabled: bool = False,
    library_path: str = "artifacts/backends/libtriposplat_repo_avx512.so",
    threads: int = 2,
    strict: bool = True,
) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False}

    import torch

    path = Path(library_path)
    if not path.exists():
        raise FileNotFoundError(path)
    lib = ctypes.CDLL(path.as_posix())
    multiply = lib.triposplat_mul_inplace_f32_avx512
    multiply.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int]
    multiply.restype = ctypes.c_int
    phasor = lib.triposplat_repo_phasor_f32_avx512
    phasor.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int64, ctypes.c_int,
    ]
    phasor.restype = ctypes.c_int

    runtime = {
        "feature_multiply": {"calls": 0, "elements": 0, "seconds": 0.0, "fallbacks": 0},
        "phasor": {"calls": 0, "vectors": 0, "complex_values": 0, "seconds": 0.0, "fallbacks": 0},
    }
    selected = []

    def valid_f32(tensor):
        return (
            torch.is_tensor(tensor)
            and tensor.device.type == "cpu"
            and tensor.dtype == torch.float32
            and tensor.is_contiguous()
        )

    def make_forward(name):
        def forward(self, hidden_states):
            h = self.norm(hidden_states)
            gate = self.act(self.gate_map(h))
            content = self.content_map(h)
            if not (valid_f32(gate) and valid_f32(content) and gate.shape == content.shape):
                runtime["feature_multiply"]["fallbacks"] += 1
                if strict:
                    raise RuntimeError(f"native RePo feature multiply strict violation: {name}")
                return self._original_forward_native_repo_avx512(hidden_states)
            started = time.perf_counter()
            status = int(multiply(gate.data_ptr(), content.data_ptr(), gate.numel(), int(threads)))
            runtime["feature_multiply"]["seconds"] += time.perf_counter() - started
            if status:
                raise RuntimeError(f"native RePo feature multiply returned {status}: {name}")
            runtime["feature_multiply"]["calls"] += 1
            runtime["feature_multiply"]["elements"] += int(gate.numel())

            delta = self.final_map(gate)
            if not valid_f32(delta) or delta.ndim != 3 or int(delta.shape[-1]) != 3 * int(self.num_heads):
                runtime["phasor"]["fallbacks"] += 1
                if strict:
                    raise RuntimeError(f"native RePo phasor strict violation: {name}")
                return self._original_forward_native_repo_avx512(hidden_states)
            batch, length = int(delta.shape[0]), int(delta.shape[1])
            dims = self._native_repo_freq_dims
            total_dim = sum(dims)
            output = torch.empty(
                (batch, length, int(self.num_heads), total_dim),
                dtype=torch.complex64,
                device=delta.device,
            )
            freq_parts = self._native_repo_freq_parts
            vectors = batch * length * int(self.num_heads)
            started = time.perf_counter()
            status = int(phasor(
                delta.data_ptr(),
                freq_parts[0][0].data_ptr(), freq_parts[0][1].data_ptr(), dims[0],
                freq_parts[1][0].data_ptr(), freq_parts[1][1].data_ptr(), dims[1],
                freq_parts[2][0].data_ptr(), freq_parts[2][1].data_ptr(), dims[2],
                output.data_ptr(), vectors, int(threads),
            ))
            runtime["phasor"]["seconds"] += time.perf_counter() - started
            if status:
                raise RuntimeError(f"native RePo phasor returned {status}: {name}")
            runtime["phasor"]["calls"] += 1
            runtime["phasor"]["vectors"] += vectors
            runtime["phasor"]["complex_values"] += int(output.numel())
            return output

        return forward

    for name, module in flow_model.named_modules():
        if module.__class__.__name__ != "RePo3DRotaryEmbedding":
            continue
        frequency_tensors = [module.freqs_0, module.freqs_1, module.freqs_2]
        parts = []
        dims = []
        for frequency in frequency_tensors:
            f = frequency.detach().to(device="cpu", dtype=torch.float32).contiguous()
            ft = f.tanh().contiguous()
            residual = (f - ft).contiguous()
            parts.append((ft, residual))
            dims.append(int(f.numel()))
        module._native_repo_freq_parts = parts
        module._native_repo_freq_dims = dims
        module._original_forward_native_repo_avx512 = module.forward
        module.forward = types.MethodType(make_forward(name), module)
        selected.append(name)

    return {
        "enabled": True,
        "kind": "native_f32_avx512_repo_patch",
        "library_path": path.as_posix(),
        "symbols": ["triposplat_mul_inplace_f32_avx512", "triposplat_repo_phasor_f32_avx512"],
        "threads": int(threads),
        "strict": bool(strict),
        "selected_count": len(selected),
        "selected": selected,
        "runtime": runtime,
    }
