from __future__ import annotations
import ctypes, time, types
from pathlib import Path
from typing import Any

def apply_triposplat_native_silu_avx512_patch(flow_model, *, enabled: bool=False,
    library_path: str="artifacts/backends/libtriposplat_activations_avx512.so",
    threads: int=2, strict: bool=True) -> dict[str, Any]:
    if not enabled: return {"enabled": False}
    import torch
    import torch.nn as nn
    path=Path(library_path)
    if not path.exists(): raise FileNotFoundError(f"native AVX-512 SiLU library not found: {path}")
    lib=ctypes.CDLL(path.as_posix()); kernel=lib.triposplat_silu_f32_avx512
    kernel.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64,ctypes.c_int]; kernel.restype=ctypes.c_int
    runtime={"calls":0,"elements":0,"seconds":0.0,"fallbacks":0,"per_module":{}}
    selected=[]
    def make_forward(name):
        def forward(self,x):
            reason=None
            if torch.is_grad_enabled(): reason="grad_enabled"
            elif x.device.type!="cpu" or x.dtype!=torch.float32: reason=f"input_{x.device.type}_{x.dtype}"
            if reason is not None:
                runtime["fallbacks"]+=1
                if strict: raise RuntimeError(f"native AVX-512 SiLU strict violation for {name}: {reason}")
                return self._original_forward_native_silu_avx512(x)
            source=x if x.is_contiguous() else x.contiguous(); out=torch.empty_like(source); started=time.perf_counter()
            status=int(kernel(source.data_ptr(),out.data_ptr(),source.numel(),int(threads))); elapsed=time.perf_counter()-started
            if status: raise RuntimeError(f"native AVX-512 SiLU returned {status} for {name}")
            runtime["calls"]+=1; runtime["elements"]+=int(source.numel()); runtime["seconds"]+=elapsed
            item=runtime["per_module"].setdefault(name,{"calls":0,"elements":0,"seconds":0.0}); item["calls"]+=1; item["elements"]+=int(source.numel()); item["seconds"]+=elapsed
            return out.view_as(x)
        return forward
    for name,module in flow_model.named_modules():
        if not isinstance(module,nn.SiLU): continue
        module._original_forward_native_silu_avx512=module.forward; module.forward=types.MethodType(make_forward(name),module); selected.append(name)
    return {"enabled":bool(selected),"kind":"native_f32_avx512_silu_patch","library_path":path.as_posix(),
        "symbol":"triposplat_silu_f32_avx512","threads":int(threads),"strict":bool(strict),"selected_count":len(selected),"selected":selected,"runtime":runtime}
