from __future__ import annotations
import ctypes,time,types
from pathlib import Path
from typing import Any

def apply_triposplat_native_block_elementwise_avx512_patch(flow_model,*,enabled:bool=False,
    library_path:str="artifacts/backends/libtriposplat_block_elementwise_avx512.so",threads:int=2,strict:bool=True)->dict[str,Any]:
    if not enabled:return {"enabled":False}
    import torch
    import triposplat_attention_patch as tap
    path=Path(library_path)
    if not path.exists():raise FileNotFoundError(path)
    lib=ctypes.CDLL(path.as_posix());mod=lib.triposplat_modulate_inplace_f32_avx512;res=lib.triposplat_gated_residual_inplace_f32_avx512;add=lib.triposplat_add_inplace_f32_avx512
    mod.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*4;res.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*4;add.argtypes=[ctypes.c_void_p]*2+[ctypes.c_int64,ctypes.c_int]
    for fn in(mod,res,add):fn.restype=ctypes.c_int
    runtime={"modulate":{"calls":0,"elements":0,"seconds":0.0,"fallbacks":0},"residual":{"calls":0,"elements":0,"seconds":0.0,"fallbacks":0},"add":{"calls":0,"elements":0,"seconds":0.0,"fallbacks":0}}
    def valid(t):return torch.is_tensor(t) and t.device.type=="cpu" and t.dtype==torch.float32 and t.is_contiguous()
    def modulate(h,scale,shift):
        if not(valid(h) and valid(scale) and valid(shift)):
            runtime["modulate"]["fallbacks"]+=1
            if strict:raise RuntimeError("native modulate strict violation")
            return tap._native_original_modulate(h,scale,shift)
        b,r,c=int(h.shape[0]),int(h.shape[-2]),int(h.shape[-1]);started=time.perf_counter();status=int(mod(h.data_ptr(),scale.data_ptr(),shift.data_ptr(),b,r,c,int(threads)));elapsed=time.perf_counter()-started
        if status:raise RuntimeError(f"native modulate returned {status}")
        runtime["modulate"]["calls"]+=1;runtime["modulate"]["elements"]+=h.numel();runtime["modulate"]["seconds"]+=elapsed;return h
    def residual(x,h,gate):
        if not(valid(x) and valid(h) and valid(gate)):
            runtime["residual"]["fallbacks"]+=1
            if strict:raise RuntimeError("native residual strict violation")
            return tap._native_original_residual(x,h,gate)
        b,r,c=int(x.shape[0]),int(x.shape[-2]),int(x.shape[-1]);started=time.perf_counter();status=int(res(x.data_ptr(),h.data_ptr(),gate.data_ptr(),b,r,c,int(threads)));elapsed=time.perf_counter()-started
        if status:raise RuntimeError(f"native residual returned {status}")
        runtime["residual"]["calls"]+=1;runtime["residual"]["elements"]+=x.numel();runtime["residual"]["seconds"]+=elapsed;return x
    def add_inplace(x,h):
        if not(valid(x) and valid(h) and x.shape == h.shape):
            runtime["add"]["fallbacks"]+=1
            if strict:raise RuntimeError("native add strict violation")
            x.add_(h);return x
        started=time.perf_counter();status=int(add(x.data_ptr(),h.data_ptr(),x.numel(),int(threads)));elapsed=time.perf_counter()-started
        if status:raise RuntimeError(f"native add returned {status}")
        runtime["add"]["calls"]+=1;runtime["add"]["elements"]+=x.numel();runtime["add"]["seconds"]+=elapsed;return x
    tap._native_original_modulate=tap._modulate_inplace_preserve_order;tap._native_original_residual=tap._residual_gate_inplace_preserve_order
    tap._modulate_inplace_preserve_order=modulate;tap._residual_gate_inplace_preserve_order=residual
    selected=[]
    def make_forward(name):
        def forward(self,x,mod=None,rotary_emb=None):
            if self.modulation:
                if not self.share_mod:mod=self.adaLN_modulation(mod)
                if hasattr(self,"shift_table") and self.shift_table is not None:
                    mod=mod.contiguous().clone();add_inplace(mod,self.shift_table.type(mod.dtype).contiguous())
                shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp=mod.chunk(6,dim=1)
                h=modulate(self.norm1(x),scale_msa,shift_msa);h=self.attn(h,rope_emb=rotary_emb);x=residual(x,h,gate_msa)
                h=modulate(self.norm2(x),scale_mlp,shift_mlp);x=residual(x,self.mlp(h),gate_mlp);return x
            x=add_inplace(x,self.attn(self.norm1(x),rope_emb=rotary_emb));return add_inplace(x,self.mlp(self.norm2(x)))
        return forward
    for name,module in flow_model.named_modules():
        if module.__class__.__name__!="UnifiedTransformerBlock":continue
        module._native_avx512_add_inplace=add_inplace;module._native_avx512_modulate_inplace=modulate
        module._original_forward_native_block_elementwise=module.forward;module.forward=types.MethodType(make_forward(name),module);selected.append(name)
    flow_model._native_avx512_add_inplace=add_inplace;flow_model._native_avx512_modulate_inplace=modulate
    return {"enabled":True,"kind":"native_f32_avx512_block_elementwise_patch","library_path":path.as_posix(),"threads":int(threads),"strict":bool(strict),"selected_count":len(selected),"selected":selected,"runtime":runtime}
