from __future__ import annotations

import re
import time


def apply_triposplat_bf16_output_roundtrip_patch(
    flow_model,
    *,
    enabled: bool = False,
    include_regex: str = r"^blocks[.][0-9]+[.]mlp[.]mlp[.]1$",
    exclude_regex: str | None = None,
):
    if not enabled:
        return {"enabled": False}

    import torch

    include = re.compile(include_regex)
    exclude = re.compile(exclude_regex) if exclude_regex else None
    selected = []
    handles = []
    runtime = {"calls": 0, "elements": 0, "seconds": 0.0, "non_float32_fallbacks": 0}

    def hook(_module, _inputs, output):
        if not torch.is_tensor(output) or output.dtype != torch.float32:
            runtime["non_float32_fallbacks"] += 1
            return output
        started = time.perf_counter()
        rounded = output.to(dtype=torch.bfloat16).to(dtype=torch.float32)
        runtime["seconds"] += time.perf_counter() - started
        runtime["calls"] += 1
        runtime["elements"] += int(output.numel())
        return rounded

    for name, module in flow_model.named_modules():
        if include.search(name) is None or (exclude is not None and exclude.search(name) is not None):
            continue
        handles.append(module.register_forward_hook(hook))
        selected.append(name)
    if not selected:
        raise ValueError(f"BF16 output roundtrip selected no modules with {include_regex!r}")
    flow_model._triposplat_bf16_output_roundtrip_handles = handles
    return {
        "enabled": True,
        "kind": "bounded_approx_bf16_module_output_roundtrip_probe",
        "include_regex": include_regex,
        "exclude_regex": exclude_regex,
        "selected_count": len(selected),
        "selected": selected,
        "runtime": runtime,
        "semantics": "Rounds selected float32 module outputs through BF16; normalization, residual state, softmax, and sampler updates remain float32.",
    }
