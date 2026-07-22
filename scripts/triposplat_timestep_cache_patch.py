from __future__ import annotations

import time
import types
from collections import OrderedDict


def apply_triposplat_timestep_modulation_cache(
    flow_model,
    *,
    enabled: bool = False,
    max_entries: int = 64,
):
    if not enabled:
        return {"enabled": False}
    if max_entries <= 0:
        raise ValueError("timestep cache max_entries must be positive")
    if not hasattr(flow_model, "t_embedder") or not hasattr(flow_model, "adaLN_modulation"):
        raise ValueError("flow model has no timestep embedder/AdaLN modulation")

    import torch

    aggregate = {
        "hits": 0,
        "misses": 0,
        "invalidated_outputs": 0,
        "evictions": 0,
        "reused_bytes": 0,
        "compute_seconds": 0.0,
        "lookup_seconds": 0.0,
    }
    module_stats = {}

    def tensor_version(value):
        try:
            return int(value._version)
        except RuntimeError:
            return None

    def install(module, label: str, key_function):
        original_attr = f"_original_forward_{label}_exact_cache"
        if not hasattr(module, original_attr):
            setattr(module, original_attr, module.forward)
        original = getattr(module, original_attr)
        cache = OrderedDict()
        stats = {
            "hits": 0,
            "misses": 0,
            "invalidated_outputs": 0,
            "evictions": 0,
            "reused_bytes": 0,
            "compute_seconds": 0.0,
            "lookup_seconds": 0.0,
            "entries": 0,
        }
        module_stats[label] = stats

        def add_stat(name: str, value):
            stats[name] += value
            aggregate[name] += value

        def patched_forward(self, x, *args, **kwargs):
            lookup_started = time.perf_counter()
            key = key_function(x, args, kwargs)
            entry = cache.get(key)
            if entry is not None:
                output, output_version = entry
                if tensor_version(output) == output_version:
                    cache.move_to_end(key)
                    add_stat("hits", 1)
                    add_stat("reused_bytes", output.numel() * output.element_size())
                    add_stat("lookup_seconds", time.perf_counter() - lookup_started)
                    return output
                add_stat("invalidated_outputs", 1)
                del cache[key]
            add_stat("misses", 1)
            add_stat("lookup_seconds", time.perf_counter() - lookup_started)
            compute_started = time.perf_counter()
            output = original(x, *args, **kwargs)
            add_stat("compute_seconds", time.perf_counter() - compute_started)
            if not torch.is_tensor(output):
                raise TypeError(f"{label} cache requires tensor output")
            cache[key] = (output, tensor_version(output))
            cache.move_to_end(key)
            while len(cache) > max_entries:
                cache.popitem(last=False)
                add_stat("evictions", 1)
            stats["entries"] = len(cache)
            return output

        module.forward = types.MethodType(patched_forward, module)
        return cache

    def timestep_key(x, args, kwargs):
        if args or kwargs:
            raise ValueError("timestep cache only supports the standard single-tensor call")
        if not torch.is_tensor(x) or x.device.type != "cpu" or not x.is_floating_point():
            raise ValueError("timestep cache requires a floating CPU tensor")
        values = tuple(float(value) for value in x.detach().reshape(-1).tolist())
        return (
            tuple(int(value) for value in x.shape),
            str(x.dtype).replace("torch.", ""),
            str(x.device),
            values,
        )

    timestep_cache = install(flow_model.t_embedder, "timestep_embedding", timestep_key)

    def modulation_key(x, args, kwargs):
        if args or kwargs:
            raise ValueError("modulation cache only supports the standard single-tensor call")
        if not torch.is_tensor(x):
            raise ValueError("modulation cache requires a tensor")
        return (
            id(x),
            tensor_version(x),
            tuple(int(value) for value in x.shape),
            str(x.dtype).replace("torch.", ""),
            str(x.device),
        )

    modulation_cache = install(flow_model.adaLN_modulation, "adaln_modulation", modulation_key)

    def clear():
        timestep_cache.clear()
        modulation_cache.clear()
        for stats in module_stats.values():
            stats["entries"] = 0

    flow_model._triposplat_clear_timestep_modulation_cache = clear

    return {
        "enabled": True,
        "kind": "exact_timestep_embedding_and_adaln_modulation_lru_cache",
        "max_entries": int(max_entries),
        "stats": aggregate,
        "module_stats": module_stats,
        "key": "timestep values, shape, dtype, device; AdaLN key uses the cached timestep output identity/version",
        "mutation_guard": "cached output _version must remain unchanged when available; inference tensors are treated as immutable",
        "semantics": "Exact for immutable inference weights and repeated schedules; state/condition-dependent tensors are not cached.",
    }


def _self_test() -> None:
    import torch

    class ToyFlow(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.t_embedder = torch.nn.Sequential(torch.nn.Linear(1, 8), torch.nn.SiLU())
            self.adaLN_modulation = torch.nn.Linear(8, 16)

    torch.manual_seed(7)
    model = ToyFlow().eval()
    metadata = apply_triposplat_timestep_modulation_cache(model, enabled=True, max_entries=2)
    t1 = torch.tensor([[1000.0]], dtype=torch.float32)
    t2 = torch.tensor([[500.0]], dtype=torch.float32)
    e1 = model.t_embedder(t1)
    m1 = model.adaLN_modulation(e1)
    assert model.t_embedder(t1.clone()).data_ptr() == e1.data_ptr()
    assert model.adaLN_modulation(e1).data_ptr() == m1.data_ptr()
    assert model.t_embedder(t2).data_ptr() != e1.data_ptr()
    e1.add_(1.0)
    assert model.t_embedder(t1).data_ptr() != e1.data_ptr()
    assert metadata["stats"]["hits"] == 2
    assert metadata["stats"]["misses"] == 4
    assert metadata["stats"]["invalidated_outputs"] == 1
    model._triposplat_clear_timestep_modulation_cache()
    assert all(item["entries"] == 0 for item in metadata["module_stats"].values())


if __name__ == "__main__":
    _self_test()
    print("triposplat_timestep_cache_patch self-test passed")
