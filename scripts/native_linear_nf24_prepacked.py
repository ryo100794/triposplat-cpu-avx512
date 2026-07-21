from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable


FORMAT = "triposplat_nf24_i16_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _linear_filename(index: int, name: str) -> str:
    suffix = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    return f"linears/{index:03d}_{suffix}.safetensors"


def _entry_is_valid(root: Path, item: dict[str, Any] | None) -> bool:
    if not item or "file" not in item or "sha256" not in item:
        return False
    path = root / item["file"]
    return path.is_file() and path.stat().st_size == item.get("bytes") and _sha256(path) == item["sha256"]


def pack_nf24_i16_checkpoint(
    source_checkpoint: str | Path,
    output_dir: str | Path,
    *,
    expected_linear_count: int = 206,
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    from native_linear_nf8_avx512_patch import make_nf8_codebook
    from native_linear_rnf8_avx512_patch import quantize_nf24_i16_per_output_channel

    source = Path(source_checkpoint).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "linears").mkdir(exist_ok=True)
    manifest_path = destination / "manifest.json"
    codebook = make_nf8_codebook(torch)
    source_info = {
        "name": source.name,
        "bytes": source.stat().st_size,
        "sha256": _sha256(source),
    }
    existing_manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    )

    with safe_open(source.as_posix(), framework="pt", device="cpu") as handle:
        keys = sorted(handle.keys())
        linear_weight_keys = [
            key for key in keys
            if key.endswith(".weight") and len(handle.get_slice(key).get_shape()) == 2
        ]
        if len(linear_weight_keys) != expected_linear_count:
            raise RuntimeError(
                f"expected {expected_linear_count} Linear weights, found {len(linear_weight_keys)}"
            )
        linear_names = [key[:-7] for key in linear_weight_keys]
        linear_state_keys = set(linear_weight_keys)
        linear_state_keys.update(
            f"{name}.bias" for name in linear_names if f"{name}.bias" in keys
        )

        if existing_manifest is not None:
            if (
                existing_manifest.get("format") != FORMAT
                or existing_manifest.get("source") != source_info
                or existing_manifest.get("linear_count") != len(linear_names)
            ):
                raise RuntimeError("existing prepacked manifest does not match source checkpoint")
            manifest = existing_manifest
        else:
            manifest = {
                "format": FORMAT,
                "complete": False,
                "source": source_info,
                "residual_mode": "nf24_i16",
                "bits_per_weight": 24,
                "linear_count": len(linear_names),
                "linears": {},
            }

        complete_linears = (
            len(manifest["linears"]) == len(linear_names)
            and all(_entry_is_valid(destination, item) for item in manifest["linears"].values())
        )
        if manifest.get("complete") and complete_linears and _entry_is_valid(destination, manifest.get("non_linear")):
            return manifest
        manifest["complete"] = False

        for index, (name, weight_key) in enumerate(zip(linear_names, linear_weight_keys)):
            relative = _linear_filename(index, name)
            target = destination / relative
            if _entry_is_valid(destination, manifest["linears"].get(name)):
                continue
            weight = handle.get_tensor(weight_key).to(torch.float32).contiguous()
            codes, scales, error = quantize_nf24_i16_per_output_channel(weight, codebook)
            bias_key = f"{name}.bias"
            bias = (
                handle.get_tensor(bias_key).to(torch.float32).contiguous()
                if bias_key in keys
                else torch.zeros(weight.shape[0], dtype=torch.float32)
            )
            temporary = target.with_suffix(target.suffix + ".tmp")
            save_file(
                {
                    "code01_t": codes[0],
                    "q2_t": codes[1],
                    "scale": scales[0],
                    "bias": bias,
                },
                temporary.as_posix(),
            )
            os.replace(temporary, target)
            manifest["linears"][name] = {
                "file": relative,
                "sha256": _sha256(target),
                "bytes": target.stat().st_size,
                "in_features": int(weight.shape[1]),
                "out_features": int(weight.shape[0]),
                "has_bias": bias_key in keys,
                "weight_elements": int(weight.numel()),
                "quantization": error,
            }
            _write_json(manifest_path, manifest)
            del weight, codes, scales, bias

        non_linear = None
        if not _entry_is_valid(destination, manifest.get("non_linear")):
            non_linear = {
                key: handle.get_tensor(key).to(torch.float32).contiguous()
                for key in keys
                if key not in linear_state_keys
            }

    if non_linear is not None:
        non_linear_path = destination / "non_linear.safetensors"
        temporary = non_linear_path.with_suffix(non_linear_path.suffix + ".tmp")
        save_file(non_linear, temporary.as_posix())
        os.replace(temporary, non_linear_path)
        manifest["non_linear"] = {
            "file": non_linear_path.name,
            "sha256": _sha256(non_linear_path),
            "bytes": non_linear_path.stat().st_size,
            "tensor_count": len(non_linear),
        }
    manifest["packed_bytes"] = sum(
        item["bytes"] for item in manifest["linears"].values()
    ) + manifest["non_linear"]["bytes"]
    manifest["complete"] = True
    _write_json(manifest_path, manifest)
    return manifest


@contextmanager
def _linear_parameters_on_meta(torch_module):
    original = torch_module.nn.Linear.__init__

    def initialize(module, in_features, out_features, bias=True, device=None, dtype=None):
        return original(
            module,
            in_features,
            out_features,
            bias=bias,
            device="meta",
            dtype=dtype,
        )

    torch_module.nn.Linear.__init__ = initialize
    try:
        yield
    finally:
        torch_module.nn.Linear.__init__ = original


def load_nf24_i16_prepacked_model(
    packed_dir: str | Path,
    model_factory: Callable[[], Any],
    *,
    verify_checksums: bool = False,
):
    import torch
    import torch.nn as nn
    from safetensors.torch import load_file

    root = Path(packed_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format") != FORMAT or not manifest.get("complete"):
        raise RuntimeError(f"invalid or incomplete NF24 checkpoint: {root}")
    if manifest.get("residual_mode") != "nf24_i16":
        raise RuntimeError("prepacked checkpoint is not NF24 int16")

    with _linear_parameters_on_meta(torch):
        model = model_factory()
    linears = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    }
    if set(linears) != set(manifest["linears"]):
        missing = sorted(set(linears) - set(manifest["linears"]))
        extra = sorted(set(manifest["linears"]) - set(linears))
        raise RuntimeError(f"prepacked/model Linear mismatch: missing={missing} extra={extra}")

    non_linear_info = manifest["non_linear"]
    non_linear_path = root / non_linear_info["file"]
    if verify_checksums and _sha256(non_linear_path) != non_linear_info["sha256"]:
        raise RuntimeError(f"checksum mismatch: {non_linear_path}")
    non_linear = load_file(non_linear_path.as_posix(), device="cpu")
    incompatible = model.load_state_dict(non_linear, strict=False, assign=True)
    expected_missing = {
        f"{name}.{field}"
        for name, module in linears.items()
        for field in (("weight", "bias") if module.bias is not None else ("weight",))
    }
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "non-Linear state mismatch: "
            f"missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
        )

    loaded_bytes = non_linear_info["bytes"]
    for name, module in linears.items():
        item = manifest["linears"][name]
        path = root / item["file"]
        if verify_checksums and _sha256(path) != item["sha256"]:
            raise RuntimeError(f"checksum mismatch: {path}")
        tensors = load_file(path.as_posix(), device="cpu")
        expected_shape = (int(item["in_features"]), int(item["out_features"]))
        if tuple(tensors["code01_t"].shape) != expected_shape:
            raise RuntimeError(f"packed shape mismatch for {name}")
        module.register_buffer("_native_rnf8_codes0_t", tensors["code01_t"], persistent=False)
        module.register_buffer("_native_rnf8_codes1_t", tensors["q2_t"], persistent=False)
        module.register_buffer("_native_rnf8_codes2_t", tensors["q2_t"], persistent=False)
        module.register_buffer("_native_rnf8_scales0", tensors["scale"], persistent=False)
        module.register_buffer("_native_rnf8_scales1", tensors["scale"], persistent=False)
        module.register_buffer("_native_rnf8_scales2", tensors["scale"], persistent=False)
        module.register_buffer("_native_rnf8_bias", tensors["bias"], persistent=False)
        module.weight = nn.Parameter(torch.empty(0, dtype=torch.float32), requires_grad=False)
        if module.bias is not None:
            module.bias = nn.Parameter(torch.empty(0, dtype=torch.float32), requires_grad=False)
        loaded_bytes += item["bytes"]

    model._native_rnf8_prepacked_manifest = manifest
    model._native_rnf8_prepacked_load = {
        "enabled": True,
        "format": FORMAT,
        "directory": root.as_posix(),
        "linear_count": len(linears),
        "loaded_bytes": loaded_bytes,
        "checksums_verified": bool(verify_checksums),
        "source_checkpoint_loaded": False,
    }
    return model
