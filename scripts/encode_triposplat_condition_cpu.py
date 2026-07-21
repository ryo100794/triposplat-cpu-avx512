#!/usr/bin/env python3
"""Encode a prepared TripoSplat RGB image into condition features on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get("GS_PROJECT_ROOT", SCRIPT_DIR.parent)).resolve()
REPO = Path(os.environ.get("TRIPOSPLAT_REPO", PROJECT / "vendor" / "TripoSplat")).resolve()
CKPTS = Path(os.environ.get("TRIPOSPLAT_CKPTS", PROJECT / "models" / "TripoSplat" / "ckpts")).resolve()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Prepared RGB image")
    parser.add_argument("--output", type=Path, required=True, help="Condition NPZ")
    parser.add_argument("--canvas-size", type=int, default=1024)
    parser.add_argument("--model-dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--vae-deterministic", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    sys.path.insert(0, str(SCRIPT_DIR))
    sys.path.insert(0, str(REPO))

    import torch
    import triposplat
    from PIL import Image
    from run_triposplat_encoded_external_noise import (
        encode_image_controlled,
        save_condition_npz,
        tensor_dict_fingerprint,
    )
    from triposplat import TripoSplatPipeline, load_dinov3, load_vae_encoder

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    torch.set_num_interop_threads(1)
    triposplat._CANVAS_SIZE = int(args.canvas_size)
    dtype = torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float32
    device = torch.device("cpu")

    total_started = time.perf_counter()
    load_started = time.perf_counter()
    pipe = TripoSplatPipeline.__new__(TripoSplatPipeline)
    pipe._device = device
    pipe.rmbg = None
    pipe.flow_model = None
    pipe.decoder = None
    pipe.dinov3 = load_dinov3(
        str(CKPTS / "clip_vision/dino_v3_vit_h.safetensors"), device=device, dtype=dtype
    )
    pipe.vae_encoder = load_vae_encoder(
        str(CKPTS / "vae/flux2-vae.safetensors"), device=device, dtype=dtype
    )
    load_sec = time.perf_counter() - load_started

    image = Image.open(args.input).convert("RGB")
    generator = torch.Generator(device=device).manual_seed(0)
    encode_started = time.perf_counter()
    with torch.inference_mode():
        condition = encode_image_controlled(
            pipe,
            image,
            generator=generator,
            vae_deterministic=bool(args.vae_deterministic),
        )
    encode_sec = time.perf_counter() - encode_started
    fingerprints = tensor_dict_fingerprint(condition)
    metadata = {
        "created_at": utc_now(),
        "source": "cpu_dinov3_vae_encode",
        "input": args.input.as_posix(),
        "input_sha256": file_sha256(args.input),
        "canvas_size": int(args.canvas_size),
        "device": "cpu",
        "model_dtype": args.model_dtype,
        "vae_deterministic": bool(args.vae_deterministic),
        "load_sec": load_sec,
        "encode_sec": encode_sec,
        "elapsed_sec": time.perf_counter() - total_started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "fingerprints": fingerprints,
    }
    save_condition_npz(args.output, condition, metadata)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    metadata["output"] = args.output.as_posix()
    metadata["output_sha256"] = file_sha256(args.output)
    manifest_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
