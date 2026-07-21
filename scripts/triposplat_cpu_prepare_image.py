#!/usr/bin/env python3
"""Run the upstream TripoSplat background-removal phase on CPU."""

from __future__ import annotations

import argparse
import json
import os
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--canvas-size", type=int, default=1024)
    parser.add_argument("--erode-radius", type=int, default=1)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    sys.path.insert(0, str(REPO))
    import torch
    import triposplat
    from PIL import Image
    from triposplat import load_rmbg, preprocess_image

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    torch.set_num_interop_threads(1)
    triposplat._CANVAS_SIZE = int(args.canvas_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rmbg = load_rmbg(
        str(CKPTS / "background_removal" / "birefnet.safetensors"),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    prepared = preprocess_image(Image.open(args.input), rmbg, erode_radius=args.erode_radius)
    prepared_path = args.output_dir / "prepared_rgb.webp"
    rgba_path = args.output_dir / "prepared_rgba.png"
    prepared.save(prepared_path)
    rgba = prepared.convert("RGBA")
    # Alpha below 255 tells the upstream preprocessing path not to rerun BiRefNet.
    rgba.putalpha(254)
    rgba.save(rgba_path)
    manifest = {
        "created_at": utc_now(),
        "input": args.input.as_posix(),
        "canvas_size": int(args.canvas_size),
        "erode_radius": int(args.erode_radius),
        "device": "cpu",
        "dtype": "float32",
        "elapsed_sec": time.perf_counter() - started,
        "prepared_rgb": prepared_path.as_posix(),
        "prepared_rgba": rgba_path.as_posix(),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
