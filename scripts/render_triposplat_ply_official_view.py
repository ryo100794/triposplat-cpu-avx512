#!/usr/bin/env python3
"""Render TripoSplat PLYs with the bundled SparkJS viewer's default pose.

The TripoSplat repository ships `static/viewer/viewer.html`, whose initial
display is not one of the simple axis presets used by the earlier local
renderer.  It applies:

    splat.rotation.y = pi / 2
    splatRoot.rotation.x = pi
    camera.position = (0, 0.3, 1.8)
    controls.target = (0, 0, 0)
    fov = 45 degrees

This script keeps the same low-resource Gaussian rasterization math as
`render_triposplat_ply_ref.py` but evaluates that official-view pose.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from render_triposplat_ply_ref import (  # noqa: E402
    Camera,
    compare,
    load_gaussians,
    look_at,
    render_ref_style,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def apply_model_transform(g: dict[str, np.ndarray], transform: np.ndarray) -> dict[str, np.ndarray]:
    out = dict(g)
    out["means"] = (g["means"] @ transform.T).astype(np.float32)
    out["rot"] = (transform[None, :, :] @ g["rot"]).astype(np.float32)
    return out


def make_official_camera(width: int, height: int, fov_deg: float, eye_y: float, distance: float) -> Camera:
    eye = np.array([0.0, eye_y, distance], dtype=np.float32)
    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    r, t = look_at(eye, target, up)
    focal = (height * 0.5) / math.tan(math.radians(fov_deg) * 0.5)
    k = np.array(
        [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return Camera(k, r, t, height, width)


def model_transform(root_x_deg: float, splat_yaw_deg: float, order: str) -> np.ndarray:
    root_x = rot_x(math.radians(root_x_deg))
    splat_y = rot_y(math.radians(splat_yaw_deg))
    if order == "root_then_splat":
        return root_x @ splat_y
    if order == "splat_then_root":
        return splat_y @ root_x
    raise ValueError(f"unknown transform order: {order}")


def contact_sheet(paths: list[Path], out_path: Path) -> None:
    images = [Image.open(p).convert("RGB") for p in paths]
    if not images:
        return
    w = max(im.width for im in images)
    h = max(im.height for im in images)
    sheet = Image.new("RGB", (w * len(images), h), (0, 0, 0))
    for i, im in enumerate(images):
        sheet.paste(im, (i * w, 0))
    sheet.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fov-deg", type=float, default=45.0)
    parser.add_argument("--eye-y", type=float, default=0.3)
    parser.add_argument("--distance", type=float, default=1.8)
    parser.add_argument("--root-x-deg", type=float, default=180.0)
    parser.add_argument("--splat-yaw-deg", type=float, default=90.0)
    parser.add_argument("--transform-order", choices=["root_then_splat", "splat_then_root"], default="root_then_splat")
    parser.add_argument("--radius-clip", type=float, default=3.0)
    parser.add_argument("--min-radius-px", type=int, default=1)
    parser.add_argument("--max-radius-px", type=int, default=24)
    parser.add_argument("--min-weight", type=float, default=1e-5)
    parser.add_argument("--alpha-scale", type=float, default=1.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = load_gaussians(args.ply)
    transform = model_transform(args.root_x_deg, args.splat_yaw_deg, args.transform_order)
    g = apply_model_transform(base, transform)
    cam = make_official_camera(args.width, args.height, args.fov_deg, args.eye_y, args.distance)
    render = render_ref_style(
        g,
        cam,
        radius_clip=args.radius_clip,
        min_radius_px=args.min_radius_px,
        max_radius_px=args.max_radius_px,
        min_weight=args.min_weight,
        alpha_scale=args.alpha_scale,
    )
    render_path = args.output_dir / "official_spark_default.png"
    render.save(render_path)
    outputs = [render_path]
    result = {"view": "official_spark_default", "render": render_path.name}
    if args.source:
        compare_path = args.output_dir / "official_spark_default_compare.png"
        result.update(compare(args.source, render, compare_path))
        result["compare"] = compare_path.name
        outputs.append(compare_path)
    contact_sheet(outputs, args.output_dir / "official_spark_default_contact.png")

    manifest = {
        "created_at": utc_now(),
        "implementation": "ref_style_cpu_renderer_official_spark_pose",
        "ply": args.ply.as_posix(),
        "source": args.source.as_posix() if args.source else None,
        "gaussian_count": int(g["means"].shape[0]),
        "model_transform": {
            "splat_rotation_y_deg": args.splat_yaw_deg,
            "splat_root_rotation_x_deg": args.root_x_deg,
            "transform_order": args.transform_order,
            "matrix": transform.astype(float).tolist(),
        },
        "camera": {
            "position": [0.0, args.eye_y, args.distance],
            "target": [0.0, 0.0, 0.0],
            "fov_deg": args.fov_deg,
            "width": args.width,
            "height": args.height,
        },
        "settings": {
            "radius_clip": args.radius_clip,
            "min_radius_px": args.min_radius_px,
            "max_radius_px": args.max_radius_px,
            "min_weight": args.min_weight,
            "alpha_scale": args.alpha_scale,
        },
        "result": result,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
