#!/usr/bin/env python3
"""Render TripoSplat/3DGS PLY files with a low-resource ref-style splatter.

This keeps the important math from the local ref implementation:

    Sigma_2d = J Sigma_3d J^T
    w(p) = exp(-0.5 * (p-mu)^T Sigma_2d^-1 (p-mu))
    C <- C + T * alpha * w * color
    T <- T * (1 - alpha * w)

Unlike the training-oriented ref script, this renderer does not need autograd,
so it updates only the touched image patch in-place and never pads each splat to
full frame size. That is the main low-resource substitution used for CPU video
rendering and same-view checks.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


C0 = 0.28209479177387814


@dataclass(frozen=True)
class Camera:
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    h: int
    w: int


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def read_binary_ply(path: Path) -> dict[str, np.ndarray]:
    data = path.read_bytes()
    marker = b"end_header\n"
    header_end = data.find(marker)
    if header_end < 0:
        raise ValueError(f"PLY header terminator not found: {path}")
    header_end += len(marker)
    header = data[:header_end].decode("ascii", errors="replace").splitlines()
    if "format binary_little_endian 1.0" not in header:
        raise ValueError("Only binary_little_endian PLY is supported")

    count = None
    props: list[str] = []
    in_vertex = False
    for line in header:
        parts = line.split()
        if not parts:
            continue
        if parts[:2] == ["element", "vertex"]:
            count = int(parts[2])
            in_vertex = True
            continue
        if parts[0] == "element" and parts[1] != "vertex":
            in_vertex = False
        if in_vertex and parts[:2] == ["property", "float"]:
            props.append(parts[2])

    if count is None:
        raise ValueError("PLY vertex count not found")
    dtype = np.dtype([(name, "<f4") for name in props])
    arr = np.frombuffer(data, dtype=dtype, count=count, offset=header_end)
    return {name: np.asarray(arr[name], dtype=np.float32) for name in props}


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = q.astype(np.float32, copy=True)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-8
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - z * w)
    r[:, 0, 2] = 2 * (x * z + y * w)
    r[:, 1, 0] = 2 * (x * y + z * w)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - x * w)
    r[:, 2, 0] = 2 * (x * z - y * w)
    r[:, 2, 1] = 2 * (y * z + x * w)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def load_gaussians(path: Path) -> dict[str, np.ndarray]:
    p = read_binary_ply(path)
    required = {
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    }
    missing = sorted(required.difference(p))
    if missing:
        raise ValueError(f"PLY is missing fields: {missing}")
    means = np.stack([p["x"], p["y"], p["z"]], axis=1).astype(np.float32)
    fdc = np.stack([p["f_dc_0"], p["f_dc_1"], p["f_dc_2"]], axis=1)
    colors = np.clip(fdc * C0 + 0.5, 0.0, 1.0).astype(np.float32)
    alpha = sigmoid(p["opacity"]).astype(np.float32)
    scales = np.exp(np.stack([p["scale_0"], p["scale_1"], p["scale_2"]], axis=1)).astype(np.float32)
    quat = np.stack([p["rot_0"], p["rot_1"], p["rot_2"], p["rot_3"]], axis=1)
    rot = quat_to_rotmat(quat)
    return {"means": means, "colors": colors, "alpha": alpha, "scales": scales, "rot": rot}


def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forward = normalize(target - eye)
    right = normalize(np.cross(up, forward))
    true_up = normalize(np.cross(forward, right))
    r = np.stack([right, true_up, forward], axis=0).astype(np.float32)
    t = -(r @ eye.astype(np.float32))
    return r, t.astype(np.float32)


def make_camera(
    preset: str,
    h: int,
    w: int,
    fov_deg: float,
    center: np.ndarray,
    radius: float,
    distance_scale: float,
) -> Camera:
    dist = max(radius * distance_scale, 1e-3)
    eye_by_preset = {
        "front_z": np.array([0.0, 0.0, dist], dtype=np.float32),
        "back_z": np.array([0.0, 0.0, -dist], dtype=np.float32),
        "front_x": np.array([dist, 0.0, 0.0], dtype=np.float32),
        "back_x": np.array([-dist, 0.0, 0.0], dtype=np.float32),
        "front_y": np.array([0.0, dist, 0.0], dtype=np.float32),
        "back_y": np.array([0.0, -dist, 0.0], dtype=np.float32),
    }
    if preset not in eye_by_preset:
        raise ValueError(f"Unknown view preset: {preset}")
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if preset in {"front_y", "back_y"}:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    eye = center.astype(np.float32) + eye_by_preset[preset]
    r, t = look_at(eye, center.astype(np.float32), up)
    focal = (w * 0.5) / math.tan(math.radians(fov_deg) * 0.5)
    k = np.array([[focal, 0.0, w * 0.5], [0.0, focal, h * 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    return Camera(k, r, t, h, w)


def project(means: np.ndarray, cam: Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xc = means @ cam.R.T + cam.t[None, :]
    z = np.clip(xc[:, 2], 1e-6, None)
    u = cam.K[0, 0] * (xc[:, 0] / z) + cam.K[0, 2]
    v = cam.K[1, 1] * (xc[:, 1] / z) + cam.K[1, 2]
    return np.stack([u, v], axis=1).astype(np.float32), z.astype(np.float32), xc.astype(np.float32)


def render_ref_style(
    g: dict[str, np.ndarray],
    cam: Camera,
    radius_clip: float,
    min_radius_px: int,
    max_radius_px: int,
    min_weight: float,
    alpha_scale: float,
) -> Image.Image:
    means = g["means"]
    colors = g["colors"]
    alphas = np.clip(g["alpha"] * alpha_scale, 0.0, 0.98)
    scales = g["scales"]
    rots = g["rot"]

    uv, depth, xcam = project(means, cam)
    visible = depth > 1e-5
    order = np.argsort(depth)
    image = np.zeros((cam.h, cam.w, 3), dtype=np.float32)
    trans = np.ones((cam.h, cam.w), dtype=np.float32)

    fx = cam.K[0, 0]
    fy = cam.K[1, 1]
    for idx in order:
        if not visible[idx] or alphas[idx] <= 1e-5:
            continue
        z = max(float(depth[idx]), 1e-6)
        x = float(xcam[idx, 0] / z)
        y = float(xcam[idx, 1] / z)
        j_cam = np.array([[fx / z, 0.0, -fx * x / z], [0.0, fy / z, -fy * y / z]], dtype=np.float32)
        j = j_cam @ cam.R
        r = rots[idx]
        sigma3 = r @ np.diag(scales[idx] * scales[idx]).astype(np.float32) @ r.T
        sigma2 = j @ sigma3 @ j.T
        sigma2[0, 0] += 1e-5
        sigma2[1, 1] += 1e-5
        try:
            eigvals = np.linalg.eigvalsh(sigma2)
        except np.linalg.LinAlgError:
            continue
        rad = int(math.ceil(radius_clip * math.sqrt(max(float(eigvals[-1]), 1e-8))))
        if rad < min_radius_px:
            continue
        rad = min(rad, max_radius_px)
        u = float(uv[idx, 0])
        v = float(uv[idx, 1])
        u0 = max(0, int(math.floor(u - rad)))
        u1 = min(cam.w - 1, int(math.ceil(u + rad)))
        v0 = max(0, int(math.floor(v - rad)))
        v1 = min(cam.h - 1, int(math.ceil(v + rad)))
        if u0 > u1 or v0 > v1:
            continue
        try:
            inv = np.linalg.inv(sigma2)
        except np.linalg.LinAlgError:
            continue
        yy, xx = np.mgrid[v0:v1 + 1, u0:u1 + 1].astype(np.float32)
        dx = xx - u
        dy = yy - v
        m2 = inv[0, 0] * dx * dx + 2.0 * inv[0, 1] * dx * dy + inv[1, 1] * dy * dy
        wgt = np.exp(-0.5 * np.clip(m2, 0.0, 30.0)).astype(np.float32)
        if float(wgt.max(initial=0.0)) < min_weight:
            continue
        a = np.clip(wgt * alphas[idx], 0.0, 0.98)
        patch_t = trans[v0:v1 + 1, u0:u1 + 1]
        image[v0:v1 + 1, u0:u1 + 1, :] += (patch_t * a)[..., None] * colors[idx]
        trans[v0:v1 + 1, u0:u1 + 1] = patch_t * (1.0 - a)

    return Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8), "RGB")


def compare(source_path: Path, render: Image.Image, out_path: Path) -> dict[str, float]:
    source = Image.open(source_path).convert("RGB").resize(render.size, Image.LANCZOS)
    src = np.asarray(source).astype(np.float32) / 255.0
    rnd = np.asarray(render).astype(np.float32) / 255.0
    diff = np.abs(src - rnd)
    mse = float(np.mean((src - rnd) ** 2))
    mae = float(np.mean(diff))
    psnr = float(-10.0 * np.log10(mse)) if mse > 0 else 99.0
    diff_img = Image.fromarray(np.clip(diff * 255.0, 0, 255).astype(np.uint8), "RGB")
    sheet = Image.new("RGB", (render.width * 3, render.height), (0, 0, 0))
    sheet.paste(source, (0, 0))
    sheet.paste(render, (render.width, 0))
    sheet.paste(diff_img, (render.width * 2, 0))
    sheet.save(out_path)
    return {"mse": mse, "mae": mae, "psnr_db": psnr}


def make_contact_sheet(paths: list[Path], out_path: Path, thumb_w: int) -> None:
    thumbs: list[Image.Image] = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        h = max(1, int(round(im.height * thumb_w / im.width)))
        thumbs.append(im.resize((thumb_w, h), Image.LANCZOS))
    if not thumbs:
        return
    sheet = Image.new("RGB", (thumb_w * len(thumbs), max(t.height for t in thumbs)), (0, 0, 0))
    for i, im in enumerate(thumbs):
        sheet.paste(im, (i * thumb_w, 0))
    sheet.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--views", default="front_z,back_z,front_x,back_x,front_y,back_y")
    parser.add_argument("--fov-deg", type=float, default=38.0)
    parser.add_argument("--distance-scale", type=float, default=2.8)
    parser.add_argument("--radius-clip", type=float, default=3.0)
    parser.add_argument("--min-radius-px", type=int, default=1)
    parser.add_argument("--max-radius-px", type=int, default=16)
    parser.add_argument("--min-weight", type=float, default=1e-5)
    parser.add_argument("--alpha-scale", type=float, default=1.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    g = load_gaussians(args.ply)
    center = np.median(g["means"], axis=0)
    radius = float(np.percentile(np.linalg.norm(g["means"] - center[None, :], axis=1), 95))
    radius = max(radius, 1e-3)

    results = []
    rendered_paths = []
    for view in [v.strip() for v in args.views.split(",") if v.strip()]:
        cam = make_camera(view, args.height, args.width, args.fov_deg, center, radius, args.distance_scale)
        img = render_ref_style(
            g, cam, args.radius_clip, args.min_radius_px, args.max_radius_px,
            args.min_weight, args.alpha_scale,
        )
        render_path = args.output_dir / f"{view}.png"
        img.save(render_path)
        rendered_paths.append(render_path)
        item = {"view": view, "render": render_path.name}
        if args.source:
            compare_path = args.output_dir / f"{view}_compare.png"
            item.update(compare(args.source, img, compare_path))
            item["compare"] = compare_path.name
        results.append(item)

    make_contact_sheet(rendered_paths, args.output_dir / "views_contact_sheet.png", min(256, args.width))
    best = None
    scored = [r for r in results if "psnr_db" in r]
    if scored:
        best = max(scored, key=lambda x: x["psnr_db"])
    manifest = {
        "implementation": "ref_style_cpu_gaussian_renderer_for_triposplat_ply",
        "ply": args.ply.as_posix(),
        "source": args.source.as_posix() if args.source else None,
        "gaussian_count": int(g["means"].shape[0]),
        "center_median": center.astype(float).tolist(),
        "radius_p95": radius,
        "settings": {
            "width": args.width,
            "height": args.height,
            "fov_deg": args.fov_deg,
            "distance_scale": args.distance_scale,
            "radius_clip": args.radius_clip,
            "min_radius_px": args.min_radius_px,
            "max_radius_px": args.max_radius_px,
            "min_weight": args.min_weight,
            "alpha_scale": args.alpha_scale,
        },
        "results": results,
        "best": best,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
