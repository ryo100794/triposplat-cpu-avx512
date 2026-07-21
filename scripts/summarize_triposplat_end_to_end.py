#!/usr/bin/env python3
"""Build and validate one manifest for the staged CPU end-to-end pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fallback_values(value, path=""):
    out = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else key
            if key in {"fallbacks", "fallback_calls"} and isinstance(item, (int, float)):
                out.append((child, item))
            out.extend(fallback_values(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            out.extend(fallback_values(item, f"{path}[{index}]"))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--raw-input", type=Path, required=True)
    parser.add_argument("--started-epoch", type=float, required=True)
    parser.add_argument("--max-flow-sec", type=float, default=7200.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "prepare_manifest": args.run_dir / "prepared/manifest.json",
        "condition_manifest": args.run_dir / "condition/condition.npz.manifest.json",
        "flow_manifest": args.run_dir / "flow/manifest.json",
        "flow_compare": args.run_dir / "flow/compare_vs_float32_s20.json",
        "decode_manifest": args.run_dir / "gaussian/manifest.json",
        "render_manifest": args.run_dir / "render/manifest.json",
        "ply": args.run_dir / "gaussian/output.ply",
        "splat": args.run_dir / "gaussian/output.splat",
        "viewer": args.run_dir / "gaussian/viewer.html",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing end-to-end outputs: {missing}")

    prepare = load_json(paths["prepare_manifest"])
    condition = load_json(paths["condition_manifest"])
    flow = load_json(paths["flow_manifest"])
    compare = load_json(paths["flow_compare"])
    decode = load_json(paths["decode_manifest"])
    render = load_json(paths["render_manifest"])
    fallbacks = fallback_values(flow)
    fallback_sum = sum(float(value) for _, value in fallbacks)
    latent = compare["per_key"]["latent"]
    camera = compare["per_key"]["camera"]
    completed = (
        len(flow.get("sampler", {}).get("groups", [{}])[0].get("timings", [])) == 20
        and float(flow["elapsed_sec"]) <= float(args.max_flow_sec)
        and fallback_sum == 0.0
        and float(compare["combined_rmse_from_key_mse"]) <= 5.0e-4
        and int(latent["nan_count"]) == 0
        and int(camera["nan_count"]) == 0
    )
    files = {}
    for name in ("ply", "splat", "viewer"):
        path = paths[name]
        files[name] = {
            "path": path.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    payload = {
        "created_at": utc_now(),
        "algorithm": "TripoSplat staged CPU-only strict native AVX-512 end-to-end",
        "completed": completed,
        "raw_input": args.raw_input.as_posix(),
        "raw_input_sha256": sha256(args.raw_input),
        "wall_sec": time.time() - float(args.started_epoch),
        "stages": {
            "background_removal": prepare,
            "condition_encode": condition,
            "flow": {
                "elapsed_sec": flow["elapsed_sec"],
                "max_flow_sec": float(args.max_flow_sec),
                "steps": len(flow["sampler"]["groups"][0]["timings"]),
                "fallback_fields": len(fallbacks),
                "fallback_sum": fallback_sum,
                "fallback_nonzero": [[path, value] for path, value in fallbacks if value],
            },
            "quality": {
                "combined_rmse": compare["combined_rmse_from_key_mse"],
                "latent_rmse": latent["rmse"],
                "camera_rmse": camera["rmse"],
                "latent_nan": latent["nan_count"],
                "camera_nan": camera["nan_count"],
            },
            "decode_export": decode,
            "render": render,
        },
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not completed:
        raise SystemExit(3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
