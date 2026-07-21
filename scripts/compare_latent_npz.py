#!/usr/bin/env python3
"""Compare TripoSplat latent NPZ files without importing torch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _metadata(data) -> dict:
    if "metadata_json" not in data:
        return {}
    try:
        return json.loads(str(np.asarray(data["metadata_json"]).item()))
    except Exception as exc:
        return {"metadata_parse_error": repr(exc)}


def _stats(ref: np.ndarray, cand: np.ndarray) -> dict:
    ref = np.asarray(ref, dtype=np.float64)
    cand = np.asarray(cand, dtype=np.float64)
    diff = cand - ref
    denom = np.maximum(np.abs(ref), 1e-8)
    finite = np.isfinite(diff)
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    return {
        "shape": list(ref.shape),
        "ref_mean": float(np.mean(ref)),
        "ref_std": float(np.std(ref)),
        "cand_mean": float(np.mean(cand)),
        "cand_std": float(np.std(cand)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mae,
        "max_abs": float(np.max(np.abs(diff))),
        "mean_relative_abs": float(np.mean(np.abs(diff) / denom)),
        "p50_abs": float(np.percentile(np.abs(diff), 50)),
        "p95_abs": float(np.percentile(np.abs(diff), 95)),
        "p99_abs": float(np.percentile(np.abs(diff), 99)),
        "finite_count": int(finite.sum()),
        "nan_count": int(np.isnan(diff).sum()),
        "posinf_count": int(np.isposinf(diff).sum()),
        "neginf_count": int(np.isneginf(diff).sum()),
    }


def compare(reference: Path, candidate: Path) -> dict:
    ref = np.load(reference)
    cand = np.load(candidate)
    keys = sorted((set(ref.files) & set(cand.files)) - {"metadata_json"})
    missing_in_candidate = sorted((set(ref.files) - {"metadata_json"}) - set(cand.files))
    extra_in_candidate = sorted((set(cand.files) - {"metadata_json"}) - set(ref.files))
    stats = {}
    for key in keys:
        if ref[key].shape != cand[key].shape:
            stats[key] = {
                "shape_mismatch": True,
                "reference_shape": list(ref[key].shape),
                "candidate_shape": list(cand[key].shape),
            }
            continue
        stats[key] = _stats(ref[key], cand[key])
    score_parts = []
    for key, value in stats.items():
        if "mse" in value:
            score_parts.append(value["mse"])
    score = float(np.sqrt(np.mean(score_parts))) if score_parts else None
    return {
        "reference": reference.as_posix(),
        "candidate": candidate.as_posix(),
        "reference_metadata": _metadata(ref),
        "candidate_metadata": _metadata(cand),
        "common_keys": keys,
        "missing_in_candidate": missing_in_candidate,
        "extra_in_candidate": extra_in_candidate,
        "per_key": stats,
        "combined_rmse_from_key_mse": score,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = compare(args.reference, args.candidate)
    text = json.dumps(out, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
