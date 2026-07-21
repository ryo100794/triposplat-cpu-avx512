#!/usr/bin/env python3
"""Generate device-independent TripoSplat flow noise with NumPy PCG64."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latent-tokens", type=int, default=8192)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--camera-channels", type=int, default=5)
    args = parser.parse_args()

    shapes = {
        "latent": (1, int(args.latent_tokens), int(args.latent_channels)),
        "camera": (1, 1, int(args.camera_channels)),
    }
    rng = np.random.default_rng(int(args.seed))
    arrays = {key: rng.standard_normal(shape).astype(np.float32) for key, shape in shapes.items()}
    metadata = {
        "created_at": utc_now(),
        "source": "numpy_pcg64_standard_normal",
        "seed": int(args.seed),
        "shape": {key: list(shape) for key, shape in shapes.items()},
        "sha256_float32": {
            key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in arrays.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **arrays,
        metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)),
    )
    print(json.dumps({**metadata, "output": args.output.as_posix()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
