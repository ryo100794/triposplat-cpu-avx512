#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from native_linear_nf24_prepacked import pack_nf24_i16_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert the official TripoSplat Flow checkpoint to resumable NF24 int16 shards."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-linear-count", type=int, default=206)
    args = parser.parse_args()
    manifest = pack_nf24_i16_checkpoint(
        args.checkpoint,
        args.output_dir,
        expected_linear_count=args.expected_linear_count,
    )
    print(json.dumps({
        "format": manifest["format"],
        "linear_count": manifest["linear_count"],
        "packed_bytes": manifest["packed_bytes"],
        "manifest": (args.output_dir / "manifest.json").resolve().as_posix(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
