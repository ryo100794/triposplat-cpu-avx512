#!/usr/bin/env python3
"""Fail early when an output filesystem cannot reserve the requested bytes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--bytes", type=int, default=268_435_456)
    args = parser.parse_args()
    if args.bytes <= 0:
        raise ValueError("--bytes must be positive")
    args.directory.mkdir(parents=True, exist_ok=True)
    probe = args.directory / f".capacity_probe_{os.getpid()}"
    fd = os.open(probe, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        if hasattr(os, "posix_fallocate"):
            os.posix_fallocate(fd, 0, args.bytes)
        else:
            block = bytes(min(1024 * 1024, args.bytes))
            remaining = args.bytes
            while remaining:
                written = os.write(fd, block[: min(len(block), remaining)])
                remaining -= written
        os.fsync(fd)
    finally:
        os.close(fd)
        probe.unlink(missing_ok=True)
    print(f"capacity_check=pass directory={args.directory} bytes={args.bytes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
