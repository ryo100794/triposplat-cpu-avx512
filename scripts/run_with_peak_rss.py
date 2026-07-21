#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def _children(pid: int) -> list[int]:
    path = Path(f"/proc/{pid}/task/{pid}/children")
    try:
        return [int(value) for value in path.read_text().split()]
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return []


def _process_tree(root: int) -> list[int]:
    pending = [root]
    observed: list[int] = []
    while pending:
        pid = pending.pop()
        if pid in observed:
            continue
        observed.append(pid)
        pending.extend(_children(pid))
    return observed


def _rss_bytes(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a command and report peak aggregate RSS for its process tree."
    )
    parser.add_argument("--interval-ms", type=int, default=50)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    if args.interval_ms <= 0:
        parser.error("--interval-ms must be positive")

    started = time.monotonic()
    process = subprocess.Popen(command)
    peak_rss = 0
    peak_processes: list[dict[str, int]] = []
    samples = 0
    while process.poll() is None:
        current = [
            {"pid": pid, "rss_bytes": _rss_bytes(pid)}
            for pid in _process_tree(process.pid)
        ]
        aggregate = sum(item["rss_bytes"] for item in current)
        if aggregate > peak_rss:
            peak_rss = aggregate
            peak_processes = current
        samples += 1
        time.sleep(args.interval_ms / 1000.0)
    return_code = process.wait()
    result = {
        "command": command,
        "return_code": return_code,
        "elapsed_sec": time.monotonic() - started,
        "peak_aggregate_rss_bytes": peak_rss,
        "peak_processes": peak_processes,
        "sample_interval_ms": args.interval_ms,
        "sample_count": samples,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        os.replace(temporary, args.output_json)
    print("peak_rss=" + json.dumps(result, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
