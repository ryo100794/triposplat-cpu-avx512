from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
from datetime import UTC, datetime
from pathlib import Path

from . import db


def _meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.strip().split()[0]) * 1024
    return values


def _cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text(encoding="ascii").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def _effective_memory(memory: dict[str, int]) -> tuple[int | None, int | None]:
    candidates = (
        (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
        (
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ),
    )
    for limit_path, usage_path in candidates:
        if not limit_path.is_file() or not usage_path.is_file():
            continue
        raw_limit = limit_path.read_text(encoding="ascii").strip()
        if raw_limit == "max":
            continue
        limit = int(raw_limit)
        if limit <= 0 or limit >= (1 << 60):
            continue
        usage = int(usage_path.read_text(encoding="ascii").strip())
        return limit, max(0, limit - usage)
    return memory.get("MemTotal"), memory.get("MemAvailable")


def collect_snapshot(workspace: Path) -> dict[str, object]:
    memory = _meminfo()
    memory_total, memory_available = _effective_memory(memory)
    disk = shutil.disk_usage(workspace)
    process_cpu_count = getattr(os, "process_cpu_count", os.cpu_count)
    return {
        "host": socket.gethostname(),
        "cpu_model": _cpu_model(),
        "cpu_threads": process_cpu_count(),
        "memory_total_bytes": memory_total,
        "memory_available_bytes": memory_available,
        "workspace_total_bytes": disk.total,
        "workspace_available_bytes": disk.free,
        "dashboard_status": "running",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage TripoSplat dashboard data.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Create PostgreSQL tables and indexes.")
    seed_parser = sub.add_parser("seed", help="Upsert dashboard seed data.")
    seed_parser.add_argument("path", type=Path)
    snapshot = sub.add_parser("snapshot", help="Record current compute resource capacity.")
    snapshot.add_argument("--workspace", type=Path, default=Path("/workspace"))
    sub.add_parser("health", help="Check PostgreSQL connectivity.")
    activity = sub.add_parser("activity", help="Append one activity event.")
    activity.add_argument("--key", required=True)
    activity.add_argument("--kind", default="work")
    activity.add_argument("--title", required=True)
    activity.add_argument("--detail", default="")
    activity.add_argument("--status", default="info")
    milestone = sub.add_parser("milestone", help="Update milestone progress and status.")
    milestone.add_argument("slug")
    milestone.add_argument("--status", choices=("pending", "running", "review", "complete", "blocked"), required=True)
    milestone.add_argument("--progress", type=int, choices=range(101), required=True)
    milestone.add_argument("--summary")
    args = parser.parse_args(argv)

    if args.command == "init":
        db.init_schema()
        result = {"status": "ok", "action": "init"}
    elif args.command == "seed":
        result = {"status": "ok", "action": "seed", "counts": db.seed(args.path)}
    elif args.command == "snapshot":
        snapshot_data = collect_snapshot(args.workspace)
        db.record_system_snapshot(snapshot_data)
        result = {"status": "ok", "action": "snapshot", "snapshot": snapshot_data}
    elif args.command == "health":
        result = db.health()
    elif args.command == "activity":
        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO activity_log (event_key,kind,title,detail,status,occurred_at)
                VALUES (%s,%s,%s,%s,%s,%s)
                ON CONFLICT (event_key) DO UPDATE SET kind=excluded.kind,title=excluded.title,
                    detail=excluded.detail,status=excluded.status,occurred_at=excluded.occurred_at
                """,
                (args.key, args.kind, args.title, args.detail, args.status, datetime.now(UTC)),
            )
        result = {"status": "ok", "action": "activity", "key": args.key}
    else:
        with db.connection() as conn:
            fields = ["status=%s", "progress=%s", "updated_at=now()"]
            values: list[object] = [args.status, args.progress]
            if args.summary is not None:
                fields.append("summary=%s")
                values.append(args.summary)
            values.append(args.slug)
            cursor = conn.execute(
                f"UPDATE milestones SET {', '.join(fields)} WHERE slug=%s RETURNING slug",
                values,
            )
            if cursor.fetchone() is None:
                raise SystemExit(f"unknown milestone: {args.slug}")
        result = {"status": "ok", "action": "milestone", "slug": args.slug}
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
