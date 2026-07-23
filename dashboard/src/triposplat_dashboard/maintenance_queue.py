from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from psycopg.types.json import Jsonb

from .db import connection


TASK_TYPE = "gdrive_unused_artifact_archive"
SCHEMA = """
CREATE TABLE IF NOT EXISTS maintenance_jobs (
    job_id bigserial PRIMARY KEY,
    task_type text NOT NULL,
    parameters jsonb NOT NULL DEFAULT '{}'::jsonb,
    dedupe_key text NOT NULL UNIQUE,
    priority integer NOT NULL DEFAULT 0,
    status text NOT NULL DEFAULT 'queued'
      CHECK (status IN ('queued','running','completed','failed','cancelled')),
    attempt integer NOT NULL DEFAULT 0,
    max_attempts integer NOT NULL DEFAULT 3,
    available_at timestamptz NOT NULL DEFAULT now(),
    worker_id text,
    locked_at timestamptz,
    started_at timestamptz,
    completed_at timestamptz,
    result_summary jsonb,
    error text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_maintenance_jobs_claim
  ON maintenance_jobs(status,available_at,priority DESC,job_id);

CREATE TABLE IF NOT EXISTS maintenance_job_runs (
    run_id bigserial PRIMARY KEY,
    job_id bigint NOT NULL REFERENCES maintenance_jobs(job_id) ON DELETE CASCADE,
    attempt integer NOT NULL,
    worker_id text NOT NULL,
    status text NOT NULL CHECK (status IN ('running','completed','failed')),
    started_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz,
    result_summary jsonb,
    error text,
    UNIQUE(job_id,attempt)
);

CREATE TABLE IF NOT EXISTS maintenance_schedules (
    schedule_key text PRIMARY KEY,
    task_type text NOT NULL,
    parameters jsonb NOT NULL DEFAULT '{}'::jsonb,
    interval_seconds integer NOT NULL CHECK (interval_seconds >= 300),
    priority integer NOT NULL DEFAULT 0,
    enabled boolean NOT NULL DEFAULT true,
    next_run_at timestamptz NOT NULL DEFAULT now(),
    last_enqueued_at timestamptz,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS archived_items (
    archive_id bigserial PRIMARY KEY,
    job_id bigint NOT NULL REFERENCES maintenance_jobs(job_id),
    root_label text NOT NULL,
    item_name text NOT NULL,
    size_bytes bigint NOT NULL,
    checksum_state text NOT NULL,
    archived_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(root_label,item_name)
);
"""


def ensure_schema() -> None:
    with connection() as conn:
        conn.execute(SCHEMA, prepare=False)


def _key(schedule_key: str, available_at: datetime) -> str:
    payload = f"{schedule_key}:{available_at.isoformat()}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def register_schedule(
    *,
    schedule_key: str = "archive-unused-artifacts",
    interval_seconds: int = 3600,
    older_than_hours: int = 24,
    max_items: int = 4,
) -> None:
    ensure_schema()
    params = {
        "older_than_hours": max(1, int(older_than_hours)),
        "max_items": max(1, min(32, int(max_items))),
        "delete_after_verify": True,
    }
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO maintenance_schedules
                (schedule_key,task_type,parameters,interval_seconds,priority,enabled,next_run_at,updated_at)
            VALUES (%s,%s,%s,%s,80,true,now(),now())
            ON CONFLICT (schedule_key) DO UPDATE SET task_type=excluded.task_type,
                parameters=excluded.parameters,interval_seconds=excluded.interval_seconds,
                priority=excluded.priority,enabled=true,updated_at=now()
            """,
            (schedule_key, TASK_TYPE, Jsonb(params), max(300, int(interval_seconds))),
        )


def enqueue_due() -> int:
    inserted = 0
    with connection() as conn:
        schedules = conn.execute(
            """
            SELECT * FROM maintenance_schedules
            WHERE enabled=true AND next_run_at <= now()
            ORDER BY priority DESC,schedule_key
            FOR UPDATE SKIP LOCKED
            """
        ).fetchall()
        for schedule in schedules:
            available_at = schedule["next_run_at"]
            row = conn.execute(
                """
                INSERT INTO maintenance_jobs
                    (task_type,parameters,dedupe_key,priority,max_attempts,available_at)
                VALUES (%s,%s,%s,%s,3,now())
                ON CONFLICT (dedupe_key) DO NOTHING RETURNING job_id
                """,
                (schedule["task_type"], Jsonb(schedule["parameters"]),
                 _key(schedule["schedule_key"], available_at), schedule["priority"]),
            ).fetchone()
            inserted += int(row is not None)
            conn.execute(
                """
                UPDATE maintenance_schedules
                SET last_enqueued_at=now(),next_run_at=now()+(interval_seconds*interval '1 second'),
                    updated_at=now() WHERE schedule_key=%s
                """,
                (schedule["schedule_key"],),
            )
    return inserted


def requeue_stale(stale_minutes: int = 30) -> int:
    with connection() as conn:
        rows = conn.execute(
            """
            UPDATE maintenance_jobs
            SET status=CASE WHEN attempt < max_attempts THEN 'queued' ELSE 'failed' END,
                available_at=now(),worker_id=NULL,locked_at=NULL,
                error=COALESCE(error,'worker lease expired'),updated_at=now()
            WHERE status='running' AND locked_at < now()-(%s*interval '1 minute')
            RETURNING job_id
            """,
            (max(1, int(stale_minutes)),),
        ).fetchall()
    return len(rows)


def claim(worker_id: str) -> dict[str, Any] | None:
    with connection() as conn:
        row = conn.execute(
            """
            WITH candidate AS (
              SELECT job_id FROM maintenance_jobs
              WHERE status='queued' AND available_at<=now() AND attempt<max_attempts
              ORDER BY priority DESC,job_id FOR UPDATE SKIP LOCKED LIMIT 1
            )
            UPDATE maintenance_jobs jobs
            SET status='running',worker_id=%s,locked_at=now(),
                started_at=COALESCE(started_at,now()),attempt=attempt+1,
                error=NULL,updated_at=now()
            FROM candidate WHERE jobs.job_id=candidate.job_id RETURNING jobs.*
            """,
            (worker_id,),
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            """
            INSERT INTO maintenance_job_runs(job_id,attempt,worker_id,status)
            VALUES (%s,%s,%s,'running')
            ON CONFLICT (job_id,attempt) DO UPDATE SET worker_id=excluded.worker_id,
                status='running',started_at=now(),completed_at=NULL,error=NULL
            """,
            (row["job_id"], row["attempt"], worker_id),
        )
    return row


def heartbeat(job_id: int, worker_id: str) -> None:
    with connection() as conn:
        conn.execute(
            "UPDATE maintenance_jobs SET locked_at=now(),updated_at=now() "
            "WHERE job_id=%s AND status='running' AND worker_id=%s",
            (job_id, worker_id),
        )


def _heartbeat_loop(stop: threading.Event, job_id: int, worker_id: str) -> None:
    while not stop.wait(20):
        try:
            heartbeat(job_id, worker_id)
        except Exception as exc:
            print(f"heartbeat failed: {type(exc).__name__}: {exc}", flush=True)


def _roots() -> dict[str, Path]:
    base = Path(os.environ.get(
        "TRIPOSPLAT_ARCHIVE_BASE", "/workspace/3dgs/sujinashi_triposplat/artifacts"
    ))
    return {
        "quant-flow": base / "quant_flow",
        "audits": base / "audits",
        "comparisons": base / "comparisons",
    }


def _latest_mtime(path: Path) -> float:
    latest = path.stat().st_mtime
    if path.is_dir():
        for item in path.rglob("*"):
            try:
                latest = max(latest, item.stat().st_mtime)
            except FileNotFoundError:
                continue
    return latest


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _candidates(older_than_hours: int, max_items: int) -> list[tuple[str, Path]]:
    cutoff = time.time() - older_than_hours * 3600
    found: list[tuple[float, str, Path]] = []
    for label, root in _roots().items():
        if not root.is_dir():
            continue
        for item in root.iterdir():
            if item.name.startswith("."):
                continue
            if item.is_dir() and (item / ".active").exists():
                continue
            try:
                modified = _latest_mtime(item)
            except FileNotFoundError:
                continue
            if modified < cutoff:
                found.append((modified, label, item))
    found.sort(key=lambda value: value[0])
    return [(label, item) for _, label, item in found[:max_items]]


def _run(command: list[str], timeout: int = 1800) -> None:
    subprocess.run(command, check=True, timeout=timeout, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def archive_unused(job: dict[str, Any]) -> dict[str, Any]:
    params = job["parameters"]
    older_than = int(params.get("older_than_hours", 24))
    max_items = int(params.get("max_items", 4))
    rclone = os.environ.get("TRIPOSPLAT_RCLONE", "rclone")
    config = os.environ.get("TRIPOSPLAT_RCLONE_CONFIG", "/workspace/google-drive/rclone.conf")
    destination = os.environ.get(
        "TRIPOSPLAT_GDRIVE_ARCHIVE", "gdrive:workspace/3dgs/sujinashi_triposplat/artifacts/maintenance-archive"
    ).rstrip("/")
    archived: list[dict[str, Any]] = []
    for label, source in _candidates(older_than, max_items):
        item_size = _size(source)
        target = f"{destination}/{label}/{source.name}"
        if source.is_dir():
            _run([rclone, "copy", "--config", config, str(source), target])
            _run([rclone, "check", "--config", config, "--one-way", str(source), target])
        else:
            _run([rclone, "copyto", "--config", config, str(source), target])
            _run([rclone, "check", "--config", config, "--one-way", str(source.parent), f"{destination}/{label}", "--include", f"/{source.name}"])
        if source.is_dir():
            shutil.rmtree(source)
        else:
            source.unlink()
        archived.append({"root": label, "item": source.name, "bytes": item_size})
        with connection() as conn:
            conn.execute(
                """
                INSERT INTO archived_items(job_id,root_label,item_name,size_bytes,checksum_state)
                VALUES (%s,%s,%s,%s,'verified')
                ON CONFLICT (root_label,item_name) DO UPDATE SET job_id=excluded.job_id,
                    size_bytes=excluded.size_bytes,checksum_state='verified',archived_at=now()
                """,
                (job["job_id"], label, source.name, item_size),
            )
    return {
        "archived_count": len(archived),
        "archived_bytes": sum(item["bytes"] for item in archived),
        "items": archived,
        "checksum": "verified" if archived else "not-required",
    }


def complete(job: dict[str, Any], summary: dict[str, Any]) -> None:
    with connection() as conn:
        conn.execute(
            """
            UPDATE maintenance_jobs SET status='completed',completed_at=now(),updated_at=now(),
                result_summary=%s,worker_id=NULL,locked_at=NULL,error=NULL WHERE job_id=%s
            """,
            (Jsonb(summary), job["job_id"]),
        )
        conn.execute(
            """
            UPDATE maintenance_job_runs SET status='completed',completed_at=now(),result_summary=%s,error=NULL
            WHERE job_id=%s AND attempt=%s
            """,
            (Jsonb(summary), job["job_id"], job["attempt"]),
        )


def fail(job: dict[str, Any], error: str) -> None:
    terminal = int(job["attempt"]) >= int(job["max_attempts"])
    status = "failed" if terminal else "queued"
    with connection() as conn:
        conn.execute(
            """
            UPDATE maintenance_jobs SET status=%s,available_at=now()+interval '15 minutes',
                worker_id=NULL,locked_at=NULL,updated_at=now(),error=%s,
                completed_at=CASE WHEN %s='failed' THEN now() ELSE completed_at END
            WHERE job_id=%s
            """,
            (status, error[-4000:], status, job["job_id"]),
        )
        conn.execute(
            """
            UPDATE maintenance_job_runs SET status='failed',completed_at=now(),error=%s
            WHERE job_id=%s AND attempt=%s
            """,
            (error[-4000:], job["job_id"], job["attempt"]),
        )


def run_worker(*, poll_seconds: int = 30, once: bool = False, stale_minutes: int = 30) -> int:
    worker_id = f"archive:{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
    ensure_schema()
    while True:
        try:
            requeue_stale(stale_minutes)
            enqueue_due()
            job = claim(worker_id)
            if job is None:
                if once:
                    return 0
                time.sleep(max(5, poll_seconds))
                continue
            stop = threading.Event()
            thread = threading.Thread(target=_heartbeat_loop, args=(stop, job["job_id"], worker_id), daemon=True)
            thread.start()
            try:
                if job["task_type"] != TASK_TYPE:
                    raise ValueError(f"unsupported task: {job['task_type']}")
                complete(job, archive_unused(job))
            except Exception as exc:
                fail(job, f"{type(exc).__name__}: {exc}")
            finally:
                stop.set()
                thread.join(timeout=2)
            if once:
                return 0
        except KeyboardInterrupt:
            return 130
        except Exception as exc:
            print(f"maintenance worker error: {type(exc).__name__}: {exc}", flush=True)
            if once:
                raise
            time.sleep(max(5, poll_seconds))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PostgreSQL maintenance queue.")
    sub = parser.add_subparsers(dest="command", required=True)
    register = sub.add_parser("register")
    register.add_argument("--interval-seconds", type=int, default=3600)
    register.add_argument("--older-than-hours", type=int, default=24)
    register.add_argument("--max-items", type=int, default=4)
    run = sub.add_parser("run")
    run.add_argument("--poll-seconds", type=int, default=30)
    run.add_argument("--stale-minutes", type=int, default=30)
    run.add_argument("--once", action="store_true")
    sub.add_parser("status")
    args = parser.parse_args(argv)
    if args.command == "register":
        register_schedule(interval_seconds=args.interval_seconds, older_than_hours=args.older_than_hours, max_items=args.max_items)
        enqueue_due()
        print(json.dumps({"status": "ok", "registered": True}))
        return 0
    if args.command == "run":
        return run_worker(poll_seconds=args.poll_seconds, once=args.once, stale_minutes=args.stale_minutes)
    ensure_schema()
    with connection() as conn:
        jobs = conn.execute(
            "SELECT job_id,task_type,status,attempt,max_attempts,available_at,started_at,completed_at,result_summary FROM maintenance_jobs ORDER BY job_id DESC LIMIT 10"
        ).fetchall()
        schedules = conn.execute("SELECT * FROM maintenance_schedules ORDER BY schedule_key").fetchall()
    print(json.dumps({"jobs": jobs, "schedules": schedules}, ensure_ascii=False, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
