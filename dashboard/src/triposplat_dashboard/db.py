from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


DEFAULT_DSN = "host=127.0.0.1 port=5432 dbname=triposplat_dashboard user=triposplat_app"
SCHEMA_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


def dsn() -> str:
    return os.environ.get("TRIPOSPLAT_DASHBOARD_DSN", DEFAULT_DSN)


def schema() -> str:
    value = os.environ.get("TRIPOSPLAT_DASHBOARD_SCHEMA", "public")
    if SCHEMA_RE.fullmatch(value) is None:
        raise ValueError(f"invalid PostgreSQL schema: {value!r}")
    return value


@contextmanager
def connection(*, autocommit: bool = False) -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(
        dsn(),
        connect_timeout=10,
        application_name="triposplat_dashboard",
        row_factory=dict_row,
        autocommit=autocommit,
    )
    conn.execute(f'SET search_path TO "{schema()}"')
    try:
        yield conn
        if not autocommit:
            conn.commit()
    except Exception:
        if not autocommit:
            conn.rollback()
        raise
    finally:
        conn.close()


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS project_state (
    slug text PRIMARY KEY,
    title text NOT NULL,
    subtitle text NOT NULL DEFAULT '',
    phase text NOT NULL,
    status text NOT NULL CHECK (status IN ('active', 'paused', 'blocked', 'complete')),
    objective text NOT NULL,
    progress smallint NOT NULL CHECK (progress BETWEEN 0 AND 100),
    repository_url text,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS milestones (
    slug text PRIMARY KEY,
    title text NOT NULL,
    stage text NOT NULL,
    status text NOT NULL CHECK (status IN ('pending', 'running', 'review', 'complete', 'blocked')),
    progress smallint NOT NULL CHECK (progress BETWEEN 0 AND 100),
    summary text NOT NULL DEFAULT '',
    sort_order integer NOT NULL DEFAULT 0,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS optimization_tasks (
    task_key text PRIMARY KEY,
    goal_key text NOT NULL,
    sequence_no integer NOT NULL,
    title text NOT NULL,
    objective text NOT NULL,
    status text NOT NULL CHECK (status IN ('pending','running','review','complete','rejected','blocked')),
    priority integer NOT NULL,
    depends_on text[] NOT NULL DEFAULT '{}',
    acceptance_criteria jsonb NOT NULL DEFAULT '{}'::jsonb,
    result_summary jsonb NOT NULL DEFAULT '{}'::jsonb,
    started_at timestamptz,
    completed_at timestamptz,
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_optimization_tasks_ready
    ON optimization_tasks(status, priority DESC, sequence_no);

CREATE TABLE IF NOT EXISTS experiments (
    slug text PRIMARY KEY,
    title text NOT NULL,
    category text NOT NULL,
    variant text NOT NULL,
    status text NOT NULL CHECK (status IN ('planned', 'running', 'passed', 'rejected', 'inconclusive')),
    hardware text NOT NULL,
    steps integer,
    duration_sec double precision,
    peak_rss_bytes bigint,
    quality_status text NOT NULL DEFAULT 'unknown',
    notes text NOT NULL DEFAULT '',
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    executed_at timestamptz,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS experiment_metrics (
    experiment_slug text NOT NULL REFERENCES experiments(slug) ON DELETE CASCADE,
    name text NOT NULL,
    label text NOT NULL,
    value double precision NOT NULL,
    unit text NOT NULL DEFAULT '',
    baseline_value double precision,
    direction text NOT NULL DEFAULT 'neutral' CHECK (direction IN ('lower', 'higher', 'neutral')),
    sort_order integer NOT NULL DEFAULT 0,
    PRIMARY KEY (experiment_slug, name)
);

CREATE TABLE IF NOT EXISTS artifacts (
    slug text PRIMARY KEY,
    title text NOT NULL,
    kind text NOT NULL,
    location text NOT NULL,
    storage text NOT NULL,
    state text NOT NULL CHECK (state IN ('available', 'archived', 'local', 'pending', 'missing')),
    size_bytes bigint,
    checksum_state text NOT NULL DEFAULT 'unknown',
    experiment_slug text REFERENCES experiments(slug) ON DELETE SET NULL,
    description text NOT NULL DEFAULT '',
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documents (
    slug text PRIMARY KEY,
    title text NOT NULL,
    category text NOT NULL,
    path text NOT NULL,
    summary text NOT NULL DEFAULT '',
    language text NOT NULL DEFAULT 'ja',
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS activity_log (
    id bigserial PRIMARY KEY,
    event_key text UNIQUE,
    kind text NOT NULL,
    title text NOT NULL,
    detail text NOT NULL DEFAULT '',
    status text NOT NULL DEFAULT 'info',
    occurred_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS system_snapshots (
    id bigserial PRIMARY KEY,
    host text NOT NULL,
    cpu_model text NOT NULL DEFAULT '',
    cpu_threads integer,
    memory_total_bytes bigint,
    memory_available_bytes bigint,
    workspace_total_bytes bigint,
    workspace_available_bytes bigint,
    dashboard_status text NOT NULL DEFAULT 'unknown',
    captured_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS dashboard_settings (
    key text PRIMARY KEY,
    value jsonb NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_milestones_order ON milestones(sort_order, slug);
CREATE INDEX IF NOT EXISTS idx_experiments_time ON experiments(executed_at DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_artifacts_updated ON artifacts(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_activity_time ON activity_log(occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_system_time ON system_snapshots(captured_at DESC);
"""


def init_schema() -> None:
    with connection() as conn:
        conn.execute(SCHEMA_SQL, prepare=False)


def _timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def seed(seed_path: Path) -> dict[str, int]:
    payload = json.loads(seed_path.read_text(encoding="utf-8"))
    init_schema()
    counts: dict[str, int] = {}
    with connection() as conn:
        project = payload["project"]
        conn.execute(
            """
            INSERT INTO project_state
                (slug, title, subtitle, phase, status, objective, progress, repository_url, updated_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,now())
            ON CONFLICT (slug) DO UPDATE SET
                title=excluded.title, subtitle=excluded.subtitle, phase=excluded.phase,
                status=excluded.status, objective=excluded.objective, progress=excluded.progress,
                repository_url=excluded.repository_url, updated_at=now()
            """,
            (
                project["slug"], project["title"], project.get("subtitle", ""),
                project["phase"], project["status"], project["objective"],
                project["progress"], project.get("repository_url"),
            ),
        )
        counts["project"] = 1

        for item in payload.get("milestones", []):
            conn.execute(
                """
                INSERT INTO milestones (slug,title,stage,status,progress,summary,sort_order,updated_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,now())
                ON CONFLICT (slug) DO UPDATE SET title=excluded.title, stage=excluded.stage,
                    status=excluded.status, progress=excluded.progress, summary=excluded.summary,
                    sort_order=excluded.sort_order, updated_at=now()
                """,
                (item["slug"], item["title"], item["stage"], item["status"],
                 item["progress"], item.get("summary", ""), item.get("sort_order", 0)),
            )
        counts["milestones"] = len(payload.get("milestones", []))

        for item in payload.get("experiments", []):
            conn.execute(
                """
                INSERT INTO experiments
                    (slug,title,category,variant,status,hardware,steps,duration_sec,peak_rss_bytes,
                     quality_status,notes,metadata,executed_at,updated_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now())
                ON CONFLICT (slug) DO UPDATE SET title=excluded.title, category=excluded.category,
                    variant=excluded.variant, status=excluded.status, hardware=excluded.hardware,
                    steps=excluded.steps, duration_sec=excluded.duration_sec,
                    peak_rss_bytes=excluded.peak_rss_bytes, quality_status=excluded.quality_status,
                    notes=excluded.notes, metadata=excluded.metadata,
                    executed_at=excluded.executed_at, updated_at=now()
                """,
                (item["slug"], item["title"], item["category"], item["variant"],
                 item["status"], item["hardware"], item.get("steps"), item.get("duration_sec"),
                 item.get("peak_rss_bytes"), item.get("quality_status", "unknown"),
                 item.get("notes", ""), Jsonb(item.get("metadata", {})),
                 _timestamp(item.get("executed_at"))),
            )
            conn.execute("DELETE FROM experiment_metrics WHERE experiment_slug=%s", (item["slug"],))
            for order, metric in enumerate(item.get("metrics", [])):
                conn.execute(
                    """
                    INSERT INTO experiment_metrics
                        (experiment_slug,name,label,value,unit,baseline_value,direction,sort_order)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (item["slug"], metric["name"], metric["label"], metric["value"],
                     metric.get("unit", ""), metric.get("baseline_value"),
                     metric.get("direction", "neutral"), order),
                )
        counts["experiments"] = len(payload.get("experiments", []))

        for item in payload.get("artifacts", []):
            conn.execute(
                """
                INSERT INTO artifacts
                    (slug,title,kind,location,storage,state,size_bytes,checksum_state,
                     experiment_slug,description,updated_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now())
                ON CONFLICT (slug) DO UPDATE SET title=excluded.title, kind=excluded.kind,
                    location=excluded.location, storage=excluded.storage, state=excluded.state,
                    size_bytes=excluded.size_bytes, checksum_state=excluded.checksum_state,
                    experiment_slug=excluded.experiment_slug, description=excluded.description,
                    updated_at=now()
                """,
                (item["slug"], item["title"], item["kind"], item["location"],
                 item["storage"], item["state"], item.get("size_bytes"),
                 item.get("checksum_state", "unknown"), item.get("experiment_slug"),
                 item.get("description", "")),
            )
        counts["artifacts"] = len(payload.get("artifacts", []))

        for item in payload.get("documents", []):
            conn.execute(
                """
                INSERT INTO documents (slug,title,category,path,summary,language,updated_at)
                VALUES (%s,%s,%s,%s,%s,%s,now())
                ON CONFLICT (slug) DO UPDATE SET title=excluded.title,
                    category=excluded.category, path=excluded.path, summary=excluded.summary,
                    language=excluded.language, updated_at=now()
                """,
                (item["slug"], item["title"], item["category"], item["path"],
                 item.get("summary", ""), item.get("language", "ja")),
            )
        counts["documents"] = len(payload.get("documents", []))

        for item in payload.get("activities", []):
            conn.execute(
                """
                INSERT INTO activity_log (event_key,kind,title,detail,status,occurred_at)
                VALUES (%s,%s,%s,%s,%s,%s)
                ON CONFLICT (event_key) DO UPDATE SET kind=excluded.kind, title=excluded.title,
                    detail=excluded.detail, status=excluded.status, occurred_at=excluded.occurred_at
                """,
                (item["event_key"], item["kind"], item["title"], item.get("detail", ""),
                 item.get("status", "info"), _timestamp(item.get("occurred_at")) or datetime.now(UTC)),
            )
        counts["activities"] = len(payload.get("activities", []))

        for key, value in payload.get("settings", {}).items():
            conn.execute(
                """
                INSERT INTO dashboard_settings (key,value,updated_at) VALUES (%s,%s,now())
                ON CONFLICT (key) DO UPDATE SET value=excluded.value, updated_at=now()
                """,
                (key, Jsonb(value)),
            )
        counts["settings"] = len(payload.get("settings", {}))
    return counts


def fetch_overview() -> dict[str, Any]:
    with connection() as conn:
        project = conn.execute("SELECT * FROM project_state ORDER BY updated_at DESC LIMIT 1").fetchone()
        milestones = conn.execute("SELECT * FROM milestones ORDER BY sort_order, slug").fetchall()
        counts = conn.execute(
            """
            SELECT
              (SELECT count(*) FROM experiments) AS experiments,
              (SELECT count(*) FROM experiments WHERE status='passed') AS passed,
              (SELECT count(*) FROM artifacts WHERE state IN ('available','archived','local')) AS artifacts,
              (SELECT count(*) FROM documents) AS documents
            """
        ).fetchone()
        latest = conn.execute(
            "SELECT * FROM system_snapshots ORDER BY captured_at DESC LIMIT 1"
        ).fetchone()
        settings_rows = conn.execute("SELECT key,value FROM dashboard_settings").fetchall()
    return {
        "project": project,
        "milestones": milestones,
        "counts": counts,
        "system": latest,
        "settings": {row["key"]: row["value"] for row in settings_rows},
    }


def fetch_experiments() -> list[dict[str, Any]]:
    with connection() as conn:
        rows = conn.execute(
            "SELECT * FROM experiments ORDER BY executed_at DESC NULLS LAST, updated_at DESC"
        ).fetchall()
        metrics = conn.execute(
            "SELECT * FROM experiment_metrics ORDER BY experiment_slug, sort_order, name"
        ).fetchall()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        grouped.setdefault(metric["experiment_slug"], []).append(metric)
    for row in rows:
        row["metrics"] = grouped.get(row["slug"], [])
    return rows


def fetch_artifacts() -> dict[str, list[dict[str, Any]]]:
    with connection() as conn:
        artifacts = conn.execute(
            "SELECT * FROM artifacts ORDER BY updated_at DESC, title"
        ).fetchall()
        documents = conn.execute(
            "SELECT * FROM documents ORDER BY category, title"
        ).fetchall()
    return {"artifacts": artifacts, "documents": documents}


def fetch_activity(limit: int = 40) -> list[dict[str, Any]]:
    with connection() as conn:
        return conn.execute(
            "SELECT * FROM activity_log ORDER BY occurred_at DESC LIMIT %s", (limit,)
        ).fetchall()


def record_system_snapshot(snapshot: dict[str, Any]) -> None:
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO system_snapshots
                (host,cpu_model,cpu_threads,memory_total_bytes,memory_available_bytes,
                 workspace_total_bytes,workspace_available_bytes,dashboard_status,captured_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,now())
            """,
            (snapshot["host"], snapshot.get("cpu_model", ""), snapshot.get("cpu_threads"),
             snapshot.get("memory_total_bytes"), snapshot.get("memory_available_bytes"),
             snapshot.get("workspace_total_bytes"), snapshot.get("workspace_available_bytes"),
             snapshot.get("dashboard_status", "running")),
        )


def health() -> dict[str, Any]:
    started = datetime.now(UTC)
    with connection() as conn:
        row = conn.execute("SELECT current_database() AS database, now() AS server_time").fetchone()
    return {
        "status": "ok",
        "database": "postgresql",
        "database_name": row["database"],
        "server_time": row["server_time"],
        "checked_at": started,
    }
