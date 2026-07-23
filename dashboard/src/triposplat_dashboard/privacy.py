from __future__ import annotations

from typing import Any

from .db import connection


PREVIEW_SCHEMA = """
CREATE TABLE IF NOT EXISTS artifact_previews (
    id bigserial PRIMARY KEY,
    artifact_slug text NOT NULL REFERENCES artifacts(slug) ON DELETE CASCADE,
    filename text NOT NULL,
    mime_type text NOT NULL CHECK (mime_type IN ('image/png', 'image/jpeg', 'image/webp')),
    caption text NOT NULL DEFAULT '',
    content bytea NOT NULL,
    sha256 text NOT NULL,
    size_bytes bigint NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (artifact_slug, filename)
);
CREATE INDEX IF NOT EXISTS idx_artifact_previews_artifact
    ON artifact_previews(artifact_slug, id);
"""


def init_preview_schema() -> None:
    with connection() as conn:
        conn.execute(PREVIEW_SCHEMA, prepare=False)


def fetch_overview() -> dict[str, Any]:
    with connection() as conn:
        project = conn.execute(
            "SELECT slug,title,subtitle,phase,status,objective,progress,repository_url,updated_at "
            "FROM project_state ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
        milestones = conn.execute(
            "SELECT slug,title,stage,status,progress,summary,sort_order,updated_at "
            "FROM milestones ORDER BY sort_order,slug"
        ).fetchall()
        counts = conn.execute(
            """
            SELECT
              (SELECT count(*) FROM experiments) AS experiments,
              (SELECT count(*) FROM experiments WHERE status='passed') AS passed,
              (SELECT count(*) FROM artifacts WHERE state IN ('available','archived','local')) AS artifacts,
              (SELECT count(*) FROM documents) AS documents
            """
        ).fetchone()
        settings_rows = conn.execute("SELECT key,value FROM dashboard_settings").fetchall()
        queue = conn.execute(
            """
            SELECT task_type,status,attempt,max_attempts,available_at,started_at,completed_at,
                   result_summary,created_at,updated_at
            FROM maintenance_jobs
            ORDER BY job_id DESC LIMIT 1
            """
        ).fetchone() if _table_exists(conn, "maintenance_jobs") else None
        schedule = conn.execute(
            """
            SELECT schedule_key,interval_seconds,enabled,next_run_at,last_enqueued_at
            FROM maintenance_schedules ORDER BY schedule_key LIMIT 1
            """
        ).fetchone() if _table_exists(conn, "maintenance_schedules") else None
    return {
        "project": project,
        "milestones": milestones,
        "counts": counts,
        "settings": {row["key"]: row["value"] for row in settings_rows},
        "maintenance": {"latest_job": queue, "schedule": schedule},
    }


def _table_exists(conn, name: str) -> bool:
    row = conn.execute("SELECT to_regclass(%s) AS name", (name,)).fetchone()
    return bool(row and row["name"])


def fetch_experiments() -> list[dict[str, Any]]:
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT slug,title,category,variant,status,hardware,steps,duration_sec,
                   peak_rss_bytes,quality_status,notes,metadata,executed_at,updated_at
            FROM experiments ORDER BY executed_at DESC NULLS LAST,updated_at DESC
            """
        ).fetchall()
        metrics = conn.execute(
            "SELECT * FROM experiment_metrics ORDER BY experiment_slug,sort_order,name"
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
            """
            SELECT slug,title,kind,storage,state,size_bytes,checksum_state,
                   experiment_slug,description,updated_at
            FROM artifacts ORDER BY updated_at DESC,title
            """
        ).fetchall()
        documents = conn.execute(
            """
            SELECT slug,title,category,summary,language,updated_at
            FROM documents ORDER BY category,title
            """
        ).fetchall()
        previews = conn.execute(
            """
            SELECT id,artifact_slug,filename,mime_type,caption,size_bytes,sha256,created_at
            FROM artifact_previews ORDER BY artifact_slug,id
            """
        ).fetchall() if _table_exists(conn, "artifact_previews") else []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for preview in previews:
        preview["url"] = f"/api/previews/{preview['id']}"
        grouped.setdefault(preview["artifact_slug"], []).append(preview)
    for artifact in artifacts:
        artifact["previews"] = grouped.get(artifact["slug"], [])
    return {"artifacts": artifacts, "documents": documents}


def attach_preview(
    artifact_slug: str,
    *,
    filename: str,
    mime_type: str,
    caption: str,
    content: bytes,
    sha256: str,
) -> dict[str, Any]:
    if mime_type not in {"image/png", "image/jpeg", "image/webp"}:
        raise ValueError(f"unsupported preview MIME type: {mime_type}")
    init_preview_schema()
    with connection() as conn:
        row = conn.execute(
            """
            INSERT INTO artifact_previews
                (artifact_slug,filename,mime_type,caption,content,sha256,size_bytes,created_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,now())
            ON CONFLICT (artifact_slug,filename) DO UPDATE SET
                mime_type=excluded.mime_type,caption=excluded.caption,content=excluded.content,
                sha256=excluded.sha256,size_bytes=excluded.size_bytes,created_at=now()
            RETURNING id,artifact_slug,filename,mime_type,caption,size_bytes,sha256,created_at
            """,
            (artifact_slug, filename, mime_type, caption, content, sha256, len(content)),
        ).fetchone()
    return row


def fetch_preview(preview_id: int) -> dict[str, Any] | None:
    with connection() as conn:
        if not _table_exists(conn, "artifact_previews"):
            return None
        return conn.execute(
            "SELECT id,filename,mime_type,content,sha256,size_bytes FROM artifact_previews WHERE id=%s",
            (preview_id,),
        ).fetchone()


def fetch_activity(limit: int = 40) -> list[dict[str, Any]]:
    with connection() as conn:
        return conn.execute(
            "SELECT id,kind,title,detail,status,occurred_at FROM activity_log "
            "ORDER BY occurred_at DESC LIMIT %s",
            (limit,),
        ).fetchall()


def health() -> dict[str, str]:
    with connection() as conn:
        conn.execute("SELECT 1").fetchone()
    return {"status": "ok", "database": "postgresql"}
