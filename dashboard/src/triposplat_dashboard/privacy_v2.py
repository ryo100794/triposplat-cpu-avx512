from __future__ import annotations

from typing import Any

from . import privacy
from .db import connection


def fetch_overview() -> dict[str, Any]:
    data = privacy.fetch_overview()
    with connection() as conn:
        exists = conn.execute("SELECT to_regclass('project_timeline') AS name").fetchone()
        data["timeline"] = conn.execute(
            """
            SELECT slug,occurred_at,phase,title,summary,metric_label,metric_value,
                   metric_unit,status,sort_order
            FROM project_timeline ORDER BY sort_order,occurred_at
            """
        ).fetchall() if exists and exists["name"] else []
    return data


def fetch_experiments() -> list[dict[str, Any]]:
    rows = privacy.fetch_experiments()
    for row in rows:
        row.pop("hardware", None)
    return rows


fetch_artifacts = privacy.fetch_artifacts
fetch_preview = privacy.fetch_preview
fetch_activity = privacy.fetch_activity
health = privacy.health
