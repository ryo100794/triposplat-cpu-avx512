from __future__ import annotations

from typing import Any

from . import privacy
from .db import connection


PUBLIC_TIMELINE_COPY = {
    "work-dashboard": {
        "title": "進捗・成果物画面を公開",
        "summary": "実験比較、画像成果物、工程別の定量指標を一画面に統合。",
    },
    "maintenance-queue": {
        "slug": "automatic-preservation",
        "phase": "PRESERVATION",
        "title": "成果物の自動保管を開始",
        "summary": "未使用成果物の確認、整合性検証、保管完了後の整理を自動化。",
    },
}


def _public_retention(data: dict[str, Any]) -> dict[str, Any]:
    maintenance = data.pop("maintenance", {}) or {}
    job = maintenance.get("latest_job") or {}
    schedule = maintenance.get("schedule") or {}
    summary = job.get("result_summary") or {}
    return {
        "status": job.get("status", "pending"),
        "cycle_minutes": round(schedule.get("interval_seconds", 0) / 60),
        "recent_items": summary.get("archived_count", 0),
        "recent_bytes": summary.get("archived_bytes", 0),
    }


def fetch_overview() -> dict[str, Any]:
    data = privacy.fetch_overview()
    data["retention"] = _public_retention(data)
    with connection() as conn:
        exists = conn.execute("SELECT to_regclass('project_timeline') AS name").fetchone()
        timeline = conn.execute(
            """
            SELECT slug,occurred_at,phase,title,summary,metric_label,metric_value,
                   metric_unit,status,sort_order
            FROM project_timeline ORDER BY sort_order,occurred_at
            """
        ).fetchall() if exists and exists["name"] else []
    for item in timeline:
        item.update(PUBLIC_TIMELINE_COPY.get(item["slug"], {}))
    data["timeline"] = timeline
    return data


def fetch_experiments() -> list[dict[str, Any]]:
    rows = privacy.fetch_experiments()
    for row in rows:
        row.pop("hardware", None)
    return rows


def fetch_artifacts() -> dict[str, list[dict[str, Any]]]:
    data = privacy.fetch_artifacts()
    public_descriptions = {
        "high-memory-audit-runs": "高メモリ候補の反復計測と検証結果。",
        "public-repository": "実装、再現手順、評価文書。",
        "nf24-checkpoint": "事前変換済みモデルデータ。",
        "reference-renderer": "公式レンダラーを使った品質比較基準。",
    }
    allowed = {
        "slug", "title", "kind", "state", "size_bytes", "experiment_slug",
        "description", "updated_at", "previews",
    }
    artifacts = []
    for source in data["artifacts"]:
        item = {key: value for key, value in source.items() if key in allowed}
        item["description"] = public_descriptions.get(item["slug"], item.get("description", ""))
        artifacts.append(item)
    return {"artifacts": artifacts, "documents": data["documents"]}


fetch_preview = privacy.fetch_preview


def health() -> dict[str, str]:
    privacy.health()
    return {"status": "ok"}
