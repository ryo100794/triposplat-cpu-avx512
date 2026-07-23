from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .db import connection


SCHEMA = """
CREATE TABLE IF NOT EXISTS project_timeline (
    slug text PRIMARY KEY,
    occurred_at timestamptz NOT NULL,
    phase text NOT NULL,
    title text NOT NULL,
    summary text NOT NULL DEFAULT '',
    metric_label text NOT NULL DEFAULT '',
    metric_value double precision,
    metric_unit text NOT NULL DEFAULT '',
    status text NOT NULL CHECK (status IN ('complete','running','planned')),
    sort_order integer NOT NULL DEFAULT 0,
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_project_timeline_order
    ON project_timeline(sort_order,occurred_at);
"""


def seed(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    with connection() as conn:
        conn.execute(SCHEMA, prepare=False)
        for item in payload:
            conn.execute(
                """
                INSERT INTO project_timeline
                    (slug,occurred_at,phase,title,summary,metric_label,metric_value,
                     metric_unit,status,sort_order,updated_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now())
                ON CONFLICT (slug) DO UPDATE SET occurred_at=excluded.occurred_at,
                    phase=excluded.phase,title=excluded.title,summary=excluded.summary,
                    metric_label=excluded.metric_label,metric_value=excluded.metric_value,
                    metric_unit=excluded.metric_unit,status=excluded.status,
                    sort_order=excluded.sort_order,updated_at=now()
                """,
                (
                    item["slug"], datetime.fromisoformat(item["occurred_at"].replace("Z", "+00:00")),
                    item["phase"], item["title"], item.get("summary", ""),
                    item.get("metric_label", ""), item.get("metric_value"),
                    item.get("metric_unit", ""), item["status"], item.get("sort_order", 0),
                ),
            )
    return len(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps({"status": "ok", "timeline": seed(args.path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
