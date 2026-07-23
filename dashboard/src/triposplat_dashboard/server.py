from __future__ import annotations

import argparse
import json
import mimetypes
import os
from datetime import date, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from . import db


ROOT = Path(__file__).resolve().parents[2]
STATIC_ROOT = ROOT / "static"


def _json_default(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"cannot serialize {type(value).__name__}")


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "TripoSplatDashboard/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.log_date_time_string()} {self.client_address[0]} {fmt % args}", flush=True)

    def _headers(self, content_type: str, length: int, *, cache: str = "no-store") -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; img-src 'self' data:; connect-src 'self'")

    def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=_json_default, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self._headers("application/json; charset=utf-8", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json({"status": "error", "message": message}, status)

    def _static(self, relative: str) -> None:
        requested = (STATIC_ROOT / relative).resolve()
        try:
            requested.relative_to(STATIC_ROOT.resolve())
        except ValueError:
            self._error(HTTPStatus.FORBIDDEN, "invalid path")
            return
        if not requested.is_file():
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return
        body = requested.read_bytes()
        content_type = mimetypes.guess_type(requested.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
            content_type += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self._headers(content_type, len(body), cache="public, max-age=300")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path == "/api/health":
                self._json(db.health())
            elif path == "/api/overview":
                self._json(db.fetch_overview())
            elif path == "/api/experiments":
                self._json({"experiments": db.fetch_experiments()})
            elif path == "/api/artifacts":
                self._json(db.fetch_artifacts())
            elif path == "/api/activity":
                self._json({"activity": db.fetch_activity()})
            elif path in {"/", "/index.html"}:
                self._static("index.html")
            elif path.startswith("/static/"):
                self._static(path.removeprefix("/static/"))
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")
        except Exception as exc:
            self.log_error("request failed: %s", exc)
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "dashboard data is temporarily unavailable")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the TripoSplat PostgreSQL dashboard.")
    parser.add_argument("--host", default=os.environ.get("TRIPOSPLAT_DASHBOARD_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("TRIPOSPLAT_DASHBOARD_PORT", "10101")))
    parser.add_argument("--init-schema", action="store_true")
    args = parser.parse_args(argv)
    if args.init_schema:
        db.init_schema()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    server.daemon_threads = True
    print(f"TripoSplat dashboard listening on http://{args.host}:{args.port} using PostgreSQL", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
