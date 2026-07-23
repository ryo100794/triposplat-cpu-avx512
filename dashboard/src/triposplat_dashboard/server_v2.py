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

from . import privacy_v2 as data


ROOT = Path(__file__).resolve().parents[2]
STATIC_ROOT = ROOT / "static"


def _json_default(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(type(value).__name__)


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 32


class Handler(BaseHTTPRequestHandler):
    server_version = "TripoSplatDashboard/0.2"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.log_date_time_string()} {fmt % args}", flush=True)

    def _headers(self, content_type: str, length: int, cache: str = "no-store") -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; frame-src 'none'",
        )

    def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=_json_default, separators=(",", ":")).encode()
        self.send_response(status)
        self._headers("application/json; charset=utf-8", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json({"status": "error", "message": message}, status)

    def _static(self, name: str) -> None:
        path = (STATIC_ROOT / name).resolve()
        try:
            path.relative_to(STATIC_ROOT.resolve())
        except ValueError:
            self._error(HTTPStatus.FORBIDDEN, "invalid path")
            return
        if not path.is_file():
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return
        body = path.read_bytes()
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if mime.startswith("text/") or mime == "application/javascript":
            mime += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self._headers(mime, len(body), "public, max-age=300")
        self.end_headers()
        self.wfile.write(body)

    def _preview(self, preview_id: int) -> None:
        preview = data.fetch_preview(preview_id)
        if preview is None:
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return
        body = bytes(preview["content"])
        self.send_response(HTTPStatus.OK)
        self._headers(preview["mime_type"], len(body), "public, max-age=86400, immutable")
        self.send_header("Content-Disposition", f"inline; filename=\"{preview['filename']}\"")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path == "/api/health":
                self._json(data.health())
            elif path == "/api/overview":
                self._json(data.fetch_overview())
            elif path == "/api/experiments":
                self._json({"experiments": data.fetch_experiments()})
            elif path == "/api/artifacts":
                self._json(data.fetch_artifacts())
            elif path == "/api/activity":
                self._json({"activity": data.fetch_activity()})
            elif path.startswith("/api/previews/"):
                value = path.removeprefix("/api/previews/")
                self._preview(int(value)) if value.isdigit() else self._error(HTTPStatus.NOT_FOUND, "not found")
            elif path in {"/", "/index.html"}:
                self._static("index_private.html")
            elif path.startswith("/static/"):
                self._static(path.removeprefix("/static/"))
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")
        except Exception as exc:
            self.log_error("request failed: %s", exc)
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "data temporarily unavailable")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("TRIPOSPLAT_DASHBOARD_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("TRIPOSPLAT_DASHBOARD_PORT", "10101")))
    args = parser.parse_args(argv)
    server = DashboardServer((args.host, args.port), Handler)
    print(f"TripoSplat dashboard listening on port {args.port} using PostgreSQL", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
