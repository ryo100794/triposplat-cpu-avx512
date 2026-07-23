from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
from pathlib import Path

from . import privacy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Store generated previews in PostgreSQL.")
    parser.add_argument("artifact_slug")
    parser.add_argument("path", type=Path)
    parser.add_argument("--caption", default="")
    args = parser.parse_args(argv)
    content = args.path.read_bytes()
    mime_type = mimetypes.guess_type(args.path.name)[0] or "application/octet-stream"
    result = privacy.attach_preview(
        args.artifact_slug,
        filename=args.path.name,
        mime_type=mime_type,
        caption=args.caption,
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
    )
    print(json.dumps({"status": "ok", "preview": result}, ensure_ascii=False, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
