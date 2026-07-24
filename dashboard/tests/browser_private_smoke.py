#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


FORBIDDEN = (
    "/workspace", "213.173.", "127.0.0.1", "cdaa59d20af0", "AMD EPYC",
    "gdrive", "google drive", "postgresql", "queue", "worker", "rclone",
    "lease", "heartbeat", "checksum", "database",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:10101")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/triposplat-dashboard-private-ui"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"url": args.url, "viewports": {}, "console_errors": []}

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        for name, viewport in (("desktop", {"width": 1440, "height": 1000}), ("mobile", {"width": 390, "height": 844})):
            page = browser.new_page(viewport=viewport)
            page.on("console", lambda msg: report["console_errors"].append(msg.text) if msg.type == "error" else None)
            page.goto(args.url, wait_until="networkidle")
            page.locator("#journey .journey-item").first.wait_for(state="visible")
            assert page.locator(".nav-button").count() == 3
            assert page.locator("#journey .journey-item").count() == 9
            assert "非線形24-bit" in page.locator(".nf24-note").inner_text()
            body = page.locator("body").inner_text()
            body_lower = body.lower()
            assert not any(value.lower() in body_lower for value in FORBIDDEN)

            overview_text = page.request.get(f"{args.url}/api/overview").text()
            artifacts_text = page.request.get(f"{args.url}/api/artifacts").text()
            health_text = page.request.get(f"{args.url}/api/health").text()
            public_responses = (overview_text + artifacts_text + health_text).lower()
            assert not any(value.lower() in public_responses for value in FORBIDDEN)
            assert '"location"' not in artifacts_text and '"path"' not in artifacts_text
            assert '"storage"' not in artifacts_text and '"checksum_state"' not in artifacts_text
            assert '"task_type"' not in overview_text and '"schedule_key"' not in overview_text

            page.locator('[data-view="artifacts"]').click()
            assert page.locator("#visual-gallery .preview-card").count() == 3
            page.locator("#visual-gallery img").first.scroll_into_view_if_needed()
            page.wait_for_function("() => document.querySelector('#visual-gallery img').naturalWidth > 0")
            assert page.locator("#visual-gallery img").first.evaluate("img => img.naturalWidth") > 0
            page.locator("#visual-gallery .preview-card").first.click()
            assert page.locator("#preview-dialog").evaluate("dialog => dialog.open") is True
            page.locator("#dialog-close").click()
            page.locator('[data-view="overview"]').click()
            screenshot = args.output_dir / f"{name}.png"
            page.screenshot(path=screenshot, full_page=True)
            report["viewports"][name] = {
                "viewport": viewport,
                "body_width": page.locator("body").evaluate("node => node.scrollWidth"),
                "client_width": page.locator("body").evaluate("node => node.clientWidth"),
                "screenshot": str(screenshot),
            }
            page.close()
        browser.close()

    report["ok"] = not report["console_errors"] and all(
        item["body_width"] == item["client_width"] for item in report["viewports"].values()
    )
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
