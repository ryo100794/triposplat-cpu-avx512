#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:10101")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/triposplat-dashboard-ui"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"url": args.url, "viewports": {}, "console_errors": []}

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        for name, viewport in (("desktop", {"width": 1440, "height": 1000}), ("mobile", {"width": 390, "height": 844})):
            page = browser.new_page(viewport=viewport, device_scale_factor=1)
            page.on(
                "console",
                lambda message: report["console_errors"].append(message.text)
                if message.type == "error" else None,
            )
            page.goto(args.url, wait_until="networkidle")
            page.locator("#metric-experiments").wait_for(state="visible")
            assert page.locator("#metric-experiments").inner_text() == "8"
            assert "PostgreSQL" in page.locator("#connection-label").inner_text()
            page.locator('[data-view="experiments"]').click()
            assert page.locator("#view-experiments").is_visible()
            assert page.locator("#experiment-rows tr").count() == 8
            page.locator('[data-filter="rejected"]').click()
            assert page.locator("#experiment-rows tr").count() == 3
            page.locator('[data-view="artifacts"]').click()
            assert page.locator("#artifact-list .artifact-row").count() == 4
            page.locator("#artifact-search").fill("GitHub")
            assert page.locator("#artifact-list .artifact-row").count() == 1
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
        item["body_width"] == item["client_width"]
        for item in report["viewports"].values()
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
