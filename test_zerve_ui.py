"""
test_zerve_ui.py — Screenshot every tab of the Zerve Streamlit app using Playwright.

Saves PNG screenshots to output/screenshots/ and writes output/zerve_ui_report.md.

Usage:
    pip install playwright
    playwright install chromium
    python test_zerve_ui.py --url https://<your-name>.hub.zerve.cloud

Requirements:
    pip install playwright
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

OUT_DIR        = Path(__file__).resolve().parent / "output"
SCREENSHOT_DIR = OUT_DIR / "screenshots"
OUT_FILE       = OUT_DIR / "zerve_ui_report.md"
OUT_DIR.mkdir(exist_ok=True)
SCREENSHOT_DIR.mkdir(exist_ok=True)

TABS = [
    ("Overview",          0),
    ("Calibration & OLS", 1),
    ("Browse Markets",    2),
    ("Live Monitor",      3),
    ("Try the Detector",  4),
]

CHECKS = [
    # (tab_index, selector_or_text, description)
    (0, "10.97",          "Error multiplier shown"),
    (0, "22.3%",          "Retail error shown"),
    (0, "2.0%",           "Sophisticated error shown"),
    (0, "4,714",          "Market count shown"),
    (1, "OLS",            "OLS section present"),
    (1, "p < 0.001",      "p-value shown"),
    (1, "R²",             "R² shown"),
    (3, "Polymarket",     "Live alerts source shown"),
    (4, "retail",         "Detector result shown"),
]


def main(base_url: str) -> None:
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError:
        print("Playwright not installed. Run: pip install playwright && playwright install chromium")
        sys.exit(1)

    base_url = base_url.rstrip("/")
    now      = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [
        f"# Zerve Streamlit UI Report",
        f"_Generated: {now}_",
        f"_Target: {base_url}_\n",
        "> Automated Playwright screenshots of every tab.\n",
        "---\n",
    ]

    passed = 0
    failed = 0

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page    = browser.new_page(viewport={"width": 1400, "height": 900})

        print(f"Opening {base_url} ...")
        try:
            page.goto(base_url, timeout=60_000)
            page.wait_for_load_state("networkidle", timeout=60_000)
        except Exception as e:
            lines.append(f"❌ **Failed to load app:** `{e}`")
            _write(lines, OUT_FILE)
            browser.close()
            return

        # Screenshot landing page
        shot = SCREENSHOT_DIR / "00_landing.png"
        page.screenshot(path=str(shot), full_page=True)
        lines.append(f"## Landing Page\n")
        lines.append(f"![landing]({shot})\n")
        print(f"  Saved: {shot.name}")

        # Click each tab and screenshot
        for tab_name, tab_idx in TABS:
            lines.append(f"## Tab {tab_idx + 1}: {tab_name}\n")
            try:
                # Streamlit tab buttons are the nth [data-baseweb='tab'] element
                tabs = page.locator("[data-baseweb='tab']")
                tabs.nth(tab_idx).click()
                time.sleep(2)  # let tab content render
                page.wait_for_load_state("networkidle", timeout=15_000)

                fname = f"tab_{tab_idx + 1:02d}_{tab_name.replace(' ', '_').lower()}.png"
                shot  = SCREENSHOT_DIR / fname
                page.screenshot(path=str(shot), full_page=True)
                lines.append(f"![{tab_name}]({shot})\n")
                print(f"  Tab '{tab_name}' → {fname}")

                # Content checks for this tab
                for chk_tab, chk_text, chk_desc in CHECKS:
                    if chk_tab != tab_idx:
                        continue
                    content = page.content()
                    if chk_text.lower() in content.lower():
                        lines.append(f"- ✅ {chk_desc} (`{chk_text}`)")
                        passed += 1
                    else:
                        lines.append(f"- ❌ {chk_desc} — `{chk_text}` NOT FOUND in page")
                        failed += 1

                lines.append("")

            except Exception as e:
                lines.append(f"- ❌ **Tab failed:** `{e}`\n")
                failed += 1

        # Tab 4: type a topic and screenshot result
        lines.append("## Detector Test (Tab 5 — type 'gamestop')\n")
        try:
            tabs = page.locator("[data-baseweb='tab']")
            tabs.nth(4).click()
            time.sleep(1)
            text_input = page.locator("input[type='text']").first
            text_input.fill("gamestop")
            text_input.press("Enter")
            time.sleep(3)
            shot = SCREENSHOT_DIR / "tab_05_detector_gamestop.png"
            page.screenshot(path=str(shot), full_page=True)
            lines.append(f"![detector gamestop]({shot})\n")
            content = page.content()
            if "retail" in content.lower():
                lines.append("- ✅ 'retail' classification visible after entering 'gamestop'")
                passed += 1
            else:
                lines.append("- ❌ Expected 'retail' in result — not found")
                failed += 1
        except Exception as e:
            lines.append(f"- ❌ Detector test failed: `{e}`")
            failed += 1

        browser.close()

    # Summary
    lines.append("\n---\n")
    lines.append("## Summary\n")
    lines.append(f"- **Screenshots saved to:** `output/screenshots/`")
    lines.append(f"- **Checks passed:** {passed}")
    lines.append(f"- **Checks failed:** {failed}")
    if failed == 0:
        lines.append("\n✅ **All UI checks passed.**")
    else:
        lines.append(f"\n⚠ **{failed} check(s) failed.**")

    _write(lines, OUT_FILE)
    print(f"\nUI report: {OUT_FILE}")
    print(f"Screenshots: {SCREENSHOT_DIR}")
    print(f"Passed: {passed} | Failed: {failed}")


def _write(lines: list[str], path: Path) -> None:
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Screenshot Zerve Streamlit UI with Playwright.")
    parser.add_argument("--url", default=os.environ.get("ZERVE_STREAMLIT_URL", ""), help="Streamlit app URL")
    args = parser.parse_args()

    url = args.url.strip()
    if not url:
        url = input("Enter Zerve Streamlit URL (e.g. https://accuracy-trap.hub.zerve.cloud): ").strip()
    if not url:
        print("No URL provided.")
        sys.exit(1)

    main(url)
