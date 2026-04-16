"""
test_zerve_api.py — Hit all Zerve FastAPI endpoints and save a report locally.

Usage:
    python test_zerve_api.py --url https://<your-name>.hub.zerve.cloud
    python test_zerve_api.py  # uses ZERVE_API_URL env var or prompts

Output: output/zerve_api_report.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUT_DIR  = Path(__file__).resolve().parent / "output"
OUT_FILE = OUT_DIR / "zerve_api_report.md"
OUT_DIR.mkdir(exist_ok=True)

TIMEOUT = 20  # seconds per request

ENDPOINTS = [
    ("GET", "/health",                    None,                          "Health check"),
    ("GET", "/accuracy-trap",             None,                          "Accuracy Trap data"),
    ("GET", "/lag",                       {"category": "political"},     "Lag — political"),
    ("GET", "/lag",                       {"category": "sports"},        "Lag — sports"),
    ("GET", "/lag",                       {"category": "crypto"},        "Lag — crypto"),
    ("GET", "/lag",                       {"category": "economic"},      "Lag — economic"),
    ("GET", "/lag",                       {"category": "climate"},       "Lag — climate"),
    ("GET", "/classify",                  {"topic": "gamestop"},         "Classify — gamestop (retail, validated)"),
    ("GET", "/classify",                  {"topic": "bitcoin"},          "Classify — bitcoin (institutional, validated)"),
    ("GET", "/classify",                  {"topic": "trump election"},   "Classify — trump election"),
    ("GET", "/classify",                  {"topic": "fed rate hike"},    "Classify — fed rate hike"),
    ("GET", "/classify",                  {"topic": "super bowl"},       "Classify — super bowl"),
    ("GET", "/live-alerts",               None,                          "Live alerts (Polymarket + Trends)"),
    ("GET", "/markets",                   None,                          "Markets dataset — all 200"),
    ("GET", "/markets",                   {"category": "Politics"},      "Markets — Politics filter"),
    ("GET", "/markets",                   {"market_type": "Retail Flood", "limit": 5}, "Markets — Retail Flood filter"),
    ("GET", "/explain",                   {"topic": "gamestop"},         "Explain — gamestop"),
    ("GET", "/explain",                   {"topic": "bitcoin"},          "Explain — bitcoin"),
]

# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------
lines: list[str] = []

def h(text: str, level: int = 2) -> None:
    lines.append(f"\n{'#' * level} {text}\n")

def p(text: str) -> None:
    lines.append(text)

def ok(label: str, value: str) -> None:
    lines.append(f"- ✅ **{label}:** {value}")

def warn(label: str, value: str) -> None:
    lines.append(f"- ⚠ **{label}:** {value}")

def fail(label: str, value: str) -> None:
    lines.append(f"- ❌ **{label}:** {value}")

def code(text: str) -> None:
    lines.append(f"```json\n{text}\n```")

def rule() -> None:
    lines.append("\n---\n")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(base_url: str) -> None:
    base_url = base_url.rstrip("/")
    now      = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines.append(f"# Zerve API Diagnostic Report")
    lines.append(f"_Generated: {now}_")
    lines.append(f"_Target: {base_url}_\n")
    lines.append("> Hits every endpoint, checks status codes and key fields.\n")
    rule()

    total = 0
    passed = 0
    failed = 0

    for method, path, params, label in ENDPOINTS:
        total += 1
        url = base_url + path
        h(f"{label}  `{method} {path}`", 3)

        try:
            t0   = time.monotonic()
            resp = requests.request(method, url, params=params, timeout=TIMEOUT)
            ms   = int((time.monotonic() - t0) * 1000)

            status_ok = resp.status_code == 200
            if status_ok:
                passed += 1
                ok("Status", f"{resp.status_code} ({ms}ms)")
            else:
                failed += 1
                fail("Status", f"{resp.status_code} ({ms}ms)")
                p(f"Response body: `{resp.text[:300]}`")
                continue

            try:
                data = resp.json()
            except Exception:
                warn("Parse", "Response is not valid JSON")
                p(f"Raw: `{resp.text[:300]}`")
                continue

            # ── Endpoint-specific checks ──────────────────────────────────
            if path == "/health":
                status = data.get("status")
                if status == "ok":
                    ok("status field", status)
                else:
                    warn("status field", str(status))
                source = data.get("data_source", "—")
                ok("data_source", source)

            elif path == "/accuracy-trap":
                hl = data.get("headline", {})
                ok("error_multiplier",    f"{hl.get('error_multiplier')}×")
                ok("retail_error",        f"{hl.get('retail_flood_calibration_error', 0):.1%}")
                ok("sophisticated_error", f"{hl.get('sophisticated_calibration_error', 0):.1%}")
                ok("n_markets",           str(hl.get('n_markets_analyzed')))
                buckets = data.get("attention_buckets", [])
                ok("buckets returned",    str(len(buckets)))
                rfd = data.get("retail_flood_detector", {})
                ok("retail_threshold",    str(rfd.get("retail_threshold_value")))

            elif path == "/lag":
                ok("category",               data.get("category", "—"))
                ok("display_name",           data.get("display_name", "—"))
                ok("n",                      str(data.get("n")))
                ok("mean_calibration_error", f"{data.get('mean_calibration_error', 0):.1%}")
                ok("retail_pct",             f"{data.get('retail_pct', 0):.0%}")
                re = data.get("retail_error")
                se = data.get("sophisticated_error")
                ok("retail_error",           f"{re:.1%}" if re else "—")
                ok("sophisticated_error",    f"{se:.1%}" if se else "—")
                ok("error_multiplier",       f"{data.get('error_multiplier')}×" if data.get('error_multiplier') else "—")

            elif path == "/classify":
                ok("topic",             data.get("topic", "—"))
                ok("market_type",       data.get("market_type", "—"))
                ok("expected_lag_days", str(data.get("expected_lag_days")))
                ok("confidence",        f"{data.get('confidence', 0):.0%}")
                ok("reasoning",         data.get("reasoning", "—")[:120])

            elif path == "/live-alerts":
                alerts = data.get("alerts", [])
                ok("alert_count",  str(len(alerts)))
                ok("generated_at", data.get("generated_at", "—"))
                ok("note",         data.get("note", "—")[:120])
                if alerts:
                    a = alerts[0]
                    ok("top alert", a.get("market", "—")[:80])
                    ok("top score", str(a.get("social_score")))

            elif path == "/explain":
                ok("topic",        data.get("topic", "—"))
                ok("market_type",  data.get("market_type", "—"))
                ok("social_score", str(data.get("current_social_score")))
                la = data.get("lag_analysis", {})
                ok("avg_lag_days",    str(la.get("avg_lag_days")))
                ok("signal_detected", str(la.get("signal_detected")))
                factors = data.get("top_factors", [])
                for i, f_ in enumerate(factors, 1):
                    ok(f"factor {i}", f_[:100])

        except requests.exceptions.ConnectionError:
            failed += 1
            fail("Connection", f"Could not reach {url} — is the deployment Active?")
        except requests.exceptions.Timeout:
            failed += 1
            fail("Timeout", f"No response within {TIMEOUT}s")
        except Exception as e:
            failed += 1
            fail("Error", f"{type(e).__name__}: {e}")

    # ── Summary ──────────────────────────────────────────────────────────────
    rule()
    h("Summary", 2)
    p(f"- **Total endpoints tested:** {total}")
    p(f"- **Passed:** {passed}")
    p(f"- **Failed:** {failed}")
    if failed == 0:
        p("\n✅ **All endpoints healthy.**")
    else:
        p(f"\n⚠ **{failed} endpoint(s) failed — check above.**")

    content = "\n".join(lines)
    OUT_FILE.write_text(content, encoding="utf-8")
    print(f"\nReport written to: {OUT_FILE}")
    print(f"Passed: {passed}/{total}")
    if failed:
        print(f"FAILED: {failed} endpoint(s)")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test all Zerve FastAPI endpoints.")
    parser.add_argument("--url", default=os.environ.get("ZERVE_API_URL", ""), help="Base URL of the Zerve deployment")
    args = parser.parse_args()

    url = args.url.strip()
    if not url:
        url = input("Enter Zerve API base URL (e.g. https://accuracy-trap-api.hub.zerve.cloud): ").strip()
    if not url:
        print("No URL provided. Exiting.")
        sys.exit(1)

    main(url)
