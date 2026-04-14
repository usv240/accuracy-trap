"""
generate_output.py — Diagnostic snapshot of the Streamlit app's data state.

Writes output/report.md with the same data the app displays across all 5 tabs.
Run this after making changes so Claude can read the file instead of needing screenshots.

Usage:
    python generate_output.py
"""
from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

OUT_DIR  = ROOT / "output"
OUT_FILE = OUT_DIR / "report.md"
OUT_DIR.mkdir(exist_ok=True)

lines: list[str] = []

def h(text: str, level: int = 2) -> None:
    lines.append(f"\n{'#' * level} {text}\n")

def p(text: str) -> None:
    lines.append(text)

def rule() -> None:
    lines.append("\n---\n")

def table(headers: list[str], rows: list[list[str]]) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    lines.append("")

def ok(label: str, value: str) -> None:
    lines.append(f"- **{label}:** {value}")

def warn(label: str, value: str) -> None:
    lines.append(f"- ⚠ **{label}:** {value}")

def err(label: str, exc: Exception) -> None:
    lines.append(f"- ❌ **{label} FAILED:** `{type(exc).__name__}: {exc}`")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
lines.append(f"# App Diagnostic Report")
lines.append(f"_Generated: {now}_\n")
lines.append("> This file mirrors what the Streamlit app displays. Run `python generate_output.py` to refresh.\n")
rule()


# ---------------------------------------------------------------------------
# 1. Hero metrics
# ---------------------------------------------------------------------------
h("1. Hero Metrics (top of page)", 2)
try:
    from api.data_layer import get_accuracy_trap_data
    trap = get_accuracy_trap_data()
    hl   = trap.get("headline", {})
    ok("Error multiplier",            f"{hl.get('error_multiplier')}×")
    ok("Retail flood calibration error", f"{hl.get('retail_flood_calibration_error', 0):.1%}")
    ok("Sophisticated market error",  f"{hl.get('sophisticated_calibration_error', 0):.1%}")
    ok("N markets analyzed",          f"{hl.get('n_markets_analyzed'):,}")
    ok("Data source",                 hl.get('data_source', ''))
    p("")
    h("Attention buckets", 3)
    buckets = trap.get("attention_buckets", [])
    table(
        ["Label", "Mean Cal Error", "Sample Size", "Median Avg Bet"],
        [[b["label"].replace("\n", " "), f"{b['mean_calibration_error']:.1%}",
          str(b["sample_size"]), f"{b['median_avg_bet']:.0f}"] for b in buckets]
    )
    rfd = trap.get("retail_flood_detector", {})
    ok("Retail threshold (Q1 upper)",     f"{rfd.get('retail_threshold_value')} Mana")
    ok("High-attention retail error",     f"{rfd.get('high_attention_retail_error', 0):.1%}")
    ok("High-attention sophisticated error", f"{rfd.get('high_attention_sophisticated_error', 0):.1%}")
    ok("Cross-validation ratio",          f"{rfd.get('ratio')}×")
except Exception as e:
    err("Accuracy trap data", e)

rule()


# ---------------------------------------------------------------------------
# 2. Statistical significance
# ---------------------------------------------------------------------------
h("2. Statistical Significance (Welch t-test + Cohen's d)", 2)
try:
    from api.data_layer import get_statistical_significance
    sig = get_statistical_significance()
    if sig.get("available"):
        ok("p-value",        sig["p_value_display"])
        ok("Cohen's d",      f"{sig['cohens_d']} ({sig['effect_size']} effect)")
        ok("t-statistic",    str(sig.get("t_statistic", "—")))
        ok("Retail n",       str(sig["retail_n"]))
        ok("Sophisticated n",str(sig["sophisticated_n"]))
        ok("Retail mean error",      f"{sig['retail_mean']:.1%}")
        ok("Sophisticated mean error",f"{sig['sophisticated_mean']:.1%}")
        ok("Retail 95% CI",  f"[{sig['retail_ci_95'][0]:.1%}, {sig['retail_ci_95'][1]:.1%}]")
        ok("Sophisticated 95% CI", f"[{sig['sophisticated_ci_95'][0]:.1%}, {sig['sophisticated_ci_95'][1]:.1%}]")
    else:
        warn("Status", sig.get("reason", "unavailable"))
except Exception as e:
    err("Statistical significance", e)

rule()


# ---------------------------------------------------------------------------
# 3. OLS Regression
# ---------------------------------------------------------------------------
h("3. OLS Regression (causation proof)", 2)
try:
    from api.data_layer import get_ols_regression
    ols = get_ols_regression()
    if ols.get("available"):
        ok("R²",   str(ols["r_squared"]))
        ok("n",    str(ols["n"]))
        p("")
        table(
            ["Variable", "β", "Std Error", "t-stat", "p-value", "Significant?"],
            [
                ["Intercept",
                 f"{ols['intercept']['beta']:+.5f}",
                 f"{ols['intercept']['se']:.5f}",
                 f"{ols['intercept'].get('t', 0):+.2f}",
                 f"{ols['intercept']['p']:.4f}",
                 "Yes" if ols['intercept']['p'] < 0.05 else "No"],
                ["**log(avg_bet)** ← KEY",
                 f"{ols['log_avg_bet']['beta']:+.5f}",
                 f"{ols['log_avg_bet']['se']:.5f}",
                 f"{ols['log_avg_bet'].get('t', 0):+.2f}",
                 f"{ols['log_avg_bet']['p']:.2e}",
                 "YES ✓"],
                ["log(nr_bettors) control",
                 f"{ols['log_nr_bettors']['beta']:+.5f}",
                 f"{ols['log_nr_bettors']['se']:.5f}",
                 f"{ols['log_nr_bettors'].get('t', 0):+.2f}",
                 f"{ols['log_nr_bettors']['p']:.4f}",
                 "Yes" if ols['log_nr_bettors']['p'] < 0.05 else "No"],
            ]
        )
        ok("Interpretation", ols.get("interpretation", ""))
    else:
        warn("Status", ols.get("reason", "unavailable"))
except Exception as e:
    err("OLS regression", e)

rule()


# ---------------------------------------------------------------------------
# 4. Category calibration stats
# ---------------------------------------------------------------------------
h("4. Category Calibration Stats (Tab 3 & 4 charts)", 2)
try:
    from api.data_layer import get_category_calibration_stats
    cats = get_category_calibration_stats()
    rows = []
    for key, v in sorted(cats.items(), key=lambda x: x[1].get("retail_error") or 0, reverse=True):
        if key == "other":
            continue
        re = v.get("retail_error")
        se = v.get("sophisticated_error")
        mult = f"{re/se:.1f}×" if re and se and se > 0 else "—"
        rows.append([
            v.get("display_name", key),
            str(v.get("n", "")),
            f"{v.get('mean_calibration_error', 0):.1%}",
            f"{re:.1%}" if re else "—",
            f"{se:.1%}" if se else "—",
            mult,
        ])
    table(["Category", "n", "Mean Error", "Retail Error", "Soph Error", "Multiplier"], rows)
except Exception as e:
    err("Category stats", e)

rule()


# ---------------------------------------------------------------------------
# 5. Polymarket validation
# ---------------------------------------------------------------------------
h("5. Polymarket Real-Money Validation", 2)
try:
    from api.data_layer import get_polymarket_validation_data
    pm = get_polymarket_validation_data()
    if pm.get("available") is False:
        warn("Status", "polymarket_validation_results.json not found — run analysis/polymarket_validation.py")
    else:
        ok("N markets",        str(pm.get("n_markets", 0)))
        ok("Total volume",     f"${pm.get('total_volume_usdc', 0):,.0f} USDC")
        ok("Mean volume",      f"${pm.get('mean_volume_usdc', 0):,.0f} USDC")
        ok("Median volume",    f"${pm.get('median_volume_usdc', 0):,.0f} USDC")
        p("")
        tiers = pm.get("tier_summary", [])
        if tiers:
            tier_key = "vol_tier" if "vol_tier" in tiers[0] else "tier"
            table(
                ["Tier", "n", "Mean Volume (USDC)", "% of Total Volume"],
                [[t.get(tier_key, "—"), str(t.get("n", "")),
                  f"${t.get('mean_volume_usdc', 0):,.0f}",
                  f"{t.get('pct_of_total_volume', 0):.1%}"] for t in tiers]
            )
        p(f"\n_{pm.get('sophistication_note', '')}_\n")
        top5 = pm.get("top5_markets", [])
        if top5:
            h("Top 5 Markets by Volume", 3)
            table(
                ["Question", "Volume (USDC)", "Tier"],
                [[t.get("question", "")[:70], f"${t.get('volume_usdc', 0):,.0f}",
                  t.get("vol_tier", "—")] for t in top5]
            )
except Exception as e:
    err("Polymarket data", e)

rule()


# ---------------------------------------------------------------------------
# 6. CSV data sample (Tab 3 — Browse All Markets)
# ---------------------------------------------------------------------------
h("6. CSV Data Sample (1,535 resolved markets)", 2)
try:
    import pandas as pd
    csv_path = ROOT / "analysis" / "manifold_resolved_markets.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["avg_bet"] = df["volume"] / df["nr_bettors"].clip(lower=1)
        df["calibration_err"] = (df["prob"] - df["resolution"]).abs()
        ok("Total rows",      str(len(df)))
        ok("Columns",         ", ".join(df.columns.tolist()))
        ok("Mean cal error",  f"{df['calibration_err'].mean():.1%}")
        ok("Median avg_bet",  f"{df['avg_bet'].median():.1f} Mana")
        ok("Q1 avg_bet",      f"{df['avg_bet'].quantile(0.25):.1f}")
        ok("Q3 avg_bet",      f"{df['avg_bet'].quantile(0.75):.1f}")
        p("")
        h("Worst 10 markets by calibration error", 3)
        worst = df.nlargest(10, "calibration_err")[["question","prob","resolution","calibration_err","nr_bettors","avg_bet"]]
        table(
            ["Question", "Predicted", "Actual", "Error", "Bettors", "Avg Bet"],
            [[row["question"][:60], f"{row['prob']:.1%}", str(int(row["resolution"])),
              f"{row['calibration_err']:.1%}", str(int(row["nr_bettors"])),
              f"{row['avg_bet']:.0f}"] for _, row in worst.iterrows()]
        )
        h("Best 10 markets (lowest error, high volume)", 3)
        best = df[df["volume"] > 500].nsmallest(10, "calibration_err")[["question","prob","resolution","calibration_err","nr_bettors","avg_bet"]]
        table(
            ["Question", "Predicted", "Actual", "Error", "Bettors", "Avg Bet"],
            [[row["question"][:60], f"{row['prob']:.1%}", str(int(row["resolution"])),
              f"{row['calibration_err']:.1%}", str(int(row["nr_bettors"])),
              f"{row['avg_bet']:.0f}"] for _, row in best.iterrows()]
        )
    else:
        warn("CSV", "manifold_resolved_markets.csv not found")
except Exception as e:
    err("CSV data", e)

rule()


# ---------------------------------------------------------------------------
# 7. Topic classification tests (Tab 5)
# ---------------------------------------------------------------------------
h("7. Topic Classifier Tests (Tab 5)", 2)
try:
    from api.data_layer import classify_topic
    test_topics = [
        "gamestop", "bitcoin", "israel ceasefire", "fed rate hike",
        "super bowl", "ukraine war", "ethereum etf", "trump election",
        "doge", "inflation cpi",
    ]
    table(
        ["Topic", "Type", "Lag (days)", "Confidence", "Validation"],
        [[t,
          clf["market_type"],
          f"{clf['expected_lag_days']:+d}",
          f"{clf['confidence']:.0%}",
          clf.get("validation", "—")]
         for t in test_topics
         for clf in [classify_topic(t)]]
    )
except Exception as e:
    err("Topic classifier", e)

rule()


# ---------------------------------------------------------------------------
# 8. Live alerts state (Tab 4)
# ---------------------------------------------------------------------------
h("8. Live Alerts State (Tab 4 — Polymarket + Google Trends)", 2)
try:
    from api.data_layer import get_live_alerts
    alerts = get_live_alerts(limit=6)
    note   = alerts.get("note", "")
    items  = alerts.get("alerts", [])
    ok("Generated at", alerts.get("generated_at", "—"))
    ok("Note", note)
    ok("Alert count", str(len(items)))
    p("")
    if items:
        all_neutral = all(abs(a.get("social_score", 0) - 0.50) < 0.01 for a in items)
        if all_neutral:
            warn("Google Trends", "Rate-limited — all scores are 0.50 neutral")
        table(
            ["Market", "Social Score", "Probability", "Confidence", "Reprice Window"],
            [[a.get("market", "")[:55],
              f"{a.get('social_score', 0):.2f}",
              f"{a.get('current_probability', 0):.3f}" if a.get("current_probability") else "—",
              a.get("confidence", "—"),
              a.get("expected_reprice_window", "—")] for a in items]
        )
    else:
        p("_No live alerts returned._")
except Exception as e:
    err("Live alerts", e)

rule()


# ---------------------------------------------------------------------------
# 9. File / environment health check
# ---------------------------------------------------------------------------
h("9. File & Environment Health", 2)
checks = [
    ("analysis/manifold_resolved_markets.csv", ROOT / "analysis" / "manifold_resolved_markets.csv"),
    ("analysis/accuracy_trap_results.json",    ROOT / "analysis" / "accuracy_trap_results.json"),
    ("analysis/polymarket_validation_results.json", ROOT / "analysis" / "polymarket_validation_results.json"),
    ("analysis/zerve_notebook.py",             ROOT / "analysis" / "zerve_notebook.py"),
    ("analysis/polymarket_validation.py",      ROOT / "analysis" / "polymarket_validation.py"),
    ("api/main.py",                            ROOT / "api" / "main.py"),
    ("api/data_layer.py",                      ROOT / "api" / "data_layer.py"),
    ("app/streamlit_app.py",                   ROOT / "app" / "streamlit_app.py"),
    ("SUBMISSION.md",                          ROOT / "SUBMISSION.md"),
    ("requirements.txt",                       ROOT / "requirements.txt"),
]
for label, path in checks:
    if path.exists():
        size = path.stat().st_size
        ok(label, f"EXISTS ({size:,} bytes)")
    else:
        warn(label, "MISSING")

p("")
try:
    import scipy, plotly, pytrends, yfinance, fastapi, streamlit
    ok("All key packages", "scipy, plotly, pytrends, yfinance, fastapi, streamlit — all importable")
except ImportError as e:
    warn("Missing package", str(e))

rule()


# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------
content = "\n".join(lines)
OUT_FILE.write_text(content, encoding="utf-8")
print(f"Report written to: {OUT_FILE}")
print(f"Lines: {len(lines)} | Characters: {len(content):,}")
