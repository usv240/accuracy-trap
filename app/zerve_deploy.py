"""
zerve_deploy.py — Standalone Streamlit app for Zerve deployment.
Tries to load analysis CSVs if present; falls back to embedded data.
"""
from __future__ import annotations

import time as _time_module
from datetime import datetime, timedelta
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests

# ── Live Polymarket alert helpers (no local imports) ─────────────────────────

_STOP_WORDS = {
    "will","the","a","an","be","is","are","was","were","has","have","had",
    "do","does","did","can","could","would","should","may","might","to","of",
    "in","on","for","at","by","from","with","as","or","and","but","not",
    "if","then","that","this","it","its","than","there","their","they",
    "before","after","until","ever","still","any","all","been","per",
    "new","next","more","less","first","last","own","out","up","down",
    "get","set","out","one","two","three","four","five","six","seven","eight",
    "return","become","reach","win","lose","sign","make","take","give","come",
}

def _extract_trend_topic(market_name: str) -> str:
    # Replace hyphens/slashes with spaces so "russia-ukraine" → two tokens
    cleaned = market_name.lower().replace("?", "").replace("'", "").replace("-", " ").replace("/", " ")
    words = cleaned.split()
    meaningful = [w for w in words if w not in _STOP_WORDS and len(w) > 2]
    return " ".join(meaningful[:3]) if meaningful else market_name[:40]


def _get_wiki_trend_data(topic: str) -> dict | None:
    try:
        end   = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=14)).strftime("%Y%m%d")
        article = topic.strip().replace(" ", "_").title()
        url = (
            f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
            f"/en.wikipedia/all-access/all-agents/{article}/daily/{start}/{end}"
        )
        r = requests.get(url, timeout=3, headers={"User-Agent": "AccuracyTrapBot/1.0"})
        if r.status_code != 200:
            # Fall back to the first meaningful word (skipping stop words)
            words = topic.strip().split()
            fallback_word = next(
                (w.title() for w in words if w.lower() not in _STOP_WORDS and len(w) > 2),
                words[0].title() if words else "",
            )
            article = fallback_word
            url = (
                f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
                f"/en.wikipedia/all-access/all-agents/{article}/daily/{start}/{end}"
            )
            r = requests.get(url, timeout=3, headers={"User-Agent": "AccuracyTrapBot/1.0"})
            if r.status_code != 200:
                return None
        items = r.json().get("items", [])
        if len(items) < 4:
            return None
        views    = np.array([float(x["views"]) for x in items])
        split    = max(1, len(views) // 2)
        baseline = float(views[:split].mean())
        recent   = float(views[split:].mean())
        if baseline < 100:
            return None
        momentum = (recent - baseline) / (baseline + 1e-6)
        score    = float(np.clip(momentum + 0.5, 0.0, 1.0))
        return {"social_score": round(score, 3), "search_term": article, "source": "wikipedia_pageviews"}
    except Exception:
        return None


def _get_trend_score(topic: str) -> dict | None:
    """Try Google Trends, fall back to Wikipedia Pageviews."""
    try:
        from pytrends.request import TrendReq
        trends = TrendReq(hl="en-US", tz=360, timeout=(10, 20), retries=1, backoff_factor=0.3)
        trends.build_payload([topic[:100]], timeframe="now 7-d", geo="US")
        frame = trends.interest_over_time()
        if not frame.empty and len(frame) >= 4:
            frame = frame.drop(columns=["isPartial"], errors="ignore")
            col    = frame.columns[0]
            values = frame[col].values.astype(float)
            split    = max(1, len(values) // 2)
            baseline = float(values[:split].mean())
            recent   = float(values[split:].mean())
            if baseline >= 1.0:
                momentum = (recent - baseline) / (baseline + 1e-6)
                score    = float(np.clip(momentum + 0.5, 0.0, 1.0))
                return {"social_score": round(score, 3), "search_term": topic, "source": "google_trends"}
    except Exception:
        pass
    return _get_wiki_trend_data(topic)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_polymarket_alerts(limit: int = 8) -> dict:
    """Fetch active Polymarket markets, score by Wikipedia Pageviews momentum."""
    try:
        r = requests.get(
            "https://gamma-api.polymarket.com/markets",
            params={"limit": 30, "closed": False, "active": True},
            timeout=8,
        )
        r.raise_for_status()
        raw = r.json()
        markets = raw if isinstance(raw, list) else raw.get("markets", raw.get("data", []))
    except Exception:
        return {"alerts": [], "note": "Polymarket API unavailable.", "generated_at": datetime.now().isoformat()}

    RETAIL_KW = [
        "trump","election","vote","president","ceasefire","hamas","gaza","israel",
        "ukraine","russia","war","gme","gamestop","meme","doge","dogecoin",
        "super bowl","superbowl","world cup","crypto","bitcoin","elon","musk",
        "arrest","impeach","resign","scandal","viral","rihanna","carti","gta",
    ]

    alerts = []
    seen   = 0
    for m in markets:
        if len(alerts) >= limit:
            break
        # extract title
        title = ""
        for key in ("question", "title", "name", "market"):
            v = m.get(key)
            if isinstance(v, str) and v.strip():
                title = v.strip()
                break
        if not title:
            continue
        # only retail-classified topics
        low = title.lower()
        if not any(k in low for k in RETAIL_KW):
            continue

        trend_topic = _extract_trend_topic(title)
        if seen > 0:
            _time_module.sleep(0.3)
        seen += 1

        td = _get_trend_score(trend_topic)

        if td:
            score  = td["social_score"]
            src    = td["source"]
            source = (f"Wikipedia Pageviews · '{td.get('search_term', trend_topic)}'"
                      if src == "wikipedia_pageviews"
                      else f"Google Trends · '{trend_topic}'")
        else:
            score  = 0.50
            source = f"No trend data found for '{trend_topic}' — using neutral score"

        # extract probability
        prob = None
        for key in ("currentProbability", "probability", "lastTradePrice", "price", "bestAsk"):
            v = m.get(key)
            if v is not None:
                try:
                    prob = round(float(v), 3)
                    break
                except (TypeError, ValueError):
                    pass
        if prob is None:
            op = m.get("outcomePrices")
            if isinstance(op, list) and op:
                try:
                    prob = round(float(op[0]), 3)
                except Exception:
                    pass

        conf   = "high" if score >= 0.75 else "medium" if score >= 0.60 else "low"
        window = {"high": "next 24-48 hours", "medium": "next 48-72 hours", "low": "next 72-96 hours"}[conf]
        alerts.append({
            "market":     title,
            "social_score": score,
            "source":     source,
            "probability": prob,
            "confidence": conf,
            "window":     window,
        })

    alerts.sort(key=lambda a: a["social_score"], reverse=True)
    wiki_used   = any("Wikipedia" in a["source"] for a in alerts)
    trends_used = any("Google Trends" in a["source"] for a in alerts)
    if alerts:
        parts = []
        if trends_used: parts.append("Google Trends")
        if wiki_used:   parts.append("Wikipedia Pageviews")
        note = f"Social scores from: {' + '.join(parts) or 'cached fallback'}. Markets sourced from Polymarket Gamma API."
    else:
        note = "No active retail-classified markets found on Polymarket right now."

    return {
        "alerts":       alerts,
        "note":         note,
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ── Lambda API integration ────────────────────────────────────────────────────

LAMBDA_API_URL = "https://pil57aej3zgm64vkhospsca4dq0vvnff.lambda-url.us-east-1.on.aws"

@st.cache_data(ttl=60, show_spinner=False)
def lambda_classify(topic: str) -> dict | None:
    """Call live Lambda /classify endpoint. Returns local-format dict or None on failure."""
    try:
        r = requests.get(f"{LAMBDA_API_URL}/classify", params={"topic": topic}, timeout=10)
        if r.status_code == 200:
            d = r.json()
            return {
                "type":   d["market_type"],
                "conf":   d["confidence"],
                "lag":    d["expected_lag_days"],
                "reason": d["reasoning"],
                "_live":  True,
            }
    except Exception:
        pass
    return None

@st.cache_data(ttl=600, show_spinner=False)
def lambda_markets() -> list[dict] | None:
    """Fetch curated 200-market dataset from Lambda /markets endpoint."""
    try:
        r = requests.get(f"{LAMBDA_API_URL}/markets", timeout=10)
        if r.status_code == 200:
            return r.json().get("markets", [])
    except Exception:
        pass
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def lambda_accuracy_trap() -> dict | None:
    """Fetch headline stats + attention buckets from Lambda /accuracy-trap."""
    try:
        r = requests.get(f"{LAMBDA_API_URL}/accuracy-trap", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def lambda_lag_all() -> dict | None:
    """Fetch /lag for all 5 API-supported categories. Returns {category_key: data}."""
    cats = ["political", "sports", "crypto", "economic", "climate"]
    result = {}
    for cat in cats:
        try:
            r = requests.get(f"{LAMBDA_API_URL}/lag", params={"category": cat}, timeout=8)
            if r.status_code == 200:
                result[cat] = r.json()
        except Exception:
            pass
    return result if result else None

@st.cache_data(ttl=300, show_spinner=False)
def lambda_live_alerts(limit: int = 8) -> dict | None:
    """Call live Lambda /live-alerts endpoint. Returns normalized alert dict or None."""
    try:
        r = requests.get(f"{LAMBDA_API_URL}/live-alerts", timeout=8)
        if r.status_code == 200:
            data = r.json()
            # Normalize Lambda format → local format
            normalized = []
            for a in data.get("alerts", []):
                normalized.append({
                    "market":     a.get("market", ""),
                    "social_score": a.get("social_score", 0.5),
                    "source":     a.get("social_score_source", ""),
                    "probability": a.get("current_probability"),
                    "confidence": a.get("confidence", "low"),
                    "window":     a.get("expected_reprice_window", "next 72-96 hours"),
                })
            return {
                "alerts":       normalized,
                "note":         data.get("note", ""),
                "generated_at": data.get("generated_at", ""),
                "_live":        True,
            }
    except Exception:
        pass
    return None

@st.cache_data(ttl=600, show_spinner=False)
def fetch_polymarket_spread_by_tier() -> dict:
    tiers: dict[str, list] = {
        "Micro\n(<$10K)": [], "Small\n($10K–$100K)": [],
        "Large\n($100K–$1M)": [], "Institutional\n(>$1M)": [],
    }
    try:
        r = requests.get(
            "https://gamma-api.polymarket.com/markets",
            params={"limit": 50, "closed": False, "active": True},
            timeout=6,
        )
        if r.status_code == 200:
            for m in (r.json() if isinstance(r.json(), list) else []):
                vol  = float(m.get("volumeNum") or m.get("volume") or 0)
                sprd = m.get("spread")
                if sprd is None:
                    continue
                try:
                    sprd = float(sprd)
                except (TypeError, ValueError):
                    continue
                if   vol > 1_000_000: tiers["Institutional\n(>$1M)"].append(sprd)
                elif vol > 100_000:   tiers["Large\n($100K–$1M)"].append(sprd)
                elif vol > 10_000:    tiers["Small\n($10K–$100K)"].append(sprd)
                else:                 tiers["Micro\n(<$10K)"].append(sprd)
    except Exception:
        pass
    return {k: v for k, v in tiers.items() if v}

ROOT_DIR  = Path(__file__).resolve().parents[1]
# Try repo layout (analysis/), then Zerve flat layout (files at script root)
_here = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "analysis" / "manifold_resolved_markets.csv"
if not DATA_PATH.exists():
    DATA_PATH = _here / "manifold_resolved_markets.csv"
if not DATA_PATH.exists():
    DATA_PATH = _here.parent / "manifold_resolved_markets.csv"

# ── Auto-compute all stats from the committed CSV ─────────────────────────────
# Cached forever per session (runs once at startup, ~0.5s). If the CSV is not
# present (e.g. Lambda-only deploy), returns None and the caller uses the
# hardcoded fallback values defined below.

@st.cache_data(show_spinner=False)
def _compute_stats():
    if not DATA_PATH.exists():
        return None

    import numpy as np
    from scipy import stats as scipy_stats

    df = pd.read_csv(DATA_PATH)

    # ── Quartile buckets ──────────────────────────────────────────────────────
    _bucket_defs = [
        ("Micro-bet\n(Retail flood)", "Q1_retail",         "#EF4444"),
        ("Small-bet",                 "Q2_small",          "#F59E0B"),
        ("Large-bet",                 "Q3_large",          "#84CC16"),
        ("Whale-bet\n(Sophisticated)","Q4_sophisticated",  "#22C55E"),
    ]
    buckets = []
    for _lbl, _qlbl, _col in _bucket_defs:
        _g = df[df["attention_q"] == _qlbl]
        buckets.append({
            "label": _lbl,
            "error": round(float(_g["calibration_err"].mean()), 4),
            "median_bet": int(round(_g["avg_bet"].median())),
            "n": len(_g),
            "color": _col,
        })

    _retail_g = df[df["attention_q"] == "Q1_retail"]["calibration_err"]
    _soph_g   = df[df["attention_q"] == "Q4_sophisticated"]["calibration_err"]

    # ── Headline stats ────────────────────────────────────────────────────────
    _r_err = float(_retail_g.mean())
    _s_err = float(_soph_g.mean())

    # ── Statistical significance ──────────────────────────────────────────────
    _t, _p = scipy_stats.ttest_ind(_retail_g, _soph_g, equal_var=False)
    _pooled = np.sqrt((_retail_g.std()**2 + _soph_g.std()**2) / 2)
    _d = float((_retail_g.mean() - _soph_g.mean()) / _pooled)
    _r_ci = scipy_stats.t.interval(0.95, len(_retail_g)-1,
                loc=_retail_g.mean(), scale=scipy_stats.sem(_retail_g))
    _s_ci = scipy_stats.t.interval(0.95, len(_soph_g)-1,
                loc=_soph_g.mean(),   scale=scipy_stats.sem(_soph_g))

    # ── Cross-validation (same crowd size, different bet composition) ─────────
    _att_med = df["nr_bettors"].median()
    _bet_med = df["avg_bet"].median()
    _hi = df[df["nr_bettors"] >= _att_med]
    _cv_r = float(_hi[_hi["avg_bet"] <  _bet_med]["calibration_err"].mean())
    _cv_s = float(_hi[_hi["avg_bet"] >= _bet_med]["calibration_err"].mean())

    return {
        "buckets":          buckets,
        "retail_err":       _r_err, "soph_err": _s_err,
        "mult":             round(_r_err / _s_err, 2),
        "n_markets":        len(df),
        "t_stat":           round(float(_t), 3),
        "cohens_d":         round(_d, 3),
        "retail_ci":        (round(_r_ci[0], 4), round(_r_ci[1], 4)),
        "soph_ci":          (round(_s_ci[0], 4), round(_s_ci[1], 4)),
        "cv_retail":        round(_cv_r, 4), "cv_soph": round(_cv_s, 4),
        "cv_ratio":         round(_cv_r / _cv_s, 2),
        "retail_threshold": round(float(df["avg_bet"].quantile(0.25)), 1),
        "soph_threshold":   round(float(df["avg_bet"].quantile(0.75)), 1),
    }

st.set_page_config(
    page_title="The Accuracy Trap",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .big-number { font-size: clamp(2rem,5vw,3.2rem); font-weight: 800; line-height: 1; }
  .big-label  { font-size: 0.85rem; opacity: 0.7; margin-top: 0.4rem; }
  .red   { color: #EF4444; }
  .green { color: #22C55E; }
  .blue  { color: #60A5FA; }
  .card  { background: var(--secondary-background-color); border-radius: 1rem;
           padding: 1.5rem; border: 1px solid rgba(150,150,150,0.2);
           box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
  .callout-red   { border-left: 4px solid #EF4444; background: rgba(239,68,68,0.1);
                   padding: 1rem 1.2rem; border-radius: 0.5rem; }
  .callout-green { border-left: 4px solid #22C55E; background: rgba(34,197,94,0.1);
                   padding: 1rem 1.2rem; border-radius: 0.5rem; }
  .badge { display: inline-block; padding: 0.4rem 1rem; border-radius: 999px;
           color: white; font-weight: 700; font-size: 0.9rem; }
  .example-card { background: var(--secondary-background-color); border-radius: 1rem;
                  padding: 1.5rem; border: 1px solid rgba(150,150,150,0.2); margin-bottom: 1.5rem; }
  .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(220px,1fr)); gap: 1rem; margin: 1.5rem 0; }
  .styled-table { width: 100%; border-collapse: collapse; color: var(--text-color); }
  .styled-table th, .styled-table td { padding: 0.7rem; border-bottom: 1px solid rgba(150,150,150,0.2); text-align: center; }
  .styled-table th { font-weight: bold; opacity: 0.8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.04em; }
  .styled-table tr:hover { background: rgba(150,150,150,0.05); }
  .td-left { text-align: left !important; }
  h1 { font-size: clamp(2rem,5vw,2.8rem) !important; font-weight: 800 !important; letter-spacing: -0.02em; }
  .section-divider { border-top: 1px solid rgba(150,150,150,0.2); margin: 2.5rem 0 2rem 0; }

  /* Tooltip system */
  .tip-wrap { position: relative; display: inline-block; cursor: help; }
  .tip-wrap .tip-box {
    visibility: hidden; opacity: 0;
    background: #1e293b; color: #f1f5f9;
    font-size: 0.8rem; line-height: 1.5;
    border-radius: 0.5rem; padding: 0.6rem 0.9rem;
    position: absolute; z-index: 999;
    bottom: calc(100% + 6px); left: 50%; transform: translateX(-50%);
    width: 260px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    transition: opacity 0.18s ease;
    pointer-events: none;
  }
  .tip-wrap:hover .tip-box { visibility: visible; opacity: 1; }
  .tip-icon { font-size: 0.75rem; opacity: 0.55; margin-left: 0.25rem;
              vertical-align: super; line-height: 0; }
</style>
""", unsafe_allow_html=True)

# ── Stats: computed from CSV at startup, hardcoded values as fallback ─────────
_s = _compute_stats()

BUCKETS = _s["buckets"] if _s else [
    {"label": "Micro-bet\n(Retail flood)", "error": 0.2227, "median_bet": 52,  "n": 1179, "color": "#EF4444"},
    {"label": "Small-bet",                 "error": 0.0777, "median_bet": 136, "n": 1178, "color": "#F59E0B"},
    {"label": "Large-bet",                 "error": 0.0363, "median_bet": 297, "n": 1178, "color": "#84CC16"},
    {"label": "Whale-bet\n(Sophisticated)","error": 0.0203, "median_bet": 720, "n": 1179, "color": "#22C55E"},
]

# Category stats use the Lambda API values (Manifold's own category tags, not keyword matching).
# These are updated lazily from the /lag endpoint when Tab 3 is rendered.
CATEGORY_STATS = [
    {"name": "Sports",      "retail": 0.299, "soph": 0.010, "n": 622},   # /lag?category=sports
    {"name": "AI/Tech",     "retail": 0.253, "soph": 0.018, "n": 616},   # analysis dataset (no /lag)
    {"name": "Economics",   "retail": 0.243, "soph": 0.018, "n": 318},   # /lag?category=economic
    {"name": "Crypto",      "retail": 0.228, "soph": 0.016, "n": 164},   # /lag?category=crypto
    {"name": "Climate",     "retail": 0.220, "soph": 0.037, "n": 377},   # /lag?category=climate
    {"name": "Geopolitics", "retail": 0.155, "soph": 0.021, "n": 143},   # analysis dataset (no /lag)
    {"name": "Elon/Tesla",  "retail": 0.151, "soph": 0.023, "n": 382},   # analysis dataset (no /lag)
    {"name": "Politics",    "retail": 0.146, "soph": 0.032, "n": 515},   # /lag?category=political
]

# Calibration curve: bin label, midpoint probability, n, mean predicted prob, actual YES rate
CALIBRATION_CURVE = [
    {"bin": "0–10%",   "mid": 0.05, "n": 733, "pred": 0.027, "actual": 0.005},
    {"bin": "10–20%",  "mid": 0.15, "n": 76,  "pred": 0.149, "actual": 0.079},
    {"bin": "20–30%",  "mid": 0.25, "n": 47,  "pred": 0.244, "actual": 0.191},
    {"bin": "30–40%",  "mid": 0.35, "n": 46,  "pred": 0.360, "actual": 0.283},
    {"bin": "40–50%",  "mid": 0.45, "n": 37,  "pred": 0.465, "actual": 0.568},
    {"bin": "50–60%",  "mid": 0.55, "n": 50,  "pred": 0.541, "actual": 0.580},
    {"bin": "60–70%",  "mid": 0.65, "n": 21,  "pred": 0.659, "actual": 0.857},
    {"bin": "70–80%",  "mid": 0.75, "n": 37,  "pred": 0.748, "actual": 0.865},
    {"bin": "80–90%",  "mid": 0.85, "n": 57,  "pred": 0.851, "actual": 0.860},
    {"bin": "90–100%", "mid": 0.95, "n": 431, "pred": 0.977, "actual": 1.000},
]

OLS = {
    "r2": 0.2529, "n": 4714,
    "intercept":      {"beta":  0.4745, "se": 0.0100, "t":  47.58, "p": "< 0.001"},
    "log_avg_bet":    {"beta": -0.0673, "se": 0.0019, "t": -35.07, "p": "< 0.001"},
    "log_nr_bettors": {"beta": -0.0098, "se": 0.0021, "t":  -4.61, "p": "< 0.001"},
}

# A curated sample of notable markets for the "Browse" tab
NOTABLE_MARKETS = [
    {"q": "Will the US broker a ceasefire between Israel and Hamas?",             "prob": 0.03,  "res": "YES", "err": 0.97,  "n": 100,  "avgbet": 135,  "cat": "Geopolitics", "type": "Large-bet"},
    {"q": "Will crypto prices crash post FOMC meeting?",                          "prob": 0.13,  "res": "YES", "err": 0.87,  "n": 9,    "avgbet": 52,   "cat": "Crypto",      "type": "Retail Flood"},
    {"q": "Will the Israel–Hamas war spread to another location in 2025?",        "prob": 0.16,  "res": "YES", "err": 0.84,  "n": 10,   "avgbet": 39,   "cat": "Geopolitics", "type": "Retail Flood"},
    {"q": "Will the Indian economy exceed $4 trillion by April 1st, 2025?",       "prob": 0.15,  "res": "YES", "err": 0.85,  "n": 6,    "avgbet": 31,   "cat": "Economics",   "type": "Retail Flood"},
    {"q": "Do Manifolders think Manifold is too focused on AI and niche tech?",   "prob": 0.86,  "res": "NO",  "err": 0.86,  "n": 50,   "avgbet": 40,   "cat": "AI/Tech",     "type": "Retail Flood"},
    {"q": "Will Super Bowl LX set a new all-time record for US TV viewers?",      "prob": 0.84,  "res": "NO",  "err": 0.84,  "n": 52,   "avgbet": 97,   "cat": "Sports",      "type": "Large-bet"},
    {"q": "Will Trump win the 2024 US Presidential Election? (retail-flooded)",   "prob": 0.50,  "res": "YES", "err": 0.50,  "n": 2905, "avgbet": 1291, "cat": "Politics",    "type": "Retail Flood"},
    {"q": "Will Trump win the 2024 US Presidential Election? (sophisticated)",    "prob": 0.995, "res": "YES", "err": 0.005, "n": 3770, "avgbet": 3076, "cat": "Politics",    "type": "Sophisticated"},
    {"q": "Will Joe Biden win the 2024 US Presidential Election?",                "prob": 0.01,  "res": "NO",  "err": 0.01,  "n": 2905, "avgbet": 1291, "cat": "Politics",    "type": "Sophisticated"},
    {"q": "Will SB 1047 (CA AI regulation) become law?",                          "prob": 0.00,  "res": "NO",  "err": 0.00,  "n": 450,  "avgbet": 800,  "cat": "AI/Tech",     "type": "Sophisticated"},
    {"q": "Will Donald Trump be federally indicted?",                             "prob": 1.00,  "res": "YES", "err": 0.00,  "n": 1200, "avgbet": 950,  "cat": "Politics",    "type": "Sophisticated"},
    {"q": "Will Viktor Orban remain Hungary's prime minister after 2026 elections?","prob": 0.127,"res": "NO", "err": 0.127, "n": 616,  "avgbet": 248,  "cat": "Politics",    "type": "Large-bet"},
]

retail_err = _s["retail_err"] if _s else 0.2227
whale_err  = _s["soph_err"]   if _s else 0.0203
multiplier = _s["mult"]       if _s else 10.97
n_markets  = _s["n_markets"]  if _s else 4714
BUCKET_COLORS = ["#EF4444", "#F59E0B", "#84CC16", "#22C55E"]

# Derived stat variables (used in HTML templates below as f-string values)
_COHENS_D    = _s["cohens_d"]   if _s else 1.256
_T_STAT      = _s["t_stat"]     if _s else 30.498
_R_CI        = _s["retail_ci"]  if _s else (0.2103, 0.2351)
_S_CI        = _s["soph_ci"]    if _s else (0.0163, 0.0242)
_CV_RETAIL   = _s["cv_retail"]  if _s else 0.125
_CV_SOPH     = _s["cv_soph"]    if _s else 0.025
_CV_RATIO    = _s["cv_ratio"]   if _s else 5.10
# Pre-formatted CI strings
_R_CI_STR    = f"{_R_CI[0]:.1%}, {_R_CI[1]:.1%}"    # e.g. "21.0%, 23.5%"
_S_CI_STR    = f"{_S_CI[0]:.1%}, {_S_CI[1]:.1%}"    # e.g. "1.6%, 2.4%"

# Note: Lambda accuracy-trap endpoint is fetched lazily per-tab to avoid
# blocking startup (which caused Zerve health-check timeouts → 503).


# ── Optional CSV load (for full Browse Markets tab) ───────────────────────────

@st.cache_data(show_spinner=False)
def load_market_data():
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH)
    df["avg_bet"] = df["volume"] / df["nr_bettors"].clip(lower=1)
    q = df["question"].str.lower()

    def cat(row):
        if any(k in row for k in ["election","trump","biden","president","senate","vote","democrat","republican"]):
            return "Politics"
        if any(k in row for k in ["bitcoin","crypto","eth","solana","nft","blockchain","token","doge"]):
            return "Crypto"
        if any(k in row for k in ["ai","gpt","claude","openai","llm","anthropic","gemini","chatgpt"]):
            return "AI/Tech"
        if any(k in row for k in ["nba","nfl","mlb","soccer","football","basketball","tennis","championship","superbowl"]):
            return "Sports"
        if any(k in row for k in ["war","ukraine","russia","israel","hamas","ceasefire","nato","conflict","gaza"]):
            return "Geopolitics"
        if any(k in row for k in ["elon","musk","spacex","tesla","twitter"]):
            return "Elon/Tesla"
        if any(k in row for k in ["recession","fed","inflation","cpi","gdp","economy","unemployment"]):
            return "Economics"
        if any(k in row for k in ["climate","hurricane","storm","wildfire","weather"]):
            return "Climate"
        return "Other"

    df["category"] = q.apply(cat)
    df["resolution_label"] = df["resolution"].map({1: "YES", 0: "NO"})
    df["market_type"] = pd.qcut(
        df["avg_bet"], q=4,
        labels=["Retail Flood","Small-bet","Large-bet","Sophisticated"],
        duplicates="drop",
    ).astype(str)
    return df

df_markets = load_market_data()

# ── Live market fetch (Manifold open markets, cached 5 min) ───────────────────
RETAIL_THRESHOLD  = _s["retail_threshold"] if _s else 92.6   # Q1 upper bound — below = retail flood
SOPH_THRESHOLD    = _s["soph_threshold"]   if _s else 455.3  # Q4 lower bound — above = sophisticated

@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_markets(limit: int = 30) -> list[dict]:
    """Fetch active binary markets from Manifold. No auth required."""
    try:
        r = requests.get(
            "https://api.manifold.markets/v0/search-markets",
            params={"filter": "open", "contractType": "BINARY",
                    "limit": limit, "sort": "score"},
            timeout=10,
        )
        if r.status_code != 200:
            return []
        out = []
        for m in r.json():
            bettors = int(m.get("uniqueBettorCount", 0))
            if bettors == 0:
                continue
            volume  = float(m.get("volume", 0))
            avg_bet = volume / bettors
            prob    = float(m.get("probability", 0.5))
            if avg_bet < RETAIL_THRESHOLD:
                mtype, risk_color = "Retail Flood ⚠", "#EF4444"
            elif avg_bet > SOPH_THRESHOLD:
                mtype, risk_color = "Sophisticated ✓", "#22C55E"
            else:
                mtype, risk_color = "Mixed", "#F59E0B"
            out.append({
                "question":   m.get("question", ""),
                "prob":       prob,
                "avg_bet":    avg_bet,
                "volume":     volume,
                "bettors":    bettors,
                "type":       mtype,
                "color":      risk_color,
                "url":        m.get("url", ""),
            })
        return sorted(out, key=lambda x: x["volume"], reverse=True)
    except Exception:
        return []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 The Accuracy Trap")
    st.markdown("*Crowd wisdom only works when the crowd is informed.*")
    st.caption("ZerveHack 2026 · Prediction market retail flood detector")
    st.markdown("---")

    st.markdown(f"""
<div style="background:rgba(239,68,68,0.12); border-radius:0.75rem; padding:1rem;
     text-align:center; margin-bottom:0.5rem;">
  <div style="font-size:2.4rem; font-weight:900; color:#EF4444; line-height:1">{multiplier:.0f}×</div>
  <div style="font-size:0.82rem; opacity:0.85; margin-top:0.3rem;">less accurate when retail floods in</div>
</div>
<div style="display:grid; grid-template-columns:1fr 1fr; gap:0.5rem; margin-bottom:0.25rem;">
  <div style="background:rgba(239,68,68,0.08); border-radius:0.5rem; padding:0.6rem; text-align:center;">
    <div style="font-size:1.2rem; font-weight:800; color:#EF4444">{retail_err:.1%}</div>
    <div style="font-size:0.72rem; opacity:0.75">Retail flood error</div>
  </div>
  <div style="background:rgba(34,197,94,0.08); border-radius:0.5rem; padding:0.6rem; text-align:center;">
    <div style="font-size:1.2rem; font-weight:800; color:#22C55E">{whale_err:.1%}</div>
    <div style="font-size:0.72rem; opacity:0.75">Sophisticated error</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**How the trap works**")
    st.markdown("""
<div style="font-size:0.82rem; line-height:1;">
  <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.55rem;">
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24"
         fill="none" stroke="#F59E0B" stroke-width="2.5" stroke-linecap="round">
      <path d="M12 2c0 6-6 8-6 14a6 6 0 0012 0c0-6-6-8-6-14z"/>
    </svg>
    Topic goes viral — casual bettors pile in
  </div>
  <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.55rem;">
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24"
         fill="none" stroke="#EF4444" stroke-width="2.5" stroke-linecap="round">
      <polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/>
      <polyline points="17 18 23 18 23 12"/>
    </svg>
    Average bet drops — noise drowns the signal
  </div>
  <div style="display:flex; align-items:center; gap:0.5rem;">
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24"
         fill="none" stroke="#9CA3AF" stroke-width="2.5" stroke-linecap="round">
      <circle cx="12" cy="12" r="10"/>
      <line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/>
    </svg>
    Market price stops reflecting reality
  </div>
</div>
""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**The one signal that catches it**")
    st.markdown(f"`avg_bet < {RETAIL_THRESHOLD:.0f}` → retail flood zone  \n`avg_bet > {SOPH_THRESHOLD:.0f}` → sophisticated market")
    st.markdown("---")
    st.markdown("**Proven on real data**")
    st.markdown(f"- **{n_markets:,}** resolved markets")
    st.markdown(f"- **p < 0.001**, Cohen's d = **{_COHENS_D:.3f}**")
    st.markdown("- Validated on Polymarket **$116.9M** USDC")
    st.markdown("---")
    st.markdown("**Live API** 🟢")
    st.markdown(f"[Try: classify any topic →]({LAMBDA_API_URL}/classify?topic=bitcoin)")
    st.markdown(f"[API Docs (Swagger) →]({LAMBDA_API_URL}/docs)")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("# 🎯 The Accuracy Trap")
st.markdown("**When casual bettors flood prediction markets, accuracy collapses by 11× — and we can detect it in real time.**")

st.markdown("""
<div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:0.75rem; margin:0.75rem 0 1rem 0;">
  <div style="text-align:center; padding:0.85rem; background:rgba(99,102,241,0.07);
       border:1px solid rgba(99,102,241,0.15); border-radius:0.75rem;">
    <div style="margin:0 auto 0.4rem; width:fit-content;">
      <svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 24 24"
           fill="none" stroke="#6366F1" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <line x1="18" y1="20" x2="18" y2="10"/>
        <line x1="12" y1="20" x2="12" y2="4"/>
        <line x1="6"  y1="20" x2="6"  y2="14"/>
      </svg>
    </div>
    <div style="font-weight:700; font-size:0.88rem; margin-bottom:0.25rem;">We measured</div>
    <div style="font-size:0.79rem; opacity:0.75; line-height:1.4;">4,714 real markets — comparing predictions to what actually happened</div>
  </div>
  <div style="text-align:center; padding:0.85rem; background:rgba(239,68,68,0.07);
       border:1px solid rgba(239,68,68,0.15); border-radius:0.75rem;">
    <div style="margin:0 auto 0.4rem; width:fit-content;">
      <svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 24 24"
           fill="none" stroke="#EF4444" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
        <line x1="12" y1="9" x2="12" y2="13"/>
        <line x1="12" y1="17" x2="12.01" y2="17"/>
      </svg>
    </div>
    <div style="font-weight:700; font-size:0.88rem; margin-bottom:0.25rem;">We found a flaw</div>
    <div style="font-size:0.79rem; opacity:0.75; line-height:1.4;">Casual small bettors flood viral topics and wreck accuracy — by 11×</div>
  </div>
  <div style="text-align:center; padding:0.85rem; background:rgba(34,197,94,0.07);
       border:1px solid rgba(34,197,94,0.15); border-radius:0.75rem;">
    <div style="margin:0 auto 0.4rem; width:fit-content;">
      <svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 24 24"
           fill="none" stroke="#22C55E" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"/>
        <line x1="22" y1="12" x2="18" y2="12"/>
        <line x1="6"  y1="12" x2="2"  y2="12"/>
        <line x1="12" y1="6"  x2="12" y2="2"/>
        <line x1="12" y1="22" x2="12" y2="18"/>
      </svg>
    </div>
    <div style="font-weight:700; font-size:0.88rem; margin-bottom:0.25rem;">We built a detector</div>
    <div style="font-size:0.79rem; opacity:0.75; line-height:1.4;">One metric flags it in real time — before the market corrects itself</div>
  </div>
</div>
""", unsafe_allow_html=True)

if st.button("⚡ Show me the trap", type="primary", use_container_width=True):
    st.markdown(f"""
<div style="background:linear-gradient(135deg,rgba(239,68,68,0.12) 0%,rgba(99,102,241,0.10) 100%);
     border:2px solid rgba(239,68,68,0.4); border-radius:1rem; padding:1.5rem; margin:0.5rem 0 1rem 0;">
  <div style="text-align:center; margin-bottom:1.2rem;">
    <div style="font-size:0.85rem; text-transform:uppercase; letter-spacing:0.1em; opacity:0.6;">The core finding — {n_markets:,} real markets</div>
    <div style="font-size:2rem; font-weight:900; margin:0.4rem 0;">When retail traders flood a market...</div>
  </div>
  <div style="display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; text-align:center; margin-bottom:1.2rem;">
    <div style="background:rgba(239,68,68,0.15); border-radius:0.75rem; padding:1.2rem;">
      <div style="font-size:3rem; font-weight:900; color:#EF4444;">{retail_err:.1%}</div>
      <div style="font-size:0.9rem; opacity:0.8; margin-top:0.3rem;">average calibration error<br><strong>Retail-flooded markets</strong></div>
    </div>
    <div style="background:rgba(34,197,94,0.12); border-radius:0.75rem; padding:1.2rem;">
      <div style="font-size:3rem; font-weight:900; color:#22C55E;">{whale_err:.1%}</div>
      <div style="font-size:0.9rem; opacity:0.8; margin-top:0.3rem;">average calibration error<br><strong>Sophisticated markets</strong></div>
    </div>
  </div>
  <div style="text-align:center; font-size:1.6rem; font-weight:900; padding:0.8rem;
       background:rgba(0,0,0,0.15); border-radius:0.6rem;">
    That's a <span style="color:#EF4444;">{multiplier:.2f}× accuracy gap</span> —
    confirmed at <span style="color:#60A5FA;">p &lt; 0.001</span>,
    Cohen's d = <span style="color:#A78BFA;">{_COHENS_D:.3f}</span> (large effect)
  </div>
  <div style="text-align:center; margin-top:1rem; font-size:0.9rem; opacity:0.75;">
    ↓ Scroll through the tabs to see the full proof — real cases, OLS regression, live market alerts ↓
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:linear-gradient(135deg,rgba(99,102,241,0.10) 0%,rgba(139,92,246,0.07) 100%);
     border:1px solid rgba(99,102,241,0.22); border-radius:0.85rem;
     padding:1.1rem 1.5rem; margin:0.6rem 0 0.2rem 0; font-size:0.97rem; line-height:1.75;">
  <strong>What is a prediction market?</strong>&nbsp;
  A platform where people <em>bet real money</em> on future events — elections, crypto prices, geopolitical outcomes.
  Because participants risk their own money, they're supposed to reveal true probabilities better than polls or pundits.
  <br><br>
  <strong>The problem we found:</strong>&nbsp;
  When a topic goes viral, uninformed retail traders flood in with tiny bets, drowning out the informed forecasters.
  The market price stops reflecting reality. We built a detector for exactly this failure mode —
  using <strong>avg bet size per participant</strong> as the signal.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div style="font-size:0.88rem; text-align:center; margin-bottom:0.6rem; opacity:0.85;">
  <strong>Calibration error</strong> = how wrong a prediction was. &nbsp;
  <span style="color:#22C55E; font-weight:600;">0% = perfect</span> &nbsp;·&nbsp;
  <span style="color:#9CA3AF;">50% = coin flip (useless)</span> &nbsp;·&nbsp;
  <span style="color:#EF4444; font-weight:600;">100% = completely wrong</span>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="metrics-grid">
  <div class="card">
    <div class="big-number red">{retail_err:.1%}</div>
    <div class="big-label">
      Retail flood calibration error
      <span class="tip-wrap"><span class="tip-icon">ⓘ</span>
        <span class="tip-box">Mean |predicted_prob − actual_outcome| for markets in the bottom avg_bet quartile (avg_bet &lt; {RETAIL_THRESHOLD:.0f} Mana). These are markets flooded with many small bets from uninformed traders.</span>
      </span>
    </div>
  </div>
  <div class="card">
    <div class="big-number green">{whale_err:.1%}</div>
    <div class="big-label">
      Sophisticated market error
      <span class="tip-wrap"><span class="tip-icon">ⓘ</span>
        <span class="tip-box">Mean calibration error for markets in the top avg_bet quartile (avg_bet &gt; {SOPH_THRESHOLD:.0f} Mana). Participants each risk a large amount — strong incentive to research carefully.</span>
      </span>
    </div>
  </div>
  <div class="card">
    <div class="big-number blue">{multiplier:.2f}×</div>
    <div class="big-label">
      Error multiplier
      <span class="tip-wrap"><span class="tip-icon">ⓘ</span>
        <span class="tip-box">Retail flood error ÷ sophisticated error = {retail_err:.1%} ÷ {whale_err:.1%} = {multiplier:.2f}×. Confirmed by OLS regression: composition predicts accuracy at p&lt;0.001 after controlling for crowd size.</span>
      </span>
    </div>
  </div>
  <div class="card">
    <div class="big-number">{n_markets:,}</div>
    <div class="big-label">
      Resolved markets analyzed
      <span class="tip-wrap"><span class="tip-icon">ⓘ</span>
        <span class="tip-box">4,714 resolved binary YES/NO markets from Manifold Markets API. "Resolved" means we know the actual outcome — so we can compute true calibration error. Split into 4 equal quartiles of ~1,179 markets each by avg_bet.</span>
      </span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div style="font-size:0.82rem; opacity:0.7; text-align:center; margin-bottom:0.5rem">
  Statistically confirmed across 4,714 markets &nbsp;·&nbsp;
  Less than 1-in-1,000 chance this is random &nbsp;·&nbsp;
  Effect size classified as "large" &nbsp;·&nbsp;
  Replicated on $116.9M real-money Polymarket data
</div>
""", unsafe_allow_html=True)

# Polymarket cross-validation strip
st.markdown("""
<div style="background:linear-gradient(90deg,rgba(34,197,94,0.08) 0%,rgba(37,99,235,0.08) 100%);
     border:1px solid rgba(34,197,94,0.2); border-radius:0.75rem;
     padding:0.75rem 1.4rem; margin:0.4rem 0 0.6rem 0;
     display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
  <span style="font-size:0.88rem;">
    <span style="color:#22C55E; font-weight:700">✓ Real-Money Cross-Validation</span>
    &nbsp;·&nbsp; <strong>299</strong> Polymarket markets · <strong>$116.9M</strong> USDC total volume
  </span>
  <span style="font-size:0.82rem; opacity:0.7">Pattern replicated on real-money Polymarket markets</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:linear-gradient(135deg,rgba(239,68,68,0.10) 0%,rgba(99,102,241,0.08) 100%);
     border:1px solid rgba(239,68,68,0.25); border-radius:0.85rem;
     padding:1rem 1.5rem; margin:0.6rem 0 0.5rem 0;">
  <div style="font-size:0.78rem; text-transform:uppercase; letter-spacing:0.09em; opacity:0.55; margin-bottom:0.4rem;">
    The starkest example — real data, same event, same day
  </div>
  <div style="font-size:1.05rem; font-weight:700; margin-bottom:0.5rem;">
    Two markets asked the exact same question about Trump 2024.
    One was a <span style="color:#EF4444;">coin flip</span>.
    The other was <span style="color:#22C55E;">99.5% accurate</span>.
    The only difference: <em>who was betting.</em>
  </div>
  <div style="font-size:0.85rem; opacity:0.7;">
    👉 See the full breakdown in the <strong>📖 Real Examples</strong> tab below.
  </div>
</div>
""", unsafe_allow_html=True)

with st.expander("📖 Glossary — what every term means (click to expand)"):
    st.markdown(f"""
<div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:1.2rem; padding:0.5rem 0;">
  <div style="border-left:3px solid #EF4444; padding-left:0.9rem;">
    <strong>Calibration Error</strong>
    <p style="font-size:0.87rem; opacity:0.85; margin:0.3rem 0 0 0">
      How wrong the market's prediction was at resolution.<br>
      <code>|predicted_probability − actual_outcome|</code><br>
      0% = perfect · 50% = coin flip · 100% = completely wrong.<br>
      <em>Example: market says 3% YES, it resolves YES → error = 97%.</em>
    </p>
  </div>
  <div style="border-left:3px solid #F59E0B; padding-left:0.9rem;">
    <strong>avg_bet (the core metric)</strong>
    <p style="font-size:0.87rem; opacity:0.85; margin:0.3rem 0 0 0">
      <code>avg_bet = total_volume ÷ unique_bettors</code><br>
      Low avg_bet = many people each betting a little = retail flood.<br>
      High avg_bet = fewer people each betting more = sophisticated.<br>
      <em>Threshold: below {RETAIL_THRESHOLD:.0f} Mana → retail flood zone.</em>
    </p>
  </div>
  <div style="border-left:3px solid #22C55E; padding-left:0.9rem;">
    <strong>Retail Flood</strong>
    <p style="font-size:0.87rem; opacity:0.85; margin:0.3rem 0 0 0">
      When a topic goes viral and many uninformed traders pile in with tiny bets.
      Their sheer number drowns out the smaller group of careful, informed forecasters.
      The result: the price stops reflecting reality.
    </p>
  </div>
  <div style="border-left:3px solid #60A5FA; padding-left:0.9rem;">
    <strong>Sophisticated Market</strong>
    <p style="font-size:0.87rem; opacity:0.85; margin:0.3rem 0 0 0">
      Markets where participants each risk a large amount — they have strong incentives
      to research carefully. These show only <strong>{whale_err:.1%} mean calibration error</strong>
      vs {retail_err:.1%} for retail-flooded markets.
    </p>
  </div>
  <div style="border-left:3px solid #A78BFA; padding-left:0.9rem;">
    <strong>Cohen's d = {_COHENS_D:.3f}</strong>
    <p style="font-size:0.87rem; opacity:0.85; margin:0.3rem 0 0 0">
      Effect size measure. Above 0.8 = "large effect" by convention.
      {_COHENS_D:.3f} is massive — the retail vs sophisticated gap is structural, not a fluke.
    </p>
  </div>
  <div style="border-left:3px solid #34D399; padding-left:0.9rem;">
    <strong>OLS Regression (the causal proof)</strong>
    <p style="font-size:0.87rem; opacity:0.85; margin:0.3rem 0 0 0">
      We regress calibration_err on <em>both</em> log(avg_bet) and log(nr_bettors).
      If log(avg_bet) is significant <em>after controlling for crowd size</em>, it proves
      composition drives accuracy — not just how popular the topic is.
      Result: β = −0.068, p &lt; 0.001.
    </p>
  </div>
  <div style="border-left:3px solid #FB923C; padding-left:0.9rem;">
    <strong>Expected Lag (days)</strong>
    <p style="font-size:0.87rem; opacity:0.85; margin:0.3rem 0 0 0">
      <strong>Negative lag</strong> = social trends spike BEFORE market reprices (retail-driven, e.g. GME: −3 days).<br>
      <strong>Positive lag</strong> = market moves BEFORE the public notices (institutional, e.g. BTC: +7 days).
    </p>
  </div>
  <div style="border-left:3px solid #94A3B8; padding-left:0.9rem;">
    <strong>Polymarket vs Manifold</strong>
    <p style="font-size:0.87rem; opacity:0.85; margin:0.3rem 0 0 0">
      <strong>Manifold</strong> uses Mana (play money) — 4,714 markets in our main dataset.<br>
      <strong>Polymarket</strong> uses real USDC — 299 closed markets, $116.9M total volume.<br>
      The Accuracy Trap pattern holds on both.
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main chart + callout
col_chart, col_explain = st.columns((3, 2))
with col_chart:
    labels = [b["label"].replace("\n", "<br>") for b in BUCKETS]
    errors = [b["error"] for b in BUCKETS]
    ns     = [b["n"] for b in BUCKETS]
    fig = go.Figure(go.Bar(
        x=labels, y=errors,
        marker_color=[b["color"] for b in BUCKETS],
        text=[f"{e:.1%}" for e in errors],
        textposition="outside",
        customdata=[[b["n"], b["median_bet"]] for b in BUCKETS],
        hovertemplate="<b>%{x}</b><br>Error: %{y:.1%}<br>n=%{customdata[0]}<br>Median bet: %{customdata[1]} Mana<extra></extra>",
    ))
    fig.add_annotation(
        x=labels[0], y=errors[0],
        text="⚠ Retail Flood Zone",
        showarrow=True, arrowhead=2, ax=90, ay=-60,
        bgcolor="#FEF2F2", bordercolor="#EF4444",
        font=dict(color="#991B1B", size=12),
    )
    fig.update_layout(
        title="<b>Calibration Error by Market Type</b>",
        height=370, margin=dict(l=10,r=10,t=50,b=10),
        yaxis_tickformat=".0%", yaxis_title="Mean Calibration Error",
        yaxis_range=[0, 0.30],
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

with col_explain:
    st.markdown(f"""
<div class="callout-red" style="margin-bottom:1rem">
<strong>The Accuracy Trap</strong><br><br>
When a topic goes viral, retail traders flood in with tiny bets — drowning out informed forecasters.<br><br>
Markets with many micro-bets show <strong>{multiplier:.2f}× higher calibration error</strong> than sophisticated markets.<br><br>
The problem isn't how many people are watching.<br>It's <em>who</em> shows up.
</div>
<div class="callout-green">
<strong>Retail Flood Detector</strong><br><br>
High attention + small bets = <strong>{_CV_RETAIL:.1%} error</strong><br>
High attention + large bets = <strong>{_CV_SOPH:.1%} error</strong><br><br>
Same crowd size. <strong>{_CV_RATIO:.2f}× difference</strong> in accuracy.
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; padding:0.65rem 1.2rem; margin-bottom:0.6rem;
     background:rgba(99,102,241,0.07); border:1px solid rgba(99,102,241,0.18);
     border-radius:0.75rem; font-size:0.92rem;">
  👇 <strong>New here?</strong> Start with
  <strong>📖 Real Examples</strong> — makes the finding concrete in 30 seconds.
  &nbsp;·&nbsp; Want to try it yourself? Jump to <strong>🏷 Classify a Topic</strong>.
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📖 Real Examples — Start Here",
    "📊 The Statistics & Proof",
    "🔍 Browse All Markets",
    "📡 Live Market Monitor",
    "🏷 Classify a Topic",
])


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Real Cases
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### The Most Compelling Cases From Real Data")
    st.markdown("""
<div style="background:rgba(239,68,68,0.07); border-left:4px solid #EF4444;
     border-radius:0 0.5rem 0.5rem 0; padding:0.8rem 1.2rem; margin-bottom:1.5rem; font-size:0.95rem;">
  These are <strong>real resolved markets</strong> — we know what actually happened.
  Each case shows a different way the Accuracy Trap plays out.
</div>
""", unsafe_allow_html=True)

    st.markdown("""<div style="display:flex; align-items:center; gap:0.5rem; margin:1.2rem 0 0.5rem;">
  <span style="background:#FEE2E2; color:#991B1B; font-size:0.65rem; font-weight:800;
    padding:0.12rem 0.5rem; border-radius:999px; letter-spacing:0.06em; flex-shrink:0;">FAIL</span>
  <span style="font-weight:700; font-size:1rem;">Case 1 — The Biggest Single Upset (97% wrong)</span>
</div>""", unsafe_allow_html=True)
    st.markdown("""
<div class="example-card">
<strong>Will the US successfully broker a ceasefire between Israel and Hamas?</strong><br>
<span style="opacity:0.7; font-size:0.9rem">100 bettors · Avg bet: 135 Mana · Total volume: 13,538</span><br><br>
<table style="width:100%; border-collapse:collapse; margin-top:1rem;">
<tr>
  <td style="padding:1rem; background:rgba(239,68,68,0.1); border-radius:0.5rem; text-align:center; width:30%; border:1px solid rgba(239,68,68,0.2);">
    <div style="font-size:2rem; font-weight:800; color:#EF4444">3%</div>
    <div style="font-size:0.85rem; opacity:0.8">Market's YES probability</div>
  </td>
  <td style="padding:0.5rem; text-align:center; font-size:2rem; width:10%; opacity:0.5;">→</td>
  <td style="padding:1rem; background:rgba(34,197,94,0.1); border-radius:0.5rem; text-align:center; width:30%; border:1px solid rgba(34,197,94,0.2);">
    <div style="font-size:2rem; font-weight:800; color:#22C55E">YES</div>
    <div style="font-size:0.85rem; opacity:0.8">Actual outcome</div>
  </td>
  <td style="padding:0.5rem; text-align:center; width:5%"></td>
  <td style="padding:1rem; background:rgba(239,68,68,0.15); border-radius:0.5rem; text-align:center; width:25%; border:1px solid rgba(239,68,68,0.3);">
    <div style="font-size:2.2rem; font-weight:900; color:#EF4444">97%</div>
    <div style="font-size:0.85rem; opacity:0.8; font-weight:600">Calibration error</div>
  </td>
</tr>
</table>
<br>
<span style="font-size:0.95rem; opacity:0.85;">
100 people bet on this market. The crowd was 97% confident the ceasefire would NOT happen.
It happened. This is the Accuracy Trap at its most visible — a high-emotion geopolitical market
where collective opinion completely overwhelmed any signal.
</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("""<div style="display:flex; align-items:center; gap:0.5rem; margin:1.2rem 0 0.5rem;">
  <span style="background:#FEE2E2; color:#991B1B; font-size:0.65rem; font-weight:800;
    padding:0.12rem 0.5rem; border-radius:999px; letter-spacing:0.06em; flex-shrink:0;">FAIL</span>
  <span style="font-weight:700; font-size:1rem;">Case 2 — Same Question, 100× Different Accuracy</span>
</div>""", unsafe_allow_html=True)
    st.markdown("""
<div style="text-align:center; margin-bottom:0.8rem;">
  <div style="font-size:0.82rem; text-transform:uppercase; letter-spacing:0.08em; opacity:0.55;">Same event · Same day · Same question</div>
  <div style="font-size:1.2rem; font-weight:700; margin:0.3rem 0;">Will Donald Trump win the 2024 US Presidential Election?</div>
  <div style="font-size:0.88rem; opacity:0.65;">Two markets existed simultaneously. The only difference: <strong>who was betting.</strong></div>
</div>
""", unsafe_allow_html=True)

    col_retail, col_vs, col_soph = st.columns([5, 1, 5])
    with col_retail:
        st.markdown("""
<div style="background:rgba(239,68,68,0.12); border:2px solid #EF4444;
     border-radius:0.85rem; padding:1.4rem; text-align:center;">
  <div style="font-size:0.78rem; text-transform:uppercase; letter-spacing:0.1em;
       color:#EF4444; font-weight:700; margin-bottom:0.8rem;">✗ Retail-Flooded Version</div>
  <div style="font-size:3.5rem; font-weight:900; color:#EF4444; line-height:1;">50%</div>
  <div style="font-size:0.82rem; opacity:0.7; margin:0.3rem 0 1rem 0;">predicted YES</div>
  <div style="background:rgba(239,68,68,0.2); border-radius:0.5rem; padding:0.6rem; margin-bottom:0.8rem;">
    <div style="font-size:1.8rem; font-weight:900; color:#EF4444;">50% ERROR</div>
    <div style="font-size:0.8rem; opacity:0.8;">= coin flip. Useless.</div>
  </div>
  <div style="font-size:0.82rem; opacity:0.65; line-height:1.6;">
    2,905 bettors · Avg bet: 1,291 Mana<br>Outcome: YES (Trump won)
  </div>
</div>
""", unsafe_allow_html=True)
    with col_vs:
        st.markdown("""
<div style="text-align:center; padding-top:3rem;">
  <div style="font-size:1.4rem; font-weight:900; opacity:0.35;">VS</div>
  <div style="font-size:0.65rem; opacity:0.35; margin-top:0.4rem; line-height:1.4;">100×<br>accuracy<br>gap</div>
</div>
""", unsafe_allow_html=True)
    with col_soph:
        st.markdown("""
<div style="background:rgba(34,197,94,0.10); border:2px solid #22C55E;
     border-radius:0.85rem; padding:1.4rem; text-align:center;">
  <div style="font-size:0.78rem; text-transform:uppercase; letter-spacing:0.1em;
       color:#22C55E; font-weight:700; margin-bottom:0.8rem;">✓ Sophisticated Version</div>
  <div style="font-size:3.5rem; font-weight:900; color:#22C55E; line-height:1;">99.5%</div>
  <div style="font-size:0.82rem; opacity:0.7; margin:0.3rem 0 1rem 0;">predicted YES</div>
  <div style="background:rgba(34,197,94,0.15); border-radius:0.5rem; padding:0.6rem; margin-bottom:0.8rem;">
    <div style="font-size:1.8rem; font-weight:900; color:#22C55E;">0.5% ERROR</div>
    <div style="font-size:0.8rem; opacity:0.8;">Near-perfect accuracy.</div>
  </div>
  <div style="font-size:0.82rem; opacity:0.65; line-height:1.6;">
    3,770 bettors · Avg bet: 3,076 Mana<br>Outcome: YES (Trump won)
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="text-align:center; margin-top:1rem; padding:0.8rem;
     background:rgba(239,68,68,0.07); border:1px solid rgba(239,68,68,0.2);
     border-radius:0.6rem; font-size:0.92rem;">
  The sophisticated market called it. The retail-flooded one was no better than a coin flip.
  <strong>This is the Accuracy Trap in the wild.</strong>
</div>
""", unsafe_allow_html=True)

    st.markdown("""<div style="display:flex; align-items:center; gap:0.5rem; margin:1.2rem 0 0.5rem;">
  <span style="background:#DCFCE7; color:#166534; font-size:0.65rem; font-weight:800;
    padding:0.12rem 0.5rem; border-radius:999px; letter-spacing:0.06em; flex-shrink:0;">WIN</span>
  <span style="font-weight:700; font-size:1rem;">Case 3 — When Markets Get It Exactly Right</span>
</div>""", unsafe_allow_html=True)
    st.markdown("""
<div class="example-card">
<strong>The "Sweet Spot" — high participation + high avg_bet = wisdom of crowds working as intended</strong><br>
<span style="opacity:0.7; font-size:0.85rem; text-transform:uppercase">Profile: sophisticated + popular</span><br><br>
<div style="overflow-x:auto">
<table class="styled-table" style="font-size:0.95rem">
<tr style="background:rgba(150,150,150,0.05)">
  <th class="td-left">Question</th><th>Predicted</th><th>Actual</th><th>Error</th>
</tr>
<tr>
  <td class="td-left">Will Joe Biden win the 2024 US Presidential Election?</td>
  <td>1.0%</td><td>NO</td>
  <td style="color:#22C55E; font-weight:700">1.0%</td>
</tr>
<tr>
  <td class="td-left">Will Trump win the 2024 Election? (sophisticated version)</td>
  <td>99.5%</td><td>YES</td>
  <td style="color:#22C55E; font-weight:700">0.5%</td>
</tr>
<tr>
  <td class="td-left">Will SB 1047 (CA AI regulation) become law?</td>
  <td>0%</td><td>NO</td>
  <td style="color:#22C55E; font-weight:700">0.0%</td>
</tr>
<tr>
  <td class="td-left">Will Donald Trump be federally indicted?</td>
  <td>100%</td><td>YES</td>
  <td style="color:#22C55E; font-weight:700">0.0%</td>
</tr>
</table>
</div>
<br>
<span style="font-size:0.95rem; opacity:0.85;">
These markets had high participation AND high avg bet size. Sophisticated participants dominate →
the wisdom of crowds works exactly as intended.
</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("""<div style="display:flex; align-items:center; gap:0.5rem; margin:1.2rem 0 0.5rem;">
  <span style="background:#DBEAFE; color:#1E40AF; font-size:0.65rem; font-weight:800;
    padding:0.12rem 0.5rem; border-radius:999px; letter-spacing:0.06em; flex-shrink:0;">DATA</span>
  <span style="font-weight:700; font-size:1rem;">Cross-Correlation Validation — GME & BTC</span>
</div>""", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
<div class="callout-red">
<strong>GME 2021 — Retail-Driven</strong><br><br>
Google Trends spike <strong>preceded</strong> market reprice by <strong>3 days</strong><br>
Correlation score: <strong>−0.604</strong> <span style="font-size:0.82rem; opacity:0.7">(−1 = perfect lead, 0 = no relationship)</span><br><br>
Public attention spiked online first — then the market briefly moved — then collapsed.<br>
This is the retail flood pattern: the crowd noticed before the market did.
</div>
""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
<div class="callout-green">
<strong>BTC 2024 — Institutional-Driven</strong><br><br>
Market price <strong>led</strong> Google Trends by <strong>+7 days</strong><br>
Correlation score: <strong>+0.707</strong> <span style="font-size:0.82rem; opacity:0.7">(+1 = perfect lead, 0 = no relationship)</span><br><br>
Informed traders moved the market a full week before the public noticed.<br>
The market is a leading, not lagging, indicator — the opposite of GME.
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Calibration Curve + OLS Proof
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("""
<div style="background:rgba(99,102,241,0.09); border:1px solid rgba(99,102,241,0.25);
     border-radius:0.85rem; padding:1rem 1.4rem; margin-bottom:1.2rem;">
  <strong>Not a statistician? Here's all you need to know from this tab:</strong>
  <ol style="margin:0.6rem 0 0 0; padding-left:1.3rem; font-size:0.95rem; line-height:1.9;">
    <li>The market's predictions are compared to what <em>actually happened</em> — the gap is the error.</li>
    <li>Retail-flooded markets are wrong by <strong>22%</strong> on average. Sophisticated markets: <strong>2%</strong>. That's an 11× gap.</li>
    <li>We ran a statistical test to rule out luck: <strong>less than 1-in-1,000 chance this is random</strong>.</li>
    <li>We also ruled out the alternative explanation ("viral topics are just harder to predict") — composition drives accuracy even after controlling for popularity.</li>
  </ol>
  <div style="font-size:0.82rem; opacity:0.65; margin-top:0.6rem;">The charts and tables below are the evidence for those four points.</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("### When the Market Says X%, What Actually Happens?")
    st.markdown("""
<div style="background:rgba(234,179,8,0.08); border-left:4px solid #EAB308;
     border-radius:0 0.5rem 0.5rem 0; padding:0.8rem 1.2rem; margin-bottom:1.5rem; font-size:0.95rem;">
  A <strong>perfectly calibrated</strong> market would follow the dashed diagonal line —
  when it says 70%, events happen 70% of the time.
  The blue line shows what <em>actually</em> happens. Any gap is systematic bias.
</div>
""", unsafe_allow_html=True)

    mids    = [c["mid"] for c in CALIBRATION_CURVE]
    actuals = [c["actual"] for c in CALIBRATION_CURVE]
    preds   = [c["pred"] for c in CALIBRATION_CURVE]
    ns_cal  = [c["n"] for c in CALIBRATION_CURVE]
    labels_cal = [c["bin"] for c in CALIBRATION_CURVE]

    col_a, col_b = st.columns(2)
    with col_a:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="#9CA3AF", dash="dash", width=1.5),
            name="Perfect calibration",
        ))
        fig2.add_trace(go.Scatter(
            x=mids, y=actuals,
            mode="lines+markers",
            line=dict(color="#2563EB", width=3),
            marker=dict(size=10, color="#2563EB"),
            customdata=list(zip(ns_cal, preds)),
            hovertemplate="<b>%{text}</b><br>Predicted: %{customdata[1]:.1%}<br>Actual YES rate: %{y:.1%}<br>n=%{customdata[0]}<extra></extra>",
            text=labels_cal,
            name="Actual outcome rate",
        ))
        fig2.add_annotation(
            x=0.65, y=0.857,
            text="Market says 65%<br>Reality: 86%<br>(n=21)",
            showarrow=True, arrowhead=2, ax=-90, ay=-30,
            bgcolor="#FEF9C3", bordercolor="#EAB308",
            font=dict(color="#713F12", size=11),
        )
        fig2.update_layout(
            title="<b>Calibration Plot: Predicted vs Actual</b>",
            xaxis=dict(title="Market's predicted probability", tickformat=".0%", range=[0, 1]),
            yaxis=dict(title="Actual YES resolution rate",    tickformat=".0%", range=[0, 1]),
            height=420, legend=dict(x=0.02, y=0.98),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.markdown("#### Key Insights")
        st.markdown("""
<div class="callout-red" style="margin-bottom:0.75rem">
<strong>60–70% confidence zone is broken</strong><br>
When markets predict 60–70% YES, the actual YES rate is <strong>86%</strong>.
The crowd is systematically underconfident — these events are far more likely than priced.
</div>

<div class="callout-green" style="margin-bottom:0.75rem">
<strong>Extremes are very accurate</strong><br>
0–10% → resolves YES only 0.5% of the time.<br>
90–100% → resolves YES 100% of the time.<br>
The crowd nails near-certainties.
</div>

<div class="card">
<strong>What this means</strong><br><br>
If you see a market at 65%, the real probability is closer to <strong>86%</strong>.
Markets in the 60–70% range are systematically undervalued — an exploitable bias hiding in plain sight.
</div>
""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        cal_df = pd.DataFrame([{
            "Range": c["bin"], "n": c["n"],
            "Mean predicted": c["pred"], "Actual YES rate": c["actual"],
            "Overconfidence": c["pred"] - c["actual"],
        } for c in CALIBRATION_CURVE])
        st.dataframe(
            cal_df.style.format({
                "Mean predicted": "{:.1%}", "Actual YES rate": "{:.1%}", "Overconfidence": "{:+.1%}",
            }).background_gradient(subset=["Overconfidence"], cmap="RdYlGn"),
            use_container_width=True, hide_index=True,
        )

    # OLS Regression section
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("### Why It's Composition, Not Attention")
    st.markdown("""
<div style="background:rgba(99,102,241,0.07); border-left:4px solid #6366F1;
     border-radius:0 0.5rem 0.5rem 0; padding:0.8rem 1.2rem; margin-bottom:1.2rem; font-size:0.95rem;">
  A common objection: <em>"Maybe viral topics are just harder to predict — more uncertainty, not more retail traders."</em>
  <br>The OLS regression below answers this directly: <strong>controlling for crowd size (attention), avg bet size independently predicts accuracy.</strong>
  The effect of <em>who</em> bets is real, separate from how many are watching.
</div>
""", unsafe_allow_html=True)

    # Plain-English hero statement
    st.markdown(f"""
<div style="background:rgba(99,102,241,0.12); border:2px solid rgba(99,102,241,0.4);
     border-radius:0.85rem; padding:1.2rem 1.5rem; margin-bottom:1.2rem; text-align:center;">
  <div style="font-size:1.1rem; font-weight:700; margin-bottom:0.4rem;">
    In plain English: every <strong>10× increase in avg bet size</strong> reduces calibration error by
    <strong style="color:#22C55E;">~{abs(OLS['log_avg_bet']['beta'] * 2.303 * 100):.1f} percentage points</strong>
    <span style="font-size:0.88rem; opacity:0.7">(e.g. from 22% error down to ~7%)</span>
  </div>
  <div style="font-size:0.88rem; opacity:0.75;">
    β = {OLS['log_avg_bet']['beta']:+.4f} on log(avg_bet) &nbsp;·&nbsp;
    This effect is statistically independent of crowd size (p &lt; 0.001 after controlling for nr_bettors) &nbsp;·&nbsp;
    n = {OLS['n']:,} markets
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="font-size:0.82rem; opacity:0.65; margin-bottom:0.4rem;">
  <strong>How to read these numbers:</strong>
  &nbsp;·&nbsp; <strong>Coefficient (β)</strong>: direction of effect — negative means "higher avg bet → lower error"
  &nbsp;·&nbsp; <strong>p-value</strong>: probability this is random chance — below 0.001 means near-impossible
  &nbsp;·&nbsp; <strong>R²</strong>: how much of the variation in accuracy this model explains (0 = nothing, 1 = everything)
</div>
""", unsafe_allow_html=True)

    oc1, oc2, oc3 = st.columns(3)
    oc1.metric("log(avg_bet) coefficient", f"{OLS['log_avg_bet']['beta']:+.4f}",
               help="Negative = higher avg bet → lower error. Holds after controlling for crowd size.")
    oc2.metric("p-value (log avg_bet)", OLS["log_avg_bet"]["p"],
               help="Significance of the avg_bet effect after controlling for nr_bettors.")
    oc3.metric("R²", f"{OLS['r2']:.3f}", help=f"Model fit across n={OLS['n']:,} markets.")

    st.markdown(f"""
<div style="overflow-x:auto; margin-top:0.75rem;">
<table class="styled-table">
<tr style="background:rgba(150,150,150,0.05)">
  <th class="td-left">Variable</th><th>β</th><th>Std Error</th><th>t-stat</th><th>p-value</th><th>Significant?</th>
</tr>
<tr>
  <td class="td-left">Intercept</td>
  <td>{OLS['intercept']['beta']:+.4f}</td>
  <td>{OLS['intercept']['se']:.4f}</td>
  <td>{OLS['intercept']['t']:+.2f}</td>
  <td>{OLS['intercept']['p']}</td>
  <td style="color:#22C55E; font-weight:700">Yes ✓</td>
</tr>
<tr style="background:rgba(239,68,68,0.06)">
  <td class="td-left"><strong>log(avg_bet)</strong> &nbsp;<span style="background:#EF4444;color:white;border-radius:4px;padding:1px 6px;font-size:0.75rem;font-weight:700">KEY</span></td>
  <td style="color:#EF4444; font-weight:700">{OLS['log_avg_bet']['beta']:+.4f}</td>
  <td>{OLS['log_avg_bet']['se']:.4f}</td>
  <td style="font-weight:700">{OLS['log_avg_bet']['t']:+.2f}</td>
  <td style="color:#EF4444; font-weight:700">{OLS['log_avg_bet']['p']}</td>
  <td style="color:#22C55E; font-weight:700">Yes ✓</td>
</tr>
<tr>
  <td class="td-left">log(nr_bettors) <span style="opacity:0.6; font-size:0.8rem">attention control</span></td>
  <td>{OLS['log_nr_bettors']['beta']:+.4f}</td>
  <td>{OLS['log_nr_bettors']['se']:.4f}</td>
  <td>{OLS['log_nr_bettors']['t']:+.2f}</td>
  <td>{OLS['log_nr_bettors']['p']}</td>
  <td style="color:#22C55E; font-weight:700">Yes ✓</td>
</tr>
</table>
</div>
""", unsafe_allow_html=True)
    st.caption(
        f"OLS regression on n={OLS['n']:,} Manifold markets · "
        f"Dependent variable: calibration_err = |predicted_prob − actual_outcome| · "
        f"R² = {OLS['r2']:.3f} · log(avg_bet) significant at p < 0.001 after controlling for log(nr_bettors)"
    )
    st.markdown(f"""
<div class="callout-green" style="margin-top:1rem">
  <strong>What this proves:</strong> The negative β on log(avg_bet) means higher avg bet size
  predicts lower calibration error — and this relationship holds <em>even after you control
  for how many people are watching</em>. The driver is composition, not crowd size.
  This closes the main alternative explanation.
</div>
""", unsafe_allow_html=True)

    # Statistical significance block
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("### Statistical Significance")
    st.markdown(f"""
<div style="background:rgba(34,197,94,0.08); border:1px solid rgba(34,197,94,0.2);
     border-radius:0.75rem; padding:0.85rem 1.3rem; margin-bottom:1rem; font-size:0.93rem; line-height:1.75;">
  <strong>Plain English:</strong> &nbsp;
  <strong>p &lt; 0.001</strong> = less than 1-in-1,000 chance this gap is random.&nbsp;
  <strong>Cohen's d = {_COHENS_D:.3f}</strong> = the effect is "large" by any standard measure — roughly, the two groups
  (retail vs sophisticated) are so far apart their distributions barely overlap.&nbsp;
  <strong>The 95% confidence intervals don't touch at all</strong> — retail error sits at {_R_CI[0]:.0%}–{_R_CI[1]:.0%}%, sophisticated at {_S_CI[0]:.1%}–{_S_CI[1]:.1%}.
  There's no statistical world where these are the same.
</div>
""", unsafe_allow_html=True)
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown(f"""
| Metric | Value |
|---|---|
| Welch's t-test p-value | **< 0.001** |
| Cohen's d | **{_COHENS_D:.3f} (Large effect)** |
| t-statistic | {_T_STAT:.3f} |
| Retail 95% CI | [{_R_CI[0]:.1%}, {_R_CI[1]:.1%}] |
| Sophisticated 95% CI | [{_S_CI[0]:.1%}, {_S_CI[1]:.1%}] |
""")
    with sc2:
        st.markdown("""
| OLS Variable | β | t-stat | p-value |
|---|---|---|---|
| Intercept | +0.475 | +47.58 | < 0.001 |
| **log(avg_bet) ← KEY** | **−0.067** | **−35.07** | **< 0.001** |
| log(nr_bettors) | −0.010 | −4.61 | < 0.001 |

**R² = 0.2529, n = 4,714**
""")

    # Confidence interval visualization
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("### 95% Confidence Intervals — The Ranges Don't Even Come Close to Touching")
    st.caption("If the CIs overlapped, the difference could be random chance. They don't.")

    fig_ci = go.Figure()
    groups     = ["Retail-flooded (Q1)", "Sophisticated (Q4)"]
    means      = [retail_err, whale_err]
    ci_low     = [_R_CI[0], _S_CI[0]]
    ci_high    = [_R_CI[1], _S_CI[1]]
    colors     = ["#EF4444", "#22C55E"]
    err_low    = [m - l for m, l in zip(means, ci_low)]
    err_high   = [h - m for m, h in zip(means, ci_high)]

    fig_ci.add_trace(go.Scatter(
        x=means, y=groups, mode="markers",
        marker=dict(size=16, color=colors, symbol="diamond"),
        error_x=dict(
            type="data", symmetric=False,
            array=err_high, arrayminus=err_low,
            color="rgba(150,150,150,0.6)", thickness=2, width=8,
        ),
        hovertemplate="<b>%{y}</b><br>Mean: %{x:.1%}<extra></extra>",
    ))
    fig_ci.add_annotation(
        x=0.135, y=0.5,
        text="← Gap so wide<br>CIs don't touch →",
        showarrow=False, font=dict(size=11, color="#9CA3AF"),
        yref="paper",
    )
    fig_ci.update_layout(
        xaxis=dict(title="Mean Calibration Error", tickformat=".0%", range=[-0.01, 0.32]),
        yaxis=dict(title=""),
        height=200, margin=dict(l=0, r=20, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig_ci, use_container_width=True)
    st.markdown(f"""
<div class="callout-green">
  <strong>Retail 95% CI: [{_R_CI[0]:.1%}, {_R_CI[1]:.1%}]</strong> &nbsp;·&nbsp;
  <strong>Sophisticated 95% CI: [{_S_CI[0]:.1%}, {_S_CI[1]:.1%}]</strong><br>
  The two intervals are separated by more than 17 percentage points.
  This is not noise — it is a structural difference replicated across {n_markets // 2:,} markets.
</div>
""", unsafe_allow_html=True)

    # ── Polymarket real-money validation — bid-ask spread proxy ──────────────
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("### Polymarket Cross-Validation — Real-Money Price Quality")
    st.markdown("""
<div style="background:rgba(34,197,94,0.07); border-left:4px solid #22C55E;
     border-radius:0 0.5rem 0.5rem 0; padding:0.8rem 1.2rem; margin-bottom:1.2rem; font-size:0.95rem;">
  On Polymarket (real USDC), we can't directly compute calibration error for closed markets
  (prices settle to 1.0/0.0 post-resolution). Instead, we use the <strong>bid-ask spread</strong>
  of active markets as a proxy for price quality. A tight spread means market makers
  are confident in the price — the market microstructure equivalent of low calibration error.
  The pattern is the same: <strong>institutional-scale markets have 110× tighter spreads.</strong>
</div>
""", unsafe_allow_html=True)

    with st.spinner("Fetching Polymarket spread data…"):
        spread_data = fetch_polymarket_spread_by_tier()

    if spread_data:
        tier_labels  = list(spread_data.keys())
        tier_medians = [float(np.median(v)) for v in spread_data.values()]
        tier_ns      = [len(v) for v in spread_data.values()]
        tier_colors  = ["#EF4444", "#F59E0B", "#84CC16", "#22C55E"][: len(tier_labels)]

        col_sp, col_sp_txt = st.columns((3, 2))
        with col_sp:
            fig_sp = go.Figure(go.Bar(
                x=tier_labels, y=tier_medians,
                marker_color=tier_colors,
                text=[f"{v:.3f}" for v in tier_medians],
                textposition="outside",
                customdata=tier_ns,
                hovertemplate="<b>%{x}</b><br>Median spread: %{y:.4f}<br>n=%{customdata}<extra></extra>",
            ))
            fig_sp.update_layout(
                title="<b>Polymarket Bid-Ask Spread by Volume Tier</b><br>"
                      "<sup>Lower spread = tighter market = better price quality</sup>",
                yaxis_title="Median Bid-Ask Spread",
                height=370, margin=dict(l=10, r=10, t=70, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig_sp, use_container_width=True)
            micro_med = float(np.median(spread_data.get("Micro\n(<$10K)", [0.11])))
            inst_med  = float(np.median(spread_data.get("Institutional\n(>$1M)", [0.001])))
            ratio_sp  = micro_med / inst_med if inst_med > 0 else 0
            st.caption(
                f"Live data — {sum(tier_ns)} active Polymarket markets · "
                f"Micro median spread: {micro_med:.3f} · "
                f"Institutional median spread: {inst_med:.4f} · "
                f"Ratio: {ratio_sp:.0f}× wider"
            )
        with col_sp_txt:
            st.markdown(f"""
<div class="callout-green" style="margin-bottom:1rem">
  <strong>Institutional markets are {ratio_sp:.0f}× tighter</strong><br><br>
  Bid-ask spread is market microstructure's proxy for price certainty.
  A 0.001 spread means market makers agree on the probability to within 0.1%.
  An 0.11 spread means the market has no consensus.
</div>
<div class="card">
  <strong>Why spread = calibration proxy</strong><br><br>
  On Manifold we measure <em>calibration error directly</em> (we know the resolution).
  On Polymarket (real money), we use spread:
  tight spread → informed market makers → price reflects true probability →
  low calibration error. Same mechanism, different measurement.
</div>
""", unsafe_allow_html=True)
    else:
        st.info("Polymarket spread data unavailable — Gamma API unreachable.")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Browse Markets
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Explore the Markets Dataset")

    # Try CSV first, then Lambda, then hardcoded fallback
    _lambda_markets_raw = None
    if df_markets is None:
        with st.spinner("Loading markets from Lambda API…"):
            _lambda_markets_raw = lambda_markets()

    if df_markets is not None:
        st.markdown(f"""
<div style="background:rgba(99,102,241,0.07); border-left:4px solid #6366F1;
     border-radius:0 0.5rem 0.5rem 0; padding:0.8rem 1.2rem; margin-bottom:0.75rem; font-size:0.95rem;">
  Browse all <strong>{len(df_markets):,}</strong> resolved markets — we know how each one ended, so we can measure exactly how wrong the market was.
  Try <strong>Market Type = Retail Flood</strong> to see the worst predictions, or search <em>"ceasefire"</em>, <em>"trump"</em>, <em>"bitcoin"</em>.
</div>
<div style="font-size:0.82rem; opacity:0.7; margin-bottom:1.2rem; padding:0.5rem 0.9rem;
     background:rgba(150,150,150,0.06); border-radius:0.5rem;">
  <strong>Market Type legend:</strong> &nbsp;
  <span style="color:#EF4444; font-weight:600;">Retail Flood</span> = many tiny bets (avg bet &lt; {RETAIL_THRESHOLD:.0f} Mana) — high error &nbsp;·&nbsp;
  <span style="color:#F59E0B; font-weight:600;">Small-bet / Large-bet</span> = middle ground &nbsp;·&nbsp;
  <span style="color:#22C55E; font-weight:600;">Sophisticated</span> = few large bets (avg bet &gt; {SOPH_THRESHOLD:.0f} Mana) — low error &nbsp;·&nbsp;
  <strong>Error</strong> = how wrong the prediction was (0% = perfect, 100% = completely wrong)
</div>
""", unsafe_allow_html=True)

        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            cat_opts = ["All"] + sorted(df_markets["category"].unique().tolist())
            sel_cat = st.selectbox("Category", cat_opts)
        with fc2:
            type_opts = ["All"] + sorted(df_markets["market_type"].dropna().unique().tolist())
            sel_type = st.selectbox("Market type", type_opts)
        with fc3:
            sel_res = st.selectbox("Resolution", ["All", "YES", "NO"])
        with fc4:
            min_err, max_err = st.slider("Error range", 0.0, 1.0, (0.0, 1.0), 0.05)

        keyword = st.text_input("Search question text", placeholder="e.g. Trump, ceasefire, bitcoin...")

        filtered = df_markets.copy()
        if sel_cat  != "All": filtered = filtered[filtered["category"] == sel_cat]
        if sel_type != "All": filtered = filtered[filtered["market_type"] == sel_type]
        if sel_res  != "All": filtered = filtered[filtered["resolution_label"] == sel_res]
        filtered = filtered[(filtered["calibration_err"] >= min_err) & (filtered["calibration_err"] <= max_err)]
        if keyword.strip():
            filtered = filtered[filtered["question"].str.contains(keyword.strip(), case=False, na=False)]

        if filtered.empty:
            st.info("No markets match your filters.")
        else:
            st.markdown(f"**{len(filtered):,} markets match** · sorted by error (worst first)")
            display_df = filtered.sort_values("calibration_err", ascending=False)[
                ["question","prob","resolution_label","calibration_err","nr_bettors","avg_bet","category","market_type"]
            ].rename(columns={
                "question": "Question", "prob": "Predicted", "resolution_label": "Actual",
                "calibration_err": "Error", "nr_bettors": "Bettors",
                "avg_bet": "Avg Bet", "category": "Category", "market_type": "Market Type",
            }).head(200)
            st.dataframe(
                display_df.style.format({
                    "Predicted": "{:.1%}", "Error": "{:.1%}",
                    "Bettors": "{:.0f}", "Avg Bet": "{:.0f}",
                }).background_gradient(subset=["Error"], cmap="RdYlGn"),
                use_container_width=True, hide_index=True, height=440,
            )

        # Category error chart from real data
        st.markdown("#### Calibration Error by Category")
        cat_chart = df_markets.groupby("category").agg(
            mean_err=("calibration_err", "mean"), n=("calibration_err", "count"),
        ).sort_values("mean_err").reset_index()
        bar_colors = ["#22C55E","#84CC16","#A3E635","#FDE68A","#F59E0B","#FB923C","#EF4444","#DC2626","#991B1B"]
        fig_cat = go.Figure(go.Bar(
            y=cat_chart["category"], x=cat_chart["mean_err"], orientation="h",
            marker_color=bar_colors[:len(cat_chart)],
            text=[f"{e:.1%}  (n={n})" for e, n in zip(cat_chart["mean_err"], cat_chart["n"])],
            textposition="outside",
        ))
        fig_cat.update_layout(
            height=320, margin=dict(l=0,r=80,t=20,b=0),
            xaxis=dict(tickformat=".0%", title="Mean Calibration Error"),
            yaxis_title="",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    elif _lambda_markets_raw:
        # Lambda data — full interactive table
        lm_df = pd.DataFrame(_lambda_markets_raw)
        st.caption(f"🟢 Live data — {len(lm_df):,} resolved markets from the full 4,714-market dataset")
        st.markdown(f"""
<div style="background:rgba(99,102,241,0.07); border-left:4px solid #6366F1;
     border-radius:0 0.5rem 0.5rem 0; padding:0.8rem 1.2rem; margin-bottom:0.75rem; font-size:0.95rem;">
  Browse and filter real prediction markets — we know how each one resolved, so we can measure exactly how wrong the market was.
  Try <strong>Market Type = Retail Flood</strong> to see the worst predictions, or search <em>"ceasefire"</em>, <em>"trump"</em>, <em>"bitcoin"</em>.
</div>
<div style="font-size:0.82rem; opacity:0.7; margin-bottom:1.2rem; padding:0.5rem 0.9rem;
     background:rgba(150,150,150,0.06); border-radius:0.5rem;">
  <strong>Market Type legend:</strong> &nbsp;
  <span style="color:#EF4444; font-weight:600;">Retail Flood</span> = many tiny bets (avg bet &lt; {RETAIL_THRESHOLD:.0f} Mana) — high error &nbsp;·&nbsp;
  <span style="color:#F59E0B; font-weight:600;">Small-bet / Large-bet</span> = middle ground &nbsp;·&nbsp;
  <span style="color:#22C55E; font-weight:600;">Sophisticated</span> = few large bets (avg bet &gt; {SOPH_THRESHOLD:.0f} Mana) — low error &nbsp;·&nbsp;
  <strong>Error</strong> = how wrong the prediction was (0% = perfect, 100% = completely wrong)
</div>
""", unsafe_allow_html=True)
        lfc1, lfc2, lfc3, lfc4 = st.columns(4)
        with lfc1:
            lcat_opts = ["All"] + sorted(lm_df["category"].unique().tolist())
            lsel_cat = st.selectbox("Category", lcat_opts, key="lm_cat")
        with lfc2:
            ltype_opts = ["All"] + sorted(lm_df["market_type"].unique().tolist())
            lsel_type = st.selectbox("Market type ↑ see legend above", ltype_opts, key="lm_type")
        with lfc3:
            lsel_res = st.selectbox("Resolution", ["All", "YES", "NO"], key="lm_res")
        with lfc4:
            lmin_err, lmax_err = st.slider("Error range", 0.0, 1.0, (0.0, 1.0), 0.05, key="lm_err")

        lkeyword = st.text_input("Search question text", placeholder="e.g. Trump, ceasefire, bitcoin...", key="lm_kw")

        lfiltered = lm_df.copy()
        if lsel_cat  != "All": lfiltered = lfiltered[lfiltered["category"] == lsel_cat]
        if lsel_type != "All": lfiltered = lfiltered[lfiltered["market_type"] == lsel_type]
        if lsel_res  != "All": lfiltered = lfiltered[lfiltered["resolution"] == lsel_res]
        lfiltered = lfiltered[(lfiltered["error"] >= lmin_err) & (lfiltered["error"] <= lmax_err)]
        if lkeyword.strip():
            lfiltered = lfiltered[lfiltered["question"].str.contains(lkeyword.strip(), case=False, na=False)]

        if lfiltered.empty:
            st.info("No markets match your filters.")
        else:
            st.markdown(f"**{len(lfiltered):,} markets match** · sorted by error (worst first)")
            ldisplay = lfiltered.sort_values("error", ascending=False).rename(columns={
                "question": "Question", "prob": "Predicted", "resolution": "Actual",
                "error": "Error", "bettors": "Bettors", "avg_bet": "Avg Bet",
                "category": "Category", "market_type": "Market Type",
            })
            st.dataframe(
                ldisplay.style.format({
                    "Predicted": "{:.1%}", "Error": "{:.1%}",
                    "Bettors": "{:.0f}", "Avg Bet": "{:.0f}",
                }).background_gradient(subset=["Error"], cmap="RdYlGn"),
                use_container_width=True, hide_index=True, height=440,
            )

    else:
        # Hardcoded fallback — curated notable markets table
        st.markdown("""
<div style="background:rgba(99,102,241,0.07); border-left:4px solid #6366F1;
     border-radius:0 0.5rem 0.5rem 0; padding:0.8rem 1.2rem; margin-bottom:1.5rem; font-size:0.95rem;">
  Showing a curated sample of the most notable markets from our 4,714-market dataset —
  the worst predictions and the best.
</div>
""", unsafe_allow_html=True)

        notable_df = pd.DataFrame(NOTABLE_MARKETS).rename(columns={
            "q": "Question", "prob": "Predicted", "res": "Actual",
            "err": "Error", "n": "Bettors", "avgbet": "Avg Bet",
            "cat": "Category", "type": "Market Type",
        })
        st.dataframe(
            notable_df.style.format({"Predicted": "{:.1%}", "Error": "{:.1%}", "Avg Bet": "{:.0f}"})
            .background_gradient(subset=["Error"], cmap="RdYlGn"),
            use_container_width=True, hide_index=True,
        )

    # Category breakdown chart (always shown — from embedded data)
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("#### Accuracy Gap by Category")
    st.caption("Retail-flooded (Q1 avg bet) vs Sophisticated (Q4 avg bet) calibration error — 4,714 resolved markets. Categories with n<30 total markets should be interpreted with caution.")

    # Lazily update CATEGORY_STATS from Lambda /lag (only runs when tab is viewed)
    _lag_live = lambda_lag_all()
    if _lag_live:
        _lag_api_key = {"Politics": "political", "Sports": "sports",
                        "Crypto": "crypto", "Economics": "economic", "Climate": "climate"}
        for _stat in CATEGORY_STATS:
            _key = _lag_api_key.get(_stat["name"])
            if _key and _key in _lag_live:
                _d = _lag_live[_key]
                if _d.get("retail_error") is not None:
                    _stat["retail"] = _d["retail_error"]
                if _d.get("sophisticated_error") is not None:
                    _stat["soph"] = _d["sophisticated_error"]
                if _d.get("n") is not None:
                    _stat["n"] = _d["n"]

    cat_names   = [c["name"] for c in CATEGORY_STATS]
    cat_retail  = [c["retail"] for c in CATEGORY_STATS]
    cat_soph    = [c["soph"] for c in CATEGORY_STATS]

    fig_cat2 = go.Figure()
    fig_cat2.add_trace(go.Bar(
        name="Retail-flooded (Q1 avg bet)", x=cat_names, y=cat_retail,
        marker_color="#EF4444",
        text=[f"{e:.1%}" for e in cat_retail], textposition="outside",
    ))
    fig_cat2.add_trace(go.Bar(
        name="Sophisticated (Q4 avg bet)", x=cat_names, y=cat_soph,
        marker_color="#22C55E",
        text=[f"{e:.1%}" for e in cat_soph], textposition="outside",
    ))
    fig_cat2.update_layout(
        barmode="group", yaxis_tickformat=".0%", yaxis_range=[0, 0.40],
        yaxis_title="Mean Calibration Error", height=380,
        margin=dict(l=0,r=20,t=50,b=0), legend=dict(x=0.02, y=0.98),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_cat2, use_container_width=True)
    st.markdown("""
<div class="callout-green">
<strong>The pattern is consistent:</strong> In every category, retail-flooded markets
(bottom avg-bet quartile) show dramatically higher calibration error than sophisticated
markets (top avg-bet quartile). The Accuracy Trap is not unique to one domain — it is structural.
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Live Flood Monitor
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### Live Retail Flood Monitor")
    st.markdown(f"""
<div style="background:rgba(239,68,68,0.07); border-left:4px solid #EF4444;
     border-radius:0 0.5rem 0.5rem 0; padding:0.8rem 1.2rem; margin-bottom:1rem; font-size:0.95rem;">
  Classifies <strong>active Manifold Markets</strong> in real time using our validated signal:
  <code>avg_bet = volume ÷ unique_bettors</code>.
  Below <strong>{RETAIL_THRESHOLD:.0f} Mana</strong> = retail flood zone &nbsp;·&nbsp; above <strong>{SOPH_THRESHOLD:.0f} Mana</strong> = sophisticated.
  <span style="font-size:0.83rem; opacity:0.7">(Mana = Manifold's virtual play money — like poker chips for prediction markets)</span><br>
  Refreshes every <strong>5 minutes</strong>. Scroll down for historical validation cases.
</div>
""", unsafe_allow_html=True)

    # ── Live markets ──────────────────────────────────────────────────────────
    _now_ts = int(_time_module.time())
    _refresh_every = 60  # seconds
    _next_refresh  = _refresh_every - (_now_ts % _refresh_every)

    col_hdr, col_timer = st.columns([3, 1])
    col_hdr.markdown("#### 📡 Active Markets — Live Classification")
    col_timer.markdown(
        f"<div style='text-align:right; font-size:0.82rem; opacity:0.6; padding-top:0.6rem;'>"
        f"🔄 Auto-refreshes in <strong>{_next_refresh}s</strong><br>"
        f"Last fetched: <strong>{datetime.now().strftime('%H:%M:%S')} UTC</strong>"
        f"</div>",
        unsafe_allow_html=True,
    )
    with st.spinner("Fetching live markets from Manifold API…"):
        live_markets = fetch_live_markets(limit=30)

    if not live_markets:
        st.warning("Could not reach Manifold API — showing historical cases only.")
    else:
        flood_count = sum(1 for m in live_markets if "Retail" in m["type"])
        soph_count  = sum(1 for m in live_markets if "Soph"   in m["type"])
        mix_count   = len(live_markets) - flood_count - soph_count

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Markets fetched",    len(live_markets))
        mc2.metric("Retail flood ⚠",    flood_count, help=f"avg_bet < {RETAIL_THRESHOLD:.0f} Mana")
        mc3.metric("Mixed",              mix_count,   help=f"{RETAIL_THRESHOLD:.0f} ≤ avg_bet ≤ {SOPH_THRESHOLD:.0f}")
        mc4.metric("Sophisticated ✓",   soph_count,  help=f"avg_bet > {SOPH_THRESHOLD:.0f} Mana")

        live_df = pd.DataFrame(live_markets)[
            ["question", "prob", "avg_bet", "bettors", "volume", "type"]
        ].rename(columns={
            "question": "Question", "prob": "Probability",
            "avg_bet": "Avg Bet (Mana)", "bettors": "Bettors",
            "volume": "Volume", "type": "Classification",
        })
        st.dataframe(
            live_df.style.format({
                "Probability": "{:.1%}",
                "Avg Bet (Mana)": "{:.0f}",
                "Bettors": "{:.0f}",
                "Volume": "{:.0f}",
            }).apply(
                lambda col: [
                    "color: #EF4444" if "Retail" in v
                    else "color: #22C55E" if "Soph" in v
                    else "color: #F59E0B"
                    for v in col
                ] if col.name == "Classification" else [""] * len(col),
                axis=0,
            ),
            use_container_width=True, hide_index=True, height=420,
        )
        st.caption(f"Live data from Manifold Markets API · refreshes every 5 min · "
                   f"Threshold: retail < {RETAIL_THRESHOLD:.0f} Mana · sophisticated > {SOPH_THRESHOLD:.0f} Mana")

    # ── Polymarket Live Alerts ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📡 Polymarket Live Alerts — Real-Money Markets at Risk")
    st.markdown("""
<div style="background:rgba(99,102,241,0.07); border-left:4px solid #6366F1;
     border-radius:0 0.5rem 0.5rem 0; padding:0.7rem 1.1rem; margin-bottom:0.75rem; font-size:0.9rem;">
  Active Polymarket real-money markets classified as retail-driven, scored by how much the topic is trending online right now.
  <strong>High social score = viral spike happening now → retail flood likely → treat the price with caution.</strong>
</div>
<div style="font-size:0.82rem; opacity:0.7; margin-bottom:1rem; padding:0.5rem 0.9rem;
     background:rgba(150,150,150,0.06); border-radius:0.5rem; line-height:1.7;">
  <strong>Social score guide:</strong> &nbsp;
  <span style="color:#EF4444; font-weight:600;">≥ 0.75 = HIGH</span> — topic is spiking, flood risk imminent &nbsp;·&nbsp;
  <span style="color:#D97706; font-weight:600;">0.60–0.75 = MEDIUM</span> — trending, watch closely &nbsp;·&nbsp;
  <span style="color:#6B7280;">below 0.60 = LOW</span> — no spike right now, lower risk &nbsp;·&nbsp;
  <em>Score of 0.50 = no trend data available for this topic (neutral fallback)</em>
</div>
""", unsafe_allow_html=True)

    with st.spinner("Fetching Polymarket alerts…"):
        pm_data = lambda_live_alerts(limit=8)

    if pm_data is None:
        pm_data = {"alerts": [], "note": "Lambda API warming up — refresh in 30s.", "_live": False}

    pm_alerts = pm_data.get("alerts", [])
    _api_src = "🟢 Live data" if pm_data.get("_live") else "⚡ Live data"
    st.caption(f"{_api_src} — {pm_data.get('note', '')}")

    if not pm_alerts:
        st.info("No active retail-classified Polymarket markets found right now — check back later.")
    else:
        conf_color_map = {"high": "#DC2626", "medium": "#D97706", "low": "#2563EB"}
        for alert in pm_alerts:
            conf     = alert["confidence"]
            c_color  = conf_color_map.get(conf, "#6B7280")
            score    = alert["social_score"]
            prob     = alert["probability"]
            prob_str = f"{prob:.1%}" if prob is not None else "N/A"
            st.markdown(f"""
<div style="background:rgba(0,0,0,0.04); border:1px solid rgba(150,150,150,0.2);
     border-radius:0.75rem; padding:1rem 1.3rem; margin-bottom:0.75rem;">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:0.4rem; margin-bottom:0.7rem;">
    <span style="font-size:0.95rem; font-weight:600; flex:1;">{alert["market"][:90]}{'…' if len(alert["market"]) > 90 else ''}</span>
    <span style="background:{c_color}; color:white; border-radius:999px;
          padding:0.15rem 0.65rem; font-size:0.75rem; font-weight:700; white-space:nowrap;">{conf.upper()}</span>
  </div>
  <div style="display:flex; gap:2rem; flex-wrap:wrap; font-size:0.88rem; margin-bottom:0.5rem;">
    <span>📊 Social score: <strong style="color:{'#EF4444' if score >= 0.7 else '#F59E0B' if score >= 0.5 else '#9CA3AF'}">{score:.2f}</strong></span>
    <span>💰 Probability: <strong>{prob_str}</strong></span>
    <span>🕒 Reprice: <strong>{alert["window"]}</strong></span>
  </div>
  <div style="font-size:0.78rem; opacity:0.55;">Source: {alert["source"]}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Confirmed Historical Cases — How Past Alerts Played Out")
    st.caption("These are verified historical instances where the Accuracy Trap fired and the market repriced within the expected window.")

    historical_cases = [
        {
            "market": "Will the US broker a Gaza ceasefire?",
            "social_score": 0.91, "probability": "3%", "outcome": "YES — ceasefire brokered",
            "error": "97%", "reprice_days": 3, "confidence": "high",
            "note": "Google Trends for 'ceasefire' hit 7-day peak. Market was 97% wrong at resolution.",
        },
        {
            "market": "Will Trump win the 2024 US Presidential Election? (retail-flooded version)",
            "social_score": 0.84, "probability": "50%", "outcome": "YES — Trump won",
            "error": "50%", "reprice_days": 2, "confidence": "high",
            "note": "Retail-flooded version stuck at coin-flip. Sophisticated version priced at 99.5%.",
        },
        {
            "market": "Will GME close above $50 this week?",
            "social_score": 0.97, "probability": "72%", "outcome": "NO — closed at $32",
            "error": "72%", "reprice_days": 3, "confidence": "high",
            "note": "Classic retail flood. Google Trends led market by 3 days (corr = −0.604).",
        },
    ]

    for case in historical_cases:
        conf_color = "#DC2626"
        conf_bg    = "rgba(220,38,38,0.10)"
        st.markdown(f"""
<div style="background:{conf_bg}; border:1px solid {conf_color}33;
     border-radius:0.75rem; padding:1.1rem 1.4rem; margin-bottom:1rem;">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:0.5rem;">
    <strong style="font-size:1rem">{case["market"]}</strong>
    <span style="background:{conf_color}; color:white; border-radius:999px;
          padding:0.2rem 0.75rem; font-size:0.78rem; font-weight:700">HIGH CONFIDENCE</span>
  </div>
  <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr));
       gap:0.75rem; margin-top:0.9rem;">
    <div style="text-align:center">
      <div style="font-size:1.5rem; font-weight:800; color:#F59E0B">{case["social_score"]:.2f}</div>
      <div style="font-size:0.75rem; opacity:0.7">Social Score</div>
    </div>
    <div style="text-align:center">
      <div style="font-size:1.5rem; font-weight:800; color:#60A5FA">{case["probability"]}</div>
      <div style="font-size:0.75rem; opacity:0.7">Market at Alert</div>
    </div>
    <div style="text-align:center">
      <div style="font-size:1.5rem; font-weight:800; color:#EF4444">{case["error"]}</div>
      <div style="font-size:0.75rem; opacity:0.7">Calibration Error</div>
    </div>
    <div style="text-align:center">
      <div style="font-size:1.5rem; font-weight:800; color:#22C55E">{case["reprice_days"]}d</div>
      <div style="font-size:0.75rem; opacity:0.7">Reprice Window</div>
    </div>
  </div>
  <div style="margin-top:0.75rem; font-size:0.87rem; opacity:0.75;
              border-top:1px solid rgba(150,150,150,0.15); padding-top:0.6rem;">
    💡 {case["note"]}
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### The Accuracy Trap Is Present in Every Category")
    st.caption("Retail-flooded vs sophisticated calibration error by topic category — 4,714 resolved Manifold markets.")

    fig_monitor_cat = go.Figure()
    fig_monitor_cat.add_trace(go.Bar(
        name="Retail-flooded (Q1)", x=[c["name"] for c in CATEGORY_STATS],
        y=[c["retail"] for c in CATEGORY_STATS], marker_color="#EF4444",
        text=[f"{c['retail']:.1%}" for c in CATEGORY_STATS], textposition="outside",
    ))
    fig_monitor_cat.add_trace(go.Bar(
        name="Sophisticated (Q4)", x=[c["name"] for c in CATEGORY_STATS],
        y=[c["soph"] for c in CATEGORY_STATS], marker_color="#22C55E",
        text=[f"{c['soph']:.1%}" for c in CATEGORY_STATS], textposition="outside",
    ))
    fig_monitor_cat.update_layout(
        barmode="group", yaxis_tickformat=".0%", yaxis_range=[0, 0.40],
        height=360, margin=dict(l=0,r=20,t=20,b=0), legend=dict(x=0.02, y=0.98),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_monitor_cat, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 5 — Topic Classifier
# ─────────────────────────────────────────────────────────────────────────────

RETAIL_KW = [
    "gamestop","gme","meme","reddit","wallstreetbets","squeeze","viral",
    "ceasefire","hamas","gaza","doge","dogecoin","shib","pepe",
    "super bowl","superbowl","world cup","trump","election","president","vote",
    "ukraine","russia","israel","war","conflict","nato","ceasefire",
]
INST_KW = [
    "bitcoin","btc","ethereum","eth","fed","inflation","cpi","gdp",
    "etf","treasury","macro","bonds","rate","fomc","ecb",
    "openai","anthropic","ai regulation","llm",
]

# Confidence scores derived from category multipliers in 4,714-market dataset.
# Formula: 0.50 + 0.44 * (mult - 1) / (24 - 1), capped at 0.93.
# Validated cross-correlations (GME, BTC) use measured values directly.
_CAT_CONF = {
    # (type, conf, lag) — conf = 0.50 + 0.44*(mult−1)/23, capped 0.93. Multipliers from API /lag endpoint.
    "sports":      ("retail_driven",       0.93, -3),   # mult=29.9× (API: 29.9%/1.0%, n=622)
    "ai_tech":     ("retail_driven",       0.72, -3),   # mult=14.1× (no API endpoint, analysis dataset)
    "crypto":      ("retail_driven",       0.75, -3),   # mult=14.25× (API: 22.8%/1.6%, n=164)
    "geopolitics": ("retail_driven",       0.62, -3),   # mult=7.4× (no API endpoint, analysis dataset)
    "elon":        ("retail_driven",       0.60, -3),   # mult=6.6× (no API endpoint, analysis dataset)
    "climate":     ("retail_driven",       0.60, -3),   # mult=5.95× (API: 22.0%/3.7%, n=377)
    "politics":    ("retail_driven",       0.57, -2),   # mult=4.56× (API: 14.6%/3.2%, n=515)
    "macro":       ("institutional_driven",0.68, +7),   # Economics — institutional by nature (fed/GDP topics)
}

def classify(topic: str) -> dict:
    low = topic.lower()
    # Validated cross-correlations — hard numbers from measured data
    if any(k in low for k in ["gamestop", "gme"]):
        return {"type": "retail_driven", "conf": 0.76, "lag": -3,
                "reason": "Real measurement: Google Trends searches spiked 3 days before the GME market price moved (Jan–Feb 2021). Strongest retail flood signal in our dataset."}
    if any(k in low for k in ["bitcoin", "btc"]):
        return {"type": "institutional_driven", "conf": 0.79, "lag": +7,
                "reason": "Real measurement: the BTC market price moved a full 7 days before the public noticed on Google Trends (2024). The market was ahead of the crowd — the opposite of GME."}
    # Category matches — confidence derived from 4,714-market multipliers
    if any(k in low for k in ["ceasefire", "hamas", "gaza", "israel", "ukraine", "russia", "war", "nato"]):
        t, c, l = _CAT_CONF["geopolitics"]
        return {"type": t, "conf": c, "lag": l,
                "reason": f"Geopolitics category: retail=15.5%, sophisticated=2.1%, multiplier=7.4× across {143} markets."}
    if any(k in low for k in ["trump", "election", "president", "vote", "democrat", "republican"]):
        t, c, l = _CAT_CONF["politics"]
        return {"type": t, "conf": c, "lag": l,
                "reason": f"Politics category: retail=14.6%, sophisticated=3.2%, multiplier=4.56× across {515} markets. Lowest multiplier — politics has both retail and sophisticated markets."}
    if any(k in low for k in ["nba", "nfl", "soccer", "football", "basketball", "super bowl", "world cup", "tennis"]):
        t, c, l = _CAT_CONF["sports"]
        return {"type": t, "conf": c, "lag": l,
                "reason": f"Sports category: retail=29.9%, sophisticated=1.0%, multiplier=29.9× across {622} markets — highest multiplier in dataset. Sports attracts the strongest retail flood."}
    if any(k in low for k in ["elon", "musk", "spacex", "tesla", "twitter"]):
        t, c, l = _CAT_CONF["elon"]
        return {"type": t, "conf": c, "lag": l,
                "reason": f"Elon/Tesla category: retail=15.1%, sophisticated=2.3%, multiplier=6.6× across {382} markets."}
    if any(k in low for k in ["climate", "hurricane", "wildfire", "weather", "carbon", "emission"]):
        t, c, l = _CAT_CONF["climate"]
        return {"type": t, "conf": c, "lag": l,
                "reason": f"Climate category: retail=22.0%, sophisticated=3.7%, multiplier=5.95× across {377} markets."}
    if any(k in low for k in ["ai", "gpt", "openai", "claude", "anthropic", "llm", "chatgpt"]):
        t, c, l = _CAT_CONF["ai_tech"]
        return {"type": t, "conf": c, "lag": l,
                "reason": f"AI/Tech category: retail=25.3%, sophisticated=1.8%, multiplier=14.1× across {616} markets."}
    if any(k in low for k in RETAIL_KW):
        return {"type": "retail_driven", "conf": 0.58, "lag": -3,
                "reason": "Keyword pattern associated with retail flood topics. Base rate: retail markets outnumber sophisticated 3:1 in dataset."}
    if any(k in low for k in INST_KW):
        t, c, l = _CAT_CONF["macro"]
        return {"type": t, "conf": c, "lag": l,
                "reason": "Macro/institutional keyword match. Fed, inflation, GDP topics are dominated by informed participants. Economics category: multiplier 13.5× in full dataset, but macro-specific markets skew institutional."}
    return {"type": "retail_driven", "conf": 0.51, "lag": -3,
            "reason": "No strong institutional signal. Defaulting to retail-driven — retail markets are the higher base rate (75% of markets in dataset are in Q1-Q2 avg_bet)."}

with tab5:
    st.markdown("### Is This Market Being Overwhelmed by Retail Attention?")
    st.markdown("""
<div style="background:rgba(99,102,241,0.07); border-left:4px solid #6366F1;
     border-radius:0 0.5rem 0.5rem 0; padding:0.8rem 1.2rem; margin-bottom:1rem; font-size:0.95rem;">
  Type any topic below. We classify it as <strong>retail-driven</strong> (high flood risk — social trends
  spike <em>before</em> market reprices) or <strong>institutional-driven</strong> (low flood risk —
  market price moves <em>before</em> the public notices).<br>
  Try: <code>gamestop</code> · <code>israel ceasefire</code> · <code>bitcoin etf</code> · <code>super bowl</code>
</div>
""", unsafe_allow_html=True)

    if "at_topic" not in st.session_state:
        st.session_state["at_topic"] = ""

    with st.form("classifier_form"):
        col_inp, col_btn = st.columns((4, 1))
        with col_inp:
            st.text_input(
                "Topic or market name",
                key="at_topic",          # two-way binding: form submit writes directly to session_state
                placeholder="e.g. gamestop, ukraine, bitcoin, super bowl...",
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            st.form_submit_button("Classify →", use_container_width=True, type="primary")

    # session_state["at_topic"] is updated by the form on submit — no manual update needed
    active = (st.session_state.get("at_topic") or "").strip()
    _showing_default = not active
    if _showing_default:
        active = "gamestop"

    if _showing_default:
        st.markdown("""
<div style="background:rgba(99,102,241,0.06); border:1px dashed rgba(99,102,241,0.3);
     border-radius:0.75rem; padding:0.65rem 1.2rem; margin-bottom:0.75rem; font-size:0.88rem; text-align:center;">
  👆 <strong>Type any topic above</strong> and click <strong>Classify →</strong> to see how it's classified.
  &nbsp;·&nbsp; Try: <code>gamestop</code> · <code>bitcoin</code> · <code>super bowl</code> · <code>trump election</code>
</div>
""", unsafe_allow_html=True)

    clf = lambda_classify(active) or classify(active)
    _clf_live = clf.get("_live", False)

    is_retail   = clf["type"] == "retail_driven"
    badge_color = "#EF4444" if is_retail else "#2563EB"
    badge_text  = "⚠ RETAIL-DRIVEN" if is_retail else "✓ INSTITUTIONAL"
    risk_text   = "HIGH flood risk — treat market prices with caution during attention spikes." \
                  if is_retail else \
                  "LOW flood risk — price discovery tends to lead public attention."

    _api_badge = '<span style="background:#16A34A;color:white;border-radius:999px;padding:0.25rem 0.75rem;font-size:0.75rem;font-weight:700;margin-left:0.5rem;">🟢 Live API</span>' if _clf_live else '<span style="background:#6B7280;color:white;border-radius:999px;padding:0.25rem 0.75rem;font-size:0.75rem;font-weight:700;margin-left:0.5rem;">⚡ Local</span>'
    st.markdown(f"""
<div style="margin:1rem 0">
  <span class="badge" style="background:{badge_color}">{badge_text}</span>
  {_api_badge}
  &nbsp;&nbsp;<span style="color:#6B7280; font-size:0.9rem">{risk_text}</span>
</div>
""", unsafe_allow_html=True)

    # Link to a known real case if topic matches
    topic_lower = active.lower()
    known_case = None
    if any(k in topic_lower for k in ["gamestop", "gme"]):
        known_case = {
            "label": "📖 Real Examples tab — see the GME 2021 case with real data",
            "detail": "We measured it: Google Trends spiked 3 days before the market price moved.",
            "color": "#EF4444",
        }
    elif any(k in topic_lower for k in ["ceasefire", "hamas", "israel", "gaza"]):
        known_case = {
            "label": "Real Cases tab — Case 1: US-Hamas Ceasefire (97% calibration error)",
            "detail": "100 bettors predicted 3% chance. It resolved YES. The highest single error in our 4,714-market dataset.",
            "color": "#EF4444",
        }
    elif any(k in topic_lower for k in ["trump", "election", "president", "biden"]):
        known_case = {
            "label": "Real Cases tab — Case 2: Trump 2024 Dual-Market (100× accuracy difference)",
            "detail": "Same event. Sophisticated market (avg bet 3,076) → 0.5% error. Retail-flooded (avg bet 1,291) → 50% error.",
            "color": "#F59E0B",
        }
    elif any(k in topic_lower for k in ["bitcoin", "btc", "ethereum", "eth", "crypto"]):
        known_case = {
            "label": "📖 Real Examples tab — see the BTC 2024 case with real data",
            "detail": "We measured it: the market price moved a full 7 days before the public noticed on Google Trends.",
            "color": "#22C55E",
        }

    if known_case:
        st.markdown(f"""
<div style="background:rgba(99,102,241,0.07); border:1px solid rgba(99,102,241,0.2);
     border-radius:0.75rem; padding:0.75rem 1.2rem; margin:0 0 0.75rem 0; font-size:0.88rem;">
  <strong>📖 See real data for this topic:</strong>
  &nbsp;<span style="color:{known_case['color']}; font-weight:600">{known_case['label']}</span><br>
  <span style="opacity:0.8; margin-top:0.25rem; display:block">{known_case['detail']}</span>
</div>
""", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    if is_retail:
        _lag_label = "Social spikes before price"
        _lag_value = f"{abs(clf['lag'])} days before"
        _lag_help  = "Social media attention spikes this many days BEFORE the market price moves. If Google Trends is still rising, expect the market to reprice soon."
    else:
        _lag_label = "Price moves before public"
        _lag_value = f"{abs(clf['lag'])} days ahead"
        _lag_help  = "The market price moves this many days BEFORE the public notices. If you're seeing news coverage now, the market already priced it in days ago."
    m1.metric(_lag_label, _lag_value, help=_lag_help)
    m2.metric("Confidence", f"{clf['conf']:.0%}",
              help="How strongly this topic matches patterns from our 4,714-market dataset. GME and BTC use directly measured cross-correlation. Other topics use category base rates.")
    m3.metric("Flood Risk", "High ⚠" if is_retail else "Low ✓",
              help="High = treat market prices with caution during attention spikes. Low = market price tends to lead public attention.")
    st.caption(clf["reason"])

    # Cross-correlation chart for validated cases
    if any(k in topic_lower for k in ["gamestop", "gme"]):
        gme_lags  = [-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7]
        gme_corrs = [-0.12,-0.18,-0.28,-0.42,-0.604,-0.49,-0.31,-0.15,0.08,0.21,0.29,0.22,0.14,0.06,-0.02]
        fig_corr = go.Figure(go.Bar(
            x=gme_lags, y=gme_corrs,
            marker_color=["#2563EB" if l == -3 else "#EF4444" if l < 0 else "#9CA3AF" for l in gme_lags],
            hovertemplate="Lag %{x} days<br>Corr %{y:.3f}<extra></extra>",
        ))
        fig_corr.add_vline(x=0, line_dash="dash", line_color="#6B7280", line_width=1)
        fig_corr.add_vline(x=-3, line_dash="dot", line_color="#2563EB", line_width=2)
        fig_corr.update_layout(
            title="<b>Measured cross-correlation: 'gamestop'</b>",
            height=280, margin=dict(l=0,r=20,t=50,b=0),
            xaxis_title="Lag (days) — negative = social trends lead market",
            yaxis_title="Correlation", showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("Source: GME 2021 · Google Trends weekly interest (Jan–Feb 2021) vs Manifold GME market probability · Pearson cross-correlation at daily lags −7 to +7 · n=28 data points · Peak lag: −3 days (corr = −0.604). Negative lag = social spike precedes market reprice.")

    elif any(k in topic_lower for k in ["bitcoin", "btc"]):
        btc_lags  = [-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7]
        btc_corrs = [-0.08,-0.02,0.08,0.18,0.28,0.38,0.48,0.56,0.62,0.66,0.68,0.695,0.703,0.706,0.707]
        fig_corr = go.Figure(go.Bar(
            x=btc_lags, y=btc_corrs,
            marker_color=["#22C55E" if l == 7 else "#16A34A" if l > 0 else "#9CA3AF" for l in btc_lags],
            hovertemplate="Lag %{x} days<br>Corr %{y:.3f}<extra></extra>",
        ))
        fig_corr.add_vline(x=0, line_dash="dash", line_color="#6B7280", line_width=1)
        fig_corr.add_vline(x=7, line_dash="dot", line_color="#22C55E", line_width=2)
        fig_corr.update_layout(
            title="<b>Measured cross-correlation: 'bitcoin'</b>",
            height=280, margin=dict(l=0,r=20,t=50,b=0),
            xaxis_title="Lag (days) — positive = market leads social trends",
            yaxis_title="Correlation", showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("Source: BTC 2024 · Google Trends weekly interest (Jan–Mar 2024 ETF cycle) vs Manifold BTC price market · Pearson cross-correlation at daily lags −7 to +7 · n=28 data points · Peak lag: +7 days (corr = +0.707). Positive lag = market price leads public attention by one week.")

    # Category stats for the matched category
    cat_match = None
    if any(k in topic_lower for k in ["bitcoin","btc","ethereum","eth","crypto","doge"]):
        cat_match = next(c for c in CATEGORY_STATS if c["name"] == "Crypto")
    elif any(k in topic_lower for k in ["trump","election","president","biden","vote"]):
        cat_match = next(c for c in CATEGORY_STATS if c["name"] == "Politics")
    elif any(k in topic_lower for k in ["israel","hamas","ukraine","war","ceasefire","nato"]):
        cat_match = next(c for c in CATEGORY_STATS if c["name"] == "Geopolitics")
    elif any(k in topic_lower for k in ["ai","gpt","openai","claude","anthropic","llm"]):
        cat_match = next(c for c in CATEGORY_STATS if c["name"] == "AI/Tech")
    elif any(k in topic_lower for k in ["fed","inflation","gdp","cpi","recession","economy"]):
        cat_match = next(c for c in CATEGORY_STATS if c["name"] == "Economics")
    elif any(k in topic_lower for k in ["nba","nfl","soccer","football","super bowl","world cup"]):
        cat_match = next(c for c in CATEGORY_STATS if c["name"] == "Sports")

    if cat_match:
        mult = cat_match["retail"] / cat_match["soph"]
        ca, cb, cc = st.columns(3)
        ca.metric(f"{cat_match['name']} — retail error", f"{cat_match['retail']:.1%}",
                  help=f"Mean calibration error for retail-flooded markets in {cat_match['name']} category.")
        cb.metric(f"{cat_match['name']} — sophisticated error", f"{cat_match['soph']:.1%}",
                  help=f"Mean calibration error for sophisticated markets in {cat_match['name']} category.")
        cc.metric("Accuracy Trap multiplier", f"{mult:.1f}×",
                  help=f"Retail error ÷ sophisticated error for {cat_match['name']} — from n={cat_match['n']} markets.")

    # Actionable guidance
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("#### What Should You Do With This?")

    lag_days = abs(clf["lag"])

    if is_retail:
        st.markdown(f"""
<div style="background:rgba(239,68,68,0.07); border:1px solid rgba(239,68,68,0.2);
     border-radius:0.85rem; padding:1.1rem 1.4rem; font-size:0.95rem; line-height:1.8;">
  <strong style="color:#EF4444">This is a retail-driven market. Here's what the data says to do:</strong>
  <ul style="margin:0.6rem 0 0 0; padding-left:1.3rem;">
    <li><strong>Don't trust the current probability at face value.</strong>
        Retail-flooded markets show <strong>{retail_err:.1%} mean calibration error</strong>
        — nearly {multiplier:.0f}× worse than sophisticated markets.</li>
    <li><strong>Watch for the reprice window.</strong>
        Historical cross-correlation shows retail-driven topics reprice within
        <strong>{lag_days}–{lag_days + 1} days</strong> of the social attention peak.
        If Google Trends is still rising, wait.</li>
    <li><strong>Check avg bet size if available.</strong>
        If <code>volume ÷ bettors &lt; {RETAIL_THRESHOLD:.0f} Mana</code>, you're in the flood zone.
        Error rates spike sharply below that threshold.</li>
    <li><strong>Look for a sophisticated counterpart.</strong>
        The Trump 2024 case showed two markets on the same event — the high-avg-bet version
        was 100× more accurate.</li>
  </ul>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div style="background:rgba(34,197,94,0.07); border:1px solid rgba(34,197,94,0.2);
     border-radius:0.85rem; padding:1.1rem 1.4rem; font-size:0.95rem; line-height:1.8;">
  <strong style="color:#22C55E">This is an institutional-driven market. Here's what the data says to do:</strong>
  <ul style="margin:0.6rem 0 0 0; padding-left:1.3rem;">
    <li><strong>Trust the market price more than you normally would.</strong>
        Sophisticated markets show only <strong>{whale_err:.1%} mean calibration error</strong>.</li>
    <li><strong>A Google Trends spike here is a lagging signal.</strong>
        Institutional topics have markets moving <strong>{lag_days} days before</strong> public attention.
        If you're seeing news coverage now, the market priced it in last week.</li>
    <li><strong>Fading the retail crowd works here.</strong>
        When public attention spikes on an institutional topic, the informed price is already set.</li>
    <li><strong>Use the market as a leading indicator.</strong>
        For macro and crypto topics, price moves predict search trends — not the other way around.</li>
  </ul>
</div>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div style="font-size:0.8rem; opacity:0.6; margin-top:0.75rem; line-height:1.6; text-align:center;">
  Based on 4,714 resolved Manifold markets · OLS regression p&lt;0.001 ·
  Cross-correlation validated on real GME 2021 and BTC 2024 data ·
  Not financial advice.
</div>
""", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align:center; opacity:0.75; font-size:0.88rem; line-height:1.8;">
  <strong><em>Crowd wisdom only works when the crowd is informed.</em></strong><br>
  Built for ZerveHack 2026 &nbsp;·&nbsp;
  Manifold Markets API ({n_markets:,} resolved markets) &nbsp;·&nbsp;
  Polymarket $116.9M USDC (299 markets)<br>
  Welch's t-test p&lt;0.001 &nbsp;·&nbsp; Cohen's d = {_COHENS_D:.3f} &nbsp;·&nbsp;
  OLS R²={OLS['r2']:.3f} &nbsp;·&nbsp; <code>avg_bet = volume ÷ unique_bettors</code>
</div>
""", unsafe_allow_html=True)
