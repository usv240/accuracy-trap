"""
The Accuracy Trap — FastAPI for Zerve deployment.

Uses `from zerve import variables` to pull pre-computed results directly
from the Zerve canvas (zerve_notebook.py). Falls back to hardcoded values
when running locally or if the canvas variable isn't available.

Paste this as main.py in a Zerve FastAPI (Org) deployment.
No local file imports — single self-contained file.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from mangum import Mangum  # AWS Lambda adapter
except ImportError:
    Mangum = None  # type: ignore[assignment,misc]

try:
    from pytrends.request import TrendReq
except ImportError:
    TrendReq = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Pull pre-computed results from Zerve canvas
# ---------------------------------------------------------------------------
# In Zerve: open your canvas (zerve_notebook.py), find the block that
# computes `results` and note its block name. Replace "zerve_notebook" below.
# Variable name is "results" (the dict saved to accuracy_trap_results.json).

_CANVAS_BLOCK = "zerve_notebook"   # ← update to your actual block name in Zerve
_CANVAS_VAR   = "results"          # ← the Python variable name in that block

def _load_canvas_results() -> dict[str, Any] | None:
    """Try to pull `results` from the Zerve canvas. Returns None if unavailable."""
    try:
        from zerve import variables  # type: ignore[import]
        data = variables(_CANVAS_BLOCK, _CANVAS_VAR)
        if isinstance(data, dict) and "headline" in data:
            return data
    except Exception:
        pass
    return None

# Hardcoded fallback — from zerve_notebook.py run on 4,714 Manifold markets
_HARDCODED_RESULTS: dict[str, Any] = {
    "headline": {
        "retail_flood_calibration_error": 0.2227,
        "sophisticated_calibration_error": 0.0203,
        "error_multiplier": 10.97,
        "n_markets_analyzed": 4714,
        "data_source": "Manifold Markets — 4,714 resolved binary markets",
        "conclusion": (
            "Markets flooded with retail traders show 22.3% calibration error "
            "— 10.97x worse than sophisticated markets (2.0%). "
            "The driver is WHO bets, not how many are watching."
        ),
    },
    "attention_buckets": [
        {"label": "Micro-bet\n(Retail flood)",  "mean_calibration_error": 0.2227, "sample_size": 1179, "median_avg_bet": 52.0,  "median_nr_bettors": 10},
        {"label": "Small-bet",                  "mean_calibration_error": 0.0780, "sample_size": 1178, "median_avg_bet": 136.0, "median_nr_bettors": 17},
        {"label": "Large-bet",                  "mean_calibration_error": 0.0360, "sample_size": 1178, "median_avg_bet": 297.0, "median_nr_bettors": 25},
        {"label": "Whale-bet\n(Sophisticated)", "mean_calibration_error": 0.0203, "sample_size": 1179, "median_avg_bet": 720.0, "median_nr_bettors": 25},
    ],
    "retail_flood_detector": {
        "metric":                             "avg_bet_size = volume / nr_bettors",
        "retail_threshold_percentile":        25,
        "retail_threshold_value":             92.6,
        "high_attention_retail_error":        0.125,
        "high_attention_sophisticated_error": 0.025,
        "ratio":                              5.10,
    },
}

# Try canvas first, fall back to hardcoded
_canvas_results = _load_canvas_results()
_ACCURACY_TRAP_DATA: dict[str, Any] = _canvas_results if _canvas_results else _HARDCODED_RESULTS
_DATA_SOURCE = "Zerve canvas (live)" if _canvas_results else "hardcoded (canvas unavailable)"

# ---------------------------------------------------------------------------
# Category stats — hardcoded from 4,714-market dataset (/lag API endpoint values)
# ---------------------------------------------------------------------------

_CATEGORY_STATS: dict[str, Any] = {
    "political": {
        "display_name": "Political", "n": 515,
        "mean_calibration_error": 0.0760, "retail_pct": 0.40,
        "retail_error": 0.1460, "sophisticated_error": 0.0320,
    },
    "sports": {
        "display_name": "Sports", "n": 622,
        "mean_calibration_error": 0.0870, "retail_pct": 0.55,
        "retail_error": 0.2990, "sophisticated_error": 0.0100,
    },
    "crypto": {
        "display_name": "Crypto", "n": 164,
        "mean_calibration_error": 0.0850, "retail_pct": 0.47,
        "retail_error": 0.2280, "sophisticated_error": 0.0160,
    },
    "economic": {
        "display_name": "Economic", "n": 318,
        "mean_calibration_error": 0.0840, "retail_pct": 0.45,
        "retail_error": 0.2430, "sophisticated_error": 0.0180,
    },
    "climate": {
        "display_name": "Climate", "n": 377,
        "mean_calibration_error": 0.0800, "retail_pct": 0.42,
        "retail_error": 0.2200, "sophisticated_error": 0.0370,
    },
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: Literal["ok"]
    data_source: str

class LagResponse(BaseModel):
    category:               str
    display_name:           str
    n:                      int
    mean_calibration_error: float
    retail_pct:             float
    retail_error:           float | None
    sophisticated_error:    float | None
    error_multiplier:       float | None
    data_source:            str

class ClassifyResponse(BaseModel):
    topic:             str
    market_type:       Literal["retail_driven", "institutional_driven"]
    expected_lag_days: int
    confidence:        float
    reasoning:         str

class LiveAlert(BaseModel):
    market:                  str
    token_id:                str
    social_score:            float
    social_score_source:     str
    hours_since_signal:      float | None
    expected_reprice_window: str
    current_probability:     float | None
    confidence:              Literal["high", "medium", "low"]

class LiveAlertsResponse(BaseModel):
    alerts:       list[LiveAlert]
    generated_at: str
    data_sources: str
    note:         str

class AttentionBucket(BaseModel):
    label:                  str
    mean_calibration_error: float
    sample_size:            int
    median_avg_bet:         float
    median_nr_bettors:      int | None = None

class AccuracyTrapResponse(BaseModel):
    headline:              dict[str, Any]
    attention_buckets:     list[AttentionBucket]
    retail_flood_detector: dict[str, Any]

class LagAnalysis(BaseModel):
    avg_lag_days:        int
    signal_detected:     bool
    signal_detected_at:  str | None
    expected_reprice_by: str | None

class ExplainResponse(BaseModel):
    topic:                      str
    market_type:                Literal["retail_driven", "institutional_driven"]
    current_social_score:       float | None
    social_score_source:        str
    current_market_probability: float | None
    lag_analysis:               LagAnalysis
    top_factors:                list[str]
    classification_validation:  str

# ---------------------------------------------------------------------------
# Classification constants
# ---------------------------------------------------------------------------

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "sports": [
        "nba", "nfl", "mlb", "nhl", "ufc", "final", "finals", "playoff", "cup",
        "match", "tournament", "goal", "team", "soccer", "football", "basketball",
        "tennis", "championship", "superbowl", "super bowl", "sports",
    ],
    "political": [
        "election", "ceasefire", "ukraine", "president", "senate", "congress",
        "vote", "war", "peace", "tariff", "policy", "trump", "biden", "democrat",
        "republican", "israel", "hamas", "nato", "conflict", "gaza",
    ],
    "economic": [
        "inflation", "fed", "rates", "gdp", "jobs", "recession", "cpi",
        "unemployment", "economy", "treasury", "yield",
    ],
    "crypto": [
        "bitcoin", "btc", "ethereum", "eth", "solana", "doge", "crypto", "etf",
        "token", "defi", "blockchain", "nft",
    ],
    "climate": [
        "climate", "hurricane", "storm", "wildfire", "heat", "temperature",
        "carbon", "emissions", "epa", "eia", "weather", "flood", "drought",
    ],
}

RETAIL_SIGNATURE_KEYWORDS = [
    "gamestop", "gme", "meme", "reddit", "wallstreetbets", "squeeze", "viral",
    "ceasefire", "hamas", "gaza", "doge", "dogecoin", "shib", "pepe", "memecoin",
    "super bowl", "superbowl", "world cup",
]

INSTITUTIONAL_SIGNATURE_KEYWORDS = [
    "bitcoin", "btc", "ethereum", "eth", "fed", "inflation",
    "cpi", "gdp", "etf", "treasury", "macro",
]

_STOP_WORDS = {
    "will","the","a","an","be","is","are","was","were","to","of","in","on","at",
    "by","for","with","from","this","that","it","as","do","have","has","had",
    "would","could","should","may","might","can","than","more","most","much",
    "any","all","some","or","and","but","if","when","before","after","during",
    "whether","what","who","how","there","their","they","we","our","its","his",
    "her","them","him","she","he","us","you","your","i","my","me","not","no",
    "yes","win","lose","happen","occur","reach","become","get","go","make","take",
    "end","start","begin","finish","pass","fail","ever","never","still","already",
    "yet","just","only","first","second","2020","2021","2022","2023","2024","2025",
    "2026","year","month","week","day",
}

GAMMA_API = "https://gamma-api.polymarket.com"

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)

def _iso_now() -> str:
    return _utc_now().isoformat().replace("+00:00", "Z")

def _r(v: float, d: int = 3) -> float:
    return float(round(float(v), d))

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _infer_category(topic: str) -> str | None:
    lowered = topic.lower().strip()
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(k in lowered for k in kws):
            return cat
    return None


def classify_topic(topic: str) -> dict[str, Any]:
    normalized = topic.strip()
    lowered    = normalized.lower()
    category   = _infer_category(normalized)

    if "gamestop" in lowered or "gme" in lowered:
        return {"topic": normalized, "market_type": "retail_driven", "expected_lag_days": -3,
                "confidence": 0.76, "reasoning": "Validated: GME 2021 — Google Trends led market by 3 days (corr=−0.604).",
                "category": category or "sports", "validation": "measured"}

    if "bitcoin" in lowered or "btc" in lowered:
        return {"topic": normalized, "market_type": "institutional_driven", "expected_lag_days": 7,
                "confidence": 0.79, "reasoning": "Validated: BTC 2024 — market price led Google Trends by 7 days (corr=+0.707).",
                "category": category or "crypto", "validation": "measured"}

    if any(k in lowered for k in ["trump", "election", "president", "vote"]):
        return {"topic": normalized, "market_type": "retail_driven", "expected_lag_days": -3,
                "confidence": 0.71,
                "reasoning": ("Political/election markets: retail-flooded versions had 100× higher "
                              "calibration error than sophisticated counterparts (Trump 2024: 50% vs 0.5%)."),
                "category": category or "political", "validation": "category_stats"}

    if any(k in lowered for k in RETAIL_SIGNATURE_KEYWORDS):
        return {"topic": normalized, "market_type": "retail_driven", "expected_lag_days": -3,
                "confidence": 0.68, "reasoning": "Keyword signature matches retail-driven topics.",
                "category": category, "validation": "keyword_match"}

    if any(k in lowered for k in INSTITUTIONAL_SIGNATURE_KEYWORDS):
        return {"topic": normalized, "market_type": "institutional_driven", "expected_lag_days": 7,
                "confidence": 0.72, "reasoning": "Keyword signature matches institutional-driven topics (macro, Fed, ETF).",
                "category": category, "validation": "keyword_match"}

    if category and category in _CATEGORY_STATS:
        stats      = _CATEGORY_STATS[category]
        retail_pct = stats["retail_pct"]
        is_retail  = retail_pct > 0.5
        return {
            "topic": normalized,
            "market_type": "retail_driven" if is_retail else "institutional_driven",
            "expected_lag_days": -3 if is_retail else 7,
            "confidence": _r(min(0.75, 0.5 + abs(retail_pct - 0.5)), 2),
            "reasoning": (f"{stats['display_name']} markets: {retail_pct:.0%} retail-flooded, "
                          f"mean error {stats['mean_calibration_error']:.1%}."),
            "category": category, "validation": "category_stats",
        }

    return {"topic": normalized, "market_type": "retail_driven", "expected_lag_days": -3,
            "confidence": 0.55, "reasoning": "No institutional signature detected; defaulting to retail regime.",
            "category": None, "validation": "default"}

# ---------------------------------------------------------------------------
# Google Trends
# ---------------------------------------------------------------------------

def _extract_trend_topic(market_name: str) -> str:
    words = market_name.lower().replace("?", "").replace("'", "").split()
    meaningful = [w for w in words if w not in _STOP_WORDS and len(w) > 2]
    return " ".join(meaningful[:3]) if meaningful else market_name[:40]


def _get_wiki_trend_data(topic: str) -> dict[str, Any] | None:
    """Wikipedia Pageviews API — free, no auth, no rate limits. Fallback for Google Trends."""
    try:
        import numpy as np
        from datetime import timedelta
        now   = _utc_now()
        end   = (now - timedelta(days=1)).strftime("%Y%m%d")
        start = (now - timedelta(days=14)).strftime("%Y%m%d")
        article = topic.strip().replace(" ", "_").title()
        for candidate in [article, topic.strip().split()[0].title()]:
            url = (
                f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
                f"/en.wikipedia/all-access/all-agents/{candidate}/daily/{start}/{end}"
            )
            r = requests.get(url, timeout=8, headers={"User-Agent": "AccuracyTrapBot/1.0"})
            if r.status_code == 200:
                items = r.json().get("items", [])
                if len(items) >= 4:
                    views    = np.array([float(i["views"]) for i in items])
                    split    = max(1, len(views) // 2)
                    baseline = float(views[:split].mean())
                    recent   = float(views[split:].mean())
                    if baseline < 100:
                        return None
                    score = float(np.clip(((recent - baseline) / (baseline + 1e-6)) + 0.5, 0.0, 1.0))
                    peak  = float(views.max())
                    hours: float | None = None
                    if peak >= baseline * 1.5:
                        for idx, item in enumerate(items):
                            if float(item["views"]) >= peak * 0.6:
                                ts    = datetime.strptime(item["timestamp"][:8], "%Y%m%d")
                                h     = float((_utc_now().replace(tzinfo=None) - ts).total_seconds() / 3600)
                                hours = _r(max(1.0, min(h, 336.0)), 1)
                                break
                    return {"social_score": _r(score, 3), "hours_since_signal": hours, "source": "wikipedia_pageviews", "search_term": candidate}
        return None
    except Exception:
        return None


def _get_trend_data(topic: str) -> dict[str, Any] | None:
    """Google Trends first, Wikipedia Pageviews as fallback."""
    # 1. Try Google Trends
    if TrendReq is not None:
        try:
            import numpy as np
            trends = TrendReq(hl="en-US", tz=360, timeout=(10, 20), retries=1, backoff_factor=0.3)
            trends.build_payload([topic[:100]], timeframe="now 7-d", geo="US")
            frame = trends.interest_over_time()
            if not frame.empty and len(frame) >= 4:
                frame  = frame.drop(columns=["isPartial"], errors="ignore")
                col    = frame.columns[0]
                values = frame[col].values.astype(float)
                split  = max(1, len(values) // 2)
                baseline, recent = float(values[:split].mean()), float(values[split:].mean())
                if baseline >= 1.0:
                    score = float(np.clip(((recent - baseline) / (baseline + 1e-6)) + 0.5, 0.0, 1.0))
                    peak  = float(frame[col].max())
                    hours: float | None = None
                    if peak >= 10:
                        import pandas as pd
                        frame.index = pd.to_datetime(frame.index)
                        above = frame[frame[col] >= peak * 0.6]
                        if not above.empty:
                            first_spike = above.index[0].to_pydatetime().replace(tzinfo=None)
                            h = float((_utc_now().replace(tzinfo=None) - first_spike).total_seconds() / 3600)
                            hours = _r(max(1.0, min(h, 168.0)), 1)
                    return {"social_score": _r(score, 3), "hours_since_signal": hours, "source": "google_trends"}
        except Exception:
            pass
    # 2. Fallback — Wikipedia Pageviews
    return _get_wiki_trend_data(topic)

# ---------------------------------------------------------------------------
# Polymarket helpers
# ---------------------------------------------------------------------------

def _extract_title(market: dict[str, Any]) -> str:
    for key in ("question", "title", "name", "market"):
        v = market.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return "Untitled prediction market"

def _extract_token_id(market: dict[str, Any]) -> str:
    for key in ("token_id", "tokenId", "conditionId"):
        v = market.get(key)
        if v:
            return str(v)
    clob = market.get("clobTokenIds")
    if isinstance(clob, str):
        try:
            parsed = json.loads(clob)
            if isinstance(parsed, list) and parsed:
                return str(parsed[0])
        except json.JSONDecodeError:
            return clob
    if isinstance(clob, list) and clob:
        return str(clob[0])
    return str(market.get("id", "unknown"))

def _extract_probability(market: dict[str, Any]) -> float | None:
    for key in ("currentProbability", "probability", "lastTradePrice", "price", "bestAsk"):
        v = market.get(key)
        if v is None:
            continue
        try:
            return _r(float(v), 3)
        except (TypeError, ValueError):
            continue
    op = market.get("outcomePrices")
    if isinstance(op, str):
        try:
            parsed = json.loads(op)
            if isinstance(parsed, list) and parsed:
                return _r(float(parsed[0]), 3)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    if isinstance(op, list) and op:
        try:
            return _r(float(op[0]), 3)
        except (TypeError, ValueError):
            pass
    return None

# ---------------------------------------------------------------------------
# Live alerts
# ---------------------------------------------------------------------------

def get_live_alerts(limit: int = 8) -> dict[str, Any]:
    alerts: list[dict[str, Any]] = []
    try:
        r = requests.get(
            f"{GAMMA_API}/markets",
            params={"limit": max(limit * 4, 30), "closed": False, "active": True},
            timeout=5,
        )
        r.raise_for_status()
        payload = r.json()
        markets = payload if isinstance(payload, list) else payload.get("markets", payload.get("data", []))
    except Exception:
        markets = []

    seen = 0
    for market in markets:
        if len(alerts) >= limit:
            break
        title = _extract_title(market)
        if classify_topic(title)["market_type"] != "retail_driven":
            continue
        trend_topic = _extract_trend_topic(title)
        if seen > 0:
            time.sleep(0.5)
        seen += 1
        td = _get_trend_data(trend_topic)
        score  = td["social_score"]       if td else 0.50
        hours  = td["hours_since_signal"] if td else None
        if td:
            data_src = td.get("source", "google_trends")
            source = (f"Wikipedia Pageviews · article: '{td.get('search_term', trend_topic)}'"
                      if data_src == "wikipedia_pageviews"
                      else f"Google Trends · search: '{trend_topic}'")
        else:
            source = f"Signal unavailable · search: '{trend_topic}'"
        confidence = "high" if score >= 0.75 else "medium" if score >= 0.60 else "low"
        window = {"high": "next 24-48 hours", "medium": "next 48-72 hours", "low": "next 72-96 hours"}[confidence]
        alerts.append({
            "market": title, "token_id": _extract_token_id(market),
            "social_score": score, "social_score_source": source,
            "hours_since_signal": hours, "expected_reprice_window": window,
            "current_probability": _extract_probability(market), "confidence": confidence,
        })

    alerts.sort(key=lambda a: a["social_score"], reverse=True)
    trends_live = any("unavailable" not in a.get("social_score_source", "") for a in alerts)
    note = (
        "Social scores are real 7-day Google Trends momentum. Markets sourced from Polymarket Gamma API."
        if (alerts and trends_live) else
        "Showing Polymarket retail-classified markets. Google Trends temporarily unavailable — scores shown as 0.50."
        if alerts else
        "No active retail-classified markets returned from Polymarket right now."
    )
    return {"alerts": alerts, "generated_at": _iso_now(),
            "data_sources": "Polymarket Gamma API + Google Trends (pytrends 7-day)", "note": note}

# ---------------------------------------------------------------------------
# Explain topic
# ---------------------------------------------------------------------------

def explain_topic(topic: str, market_id: str | None = None) -> dict[str, Any]:
    cl       = classify_topic(topic)
    lag_days = cl["expected_lag_days"]
    now      = _utc_now()
    tt       = _extract_trend_topic(topic)
    td       = _get_trend_data(tt)

    social_score = td["social_score"] if td else None
    hours        = td["hours_since_signal"] if td else None
    source       = f"Google Trends (7-day momentum) · search: '{tt}'" if td else "Google Trends unavailable"

    detected = cl["market_type"] == "retail_driven" and social_score is not None and social_score >= 0.52
    det_at, reprice_by = None, None
    if detected and hours:
        sig_dt    = now - timedelta(hours=hours)
        det_at    = sig_dt.date().isoformat()
        reprice_by = (sig_dt + timedelta(days=abs(lag_days))).date().isoformat()

    factors = (
        [f"Google Trends signal: {social_score:.2f} (7-day momentum)" if social_score else "Google Trends: no spike detected",
         "Retail-flooded markets show trend momentum leading market reprice",
         "Validated: GME 2021 — Google Trends led market by 3 days (corr=−0.604)"]
        if cl["market_type"] == "retail_driven" else
        [f"Google Trends signal: {social_score:.2f}" if social_score else "Google Trends: no elevated signal",
         "Institutional markets: price leads public search interest",
         "Validated: BTC 2024 — market led Google Trends by 7 days (corr=+0.707)"]
    )

    return {
        "topic": topic, "market_type": cl["market_type"],
        "current_social_score": social_score, "social_score_source": source,
        "current_market_probability": None,
        "lag_analysis": {"avg_lag_days": lag_days, "signal_detected": detected,
                         "signal_detected_at": det_at, "expected_reprice_by": reprice_by},
        "top_factors": factors,
        "classification_validation": cl.get("validation", "unknown"),
    }

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="The Accuracy Trap API",
    description="Prediction market retail-flood detector. Data source: " + _DATA_SOURCE,
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="ok", data_source=_DATA_SOURCE)


@app.get("/accuracy-trap", response_model=AccuracyTrapResponse)
def accuracy_trap() -> dict:
    """Accuracy Trap calibration curve — 4,714 resolved Manifold markets."""
    return _ACCURACY_TRAP_DATA


@app.get("/lag", response_model=LagResponse)
def lag(category: str = Query(..., description="Category: political, sports, crypto, economic, climate")) -> LagResponse:
    normalized = category.lower().strip()
    if normalized not in _CATEGORY_STATS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown category '{category}'. Valid: {', '.join(_CATEGORY_STATS)}",
        )
    stats      = _CATEGORY_STATS[normalized]
    retail_err = stats.get("retail_error")
    soph_err   = stats.get("sophisticated_error")
    multiplier = _r(retail_err / soph_err, 2) if retail_err and soph_err and soph_err > 0 else None
    return LagResponse(
        category=normalized, display_name=stats["display_name"], n=stats["n"],
        mean_calibration_error=stats["mean_calibration_error"], retail_pct=stats["retail_pct"],
        retail_error=retail_err, sophisticated_error=soph_err, error_multiplier=multiplier,
        data_source="Manifold Markets — 4,714 resolved binary markets",
    )


@app.get("/classify", response_model=ClassifyResponse)
def classify(topic: str = Query(..., min_length=1, description="Topic or market name")) -> ClassifyResponse:
    cleaned = topic.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="topic must not be blank")
    result = classify_topic(cleaned)
    return ClassifyResponse(
        topic=result["topic"], market_type=result["market_type"],
        expected_lag_days=result["expected_lag_days"],
        confidence=result["confidence"], reasoning=result["reasoning"],
    )


@app.get("/live-alerts", response_model=LiveAlertsResponse)
def live_alerts() -> LiveAlertsResponse:
    return LiveAlertsResponse(**get_live_alerts())


@app.get("/explain", response_model=ExplainResponse)
def explain(
    topic: str | None = Query(default=None, description="Topic or market name"),
    market_id: str | None = Query(default=None, description="Optional market identifier"),
) -> ExplainResponse:
    target = (topic or "").strip() or (market_id or "").strip()
    if not target:
        raise HTTPException(status_code=400, detail="Provide either topic or market_id")
    return ExplainResponse(**explain_topic(target, market_id=market_id))

# ---------------------------------------------------------------------------
# Markets dataset — 200 representative records (top/bottom/mid calibration error)
# ---------------------------------------------------------------------------
import json as _json
_MARKETS_DATA: list[dict] = _json.loads('[{"question": "Will the Tesla (TSLA) stock price close over $1000 on March 23, 2022?", "prob": 0.996, "resolution": "NO", "error": 0.996, "bettors": 5, "avg_bet": 583.2, "category": "Elon/Tesla", "market_type": "Sophisticated"}, {"question": "Will train traffic on the Dovre line be restored before July 2024?", "prob": 0.02, "resolution": "YES", "error": 0.98, "bettors": 5, "avg_bet": 127.1, "category": "AI/Tech", "market_type": "Small-bet"}, {"question": "Will the US successfully broker a ceasefire between Israel and Hamas by the end ", "prob": 0.03, "resolution": "YES", "error": 0.97, "bettors": 100, "avg_bet": 135.4, "category": "Geopolitics", "market_type": "Small-bet"}, {"question": "Will the Nasdaq Composite (IXIC) close higher on July 27th than it closed on July 26th?", "prob": 0.97, "resolution": "NO", "error": 0.97, "bettors": 16, "avg_bet": 180.9, "category": "Other", "market_type": "Small-bet"}, {"question": "Will the 2023 peak blossom of cherry trees in Kyoto happen earlier than ever? (peak before March 26th)", "prob": 0.04, "resolution": "YES", "error": 0.96, "bettors": 12, "avg_bet": 72.7, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will Richmond, Virginia, take concrete steps towards a Land Value Tax by 2023?", "prob": 0.04, "resolution": "YES", "error": 0.96, "bettors": 17, "avg_bet": 107.7, "category": "Geopolitics", "market_type": "Small-bet"}, {"question": "Will the Nasdaq Composite (IXIC) close higher on July 21st than it closed on July 20th?", "prob": 0.95, "resolution": "NO", "error": 0.95, "bettors": 18, "avg_bet": 216.5, "category": "Other", "market_type": "Large-bet"}, {"question": "Will the Dow Jones (DJI) close higher on August 4th than it closed on August 3rd?   [\\u1e40ana Leaderboard]", "prob": 0.95, "resolution": "NO", "error": 0.95, "bettors": 22, "avg_bet": 91.2, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will @firstuserhere not earn a trustworthy-ish badge by end of 2023?", "prob": 0.95, "resolution": "NO", "error": 0.95, "bettors": 39, "avg_bet": 307.1, "category": "Other", "market_type": "Large-bet"}, {"question": "Will an AI model be used for operational weather forecasts by the end of 2024?", "prob": 0.941, "resolution": "NO", "error": 0.9409, "bettors": 23, "avg_bet": 173.4, "category": "AI/Tech", "market_type": "Small-bet"}, {"question": "Will Crude Oil Jan 24 close higher on Dec 7 than Dec 6? (CL=F Daily)", "prob": 0.94, "resolution": "NO", "error": 0.94, "bettors": 3, "avg_bet": 134.8, "category": "AI/Tech", "market_type": "Small-bet"}, {"question": "The NYTimes will still use \\"Twitter\\" when referring to \\"X\\" in 2024.", "prob": 0.92, "resolution": "NO", "error": 0.92, "bettors": 33, "avg_bet": 150.6, "category": "Elon/Tesla", "market_type": "Small-bet"}, {"question": "Will User alexlyzhov Solicitation of $10 Crypto Result in Accusations of Fraud o", "prob": 0.081, "resolution": "YES", "error": 0.9192, "bettors": 7, "avg_bet": 11.6, "category": "Crypto", "market_type": "Retail Flood"}, {"question": "$NVDA up after earnings?", "prob": 0.913, "resolution": "NO", "error": 0.9133, "bettors": 29, "avg_bet": 210.3, "category": "Other", "market_type": "Large-bet"}, {"question": "will the temperature in West Hollywood exceed 75 degrees farenheit today", "prob": 0.912, "resolution": "NO", "error": 0.9117, "bettors": 3, "avg_bet": 106.7, "category": "Other", "market_type": "Small-bet"}, {"question": "Will there be smoke at the start of Manifest?", "prob": 0.91, "resolution": "NO", "error": 0.91, "bettors": 25, "avg_bet": 61.3, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will BnB reach 1400 this year", "prob": 0.093, "resolution": "YES", "error": 0.9068, "bettors": 2, "avg_bet": 106.0, "category": "Other", "market_type": "Small-bet"}, {"question": "Will a DeSci (decentralized science) project be valued at $100 million or more a", "prob": 0.1, "resolution": "YES", "error": 0.9, "bettors": 5, "avg_bet": 102.4, "category": "Other", "market_type": "Small-bet"}, {"question": "GPT4-Resolving:   Was Tesla the most active stock on NASDAQ during regular trading hours?", "prob": 0.1, "resolution": "YES", "error": 0.9, "bettors": 3, "avg_bet": 41.0, "category": "AI/Tech", "market_type": "Retail Flood"}, {"question": "Will the Dow Jones (DJI) close higher on August 14th than it closed on August 11th?   [\\u1e40ana Leaderboard]", "prob": 0.1, "resolution": "YES", "error": 0.9, "bettors": 19, "avg_bet": 106.4, "category": "Other", "market_type": "Small-bet"}, {"question": "Will the resolution of \\"Does trump have any clue what he\\u2019s doing with his trade ", "prob": 0.894, "resolution": "NO", "error": 0.8939, "bettors": 14, "avg_bet": 25.5, "category": "Politics", "market_type": "Retail Flood"}, {"question": "10) Many billions of dollars of new investment commitments will be announced to build chip manufacturing facilities in the United States as the U.S. makes contingency plans for Taiwan.", "prob": 0.89, "resolution": "NO", "error": 0.89, "bettors": 15, "avg_bet": 41.1, "category": "AI/Tech", "market_type": "Retail Flood"}, {"question": "Will Emerson get a spacex internship this cycle?", "prob": 0.887, "resolution": "NO", "error": 0.8869, "bettors": 5, "avg_bet": 120.2, "category": "Elon/Tesla", "market_type": "Small-bet"}, {"question": "Will Japan beat Italy in the women\\u2019s ice hockey match at Milano-Cortina 2026 on ", "prob": 0.887, "resolution": "NO", "error": 0.8866, "bettors": 3, "avg_bet": 76.7, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will Trump reduce the trade deficit in his first year?", "prob": 0.115, "resolution": "YES", "error": 0.8852, "bettors": 96, "avg_bet": 276.8, "category": "Politics", "market_type": "Large-bet"}, {"question": "Will the Dow Jones (DJI) close higher on Wed. December 20th than on Tue. December 19th? {DAILY}", "prob": 0.88, "resolution": "NO", "error": 0.88, "bettors": 16, "avg_bet": 65.4, "category": "AI/Tech", "market_type": "Retail Flood"}, {"question": "Will crypto prices crash post FOMC meeting?", "prob": 0.128, "resolution": "YES", "error": 0.8722, "bettors": 9, "avg_bet": 51.7, "category": "Crypto", "market_type": "Retail Flood"}, {"question": "Will the Dow Jones (DJI) close higher on Fri. December 8th than on Fri. December 1st? {WEEKLY}", "prob": 0.13, "resolution": "YES", "error": 0.87, "bettors": 44, "avg_bet": 151.0, "category": "Other", "market_type": "Small-bet"}, {"question": "Will the average global temperature in 2025 exceed 2023?", "prob": 0.868, "resolution": "NO", "error": 0.8684, "bettors": 61, "avg_bet": 547.0, "category": "Other", "market_type": "Sophisticated"}, {"question": "Will the United States move up in the 2026 World Happiness Report", "prob": 0.134, "resolution": "YES", "error": 0.8659, "bettors": 16, "avg_bet": 18.0, "category": "Other", "market_type": "Retail Flood"}, {"question": "Do Manifolders think that Manifold is too focused on AI and niche tech topics?", "prob": 0.86, "resolution": "NO", "error": 0.86, "bettors": 50, "avg_bet": 39.5, "category": "AI/Tech", "market_type": "Retail Flood"}, {"question": "Kkr vs pbks whois winyes kkr no pbks", "prob": 0.145, "resolution": "YES", "error": 0.8554, "bettors": 5, "avg_bet": 67.6, "category": "Other", "market_type": "Retail Flood"}, {"question": "test market", "prob": 0.85, "resolution": "NO", "error": 0.85, "bettors": 1, "avg_bet": 6.2, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will \\"Will a device from Neuralink receive FDA approval in 2023 for implantation in a human?\\" resolve N/A?", "prob": 0.85, "resolution": "NO", "error": 0.85, "bettors": 1, "avg_bet": 70.0, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will the Indian economy exceed $4 trillion by April 1st, 2025? (End of financial", "prob": 0.152, "resolution": "YES", "error": 0.8482, "bettors": 6, "avg_bet": 31.2, "category": "Economics", "market_type": "Retail Flood"}, {"question": "Liverpool beats AC Milan in Hong Kong on 26 July 2025?", "prob": 0.842, "resolution": "NO", "error": 0.8423, "bettors": 17, "avg_bet": 97.0, "category": "Other", "market_type": "Small-bet"}, {"question": "Will the Israel - Hamas war spread to another location in 2025?", "prob": 0.16, "resolution": "YES", "error": 0.84, "bettors": 10, "avg_bet": 39.0, "category": "Geopolitics", "market_type": "Retail Flood"}, {"question": "Will Super Bowl LX set a new all-time record for US television viewership?", "prob": 0.84, "resolution": "NO", "error": 0.84, "bettors": 52, "avg_bet": 96.8, "category": "Other", "market_type": "Small-bet"}, {"question": "Will the May 2025 US Unemployment rate surpass the projected 4.2% (as reported on June 6, 2025 08:30 EST)?", "prob": 0.837, "resolution": "NO", "error": 0.8371, "bettors": 5, "avg_bet": 27.4, "category": "Economics", "market_type": "Retail Flood"}, {"question": "Will Simone Biles win the gold medal in the Floor Exercise at the 2024 Paris Olympics? \\ud83e\\udd47", "prob": 0.83, "resolution": "NO", "error": 0.83, "bettors": 60, "avg_bet": 125.6, "category": "Other", "market_type": "Small-bet"}, {"question": "Will the Dow Jones Industrial Average (DJI) close higher on June 26th than it closed on June 23rd?", "prob": 0.83, "resolution": "NO", "error": 0.83, "bettors": 16, "avg_bet": 122.1, "category": "Other", "market_type": "Small-bet"}, {"question": "Community member in ZC discord creates a Manifold prediction market?", "prob": 0.819, "resolution": "NO", "error": 0.8193, "bettors": 7, "avg_bet": 50.1, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will Weaver get into a T20?", "prob": 0.819, "resolution": "NO", "error": 0.819, "bettors": 10, "avg_bet": 95.4, "category": "Other", "market_type": "Small-bet"}, {"question": "Will X \\u2014 Twitter lose at least 75 million in ad revenue by the end of the year?", "prob": 0.19, "resolution": "YES", "error": 0.81, "bettors": 14, "avg_bet": 78.3, "category": "Elon/Tesla", "market_type": "Retail Flood"}, {"question": "Will Spendthrift\'s four first-crop sires finish as the top four on the TDN N.A. sire list by earnings in 2023?", "prob": 0.19, "resolution": "YES", "error": 0.81, "bettors": 16, "avg_bet": 20.0, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will the January unemployment rate be 3.8% or above?", "prob": 0.81, "resolution": "NO", "error": 0.81, "bettors": 52, "avg_bet": 56.6, "category": "Economics", "market_type": "Retail Flood"}, {"question": "Will I achieve my maximum Calorie goal over the next week?", "prob": 0.81, "resolution": "NO", "error": 0.81, "bettors": 4, "avg_bet": 19.5, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will Manifold make any changes to liquidity application on Multiple Choice or Se", "prob": 0.191, "resolution": "YES", "error": 0.8089, "bettors": 7, "avg_bet": 195.8, "category": "Other", "market_type": "Small-bet"}, {"question": "Will Kalshi offer sports markets in Nevada on March 16th?", "prob": 0.194, "resolution": "YES", "error": 0.8061, "bettors": 9, "avg_bet": 241.8, "category": "Other", "market_type": "Large-bet"}, {"question": "Will Hurricane Helene shut down Georgia Schools for the Weekend and Monday onwards?", "prob": 0.8, "resolution": "NO", "error": 0.8, "bettors": 4, "avg_bet": 28.8, "category": "Geopolitics", "market_type": "Retail Flood"}, {"question": "Will the S&P 500 open at its ATH today?", "prob": 0.799, "resolution": "NO", "error": 0.7989, "bettors": 10, "avg_bet": 346.5, "category": "Other", "market_type": "Large-bet"}, {"question": "Will Netflix\\u2019s stock price on 19 October 2025 be higher than on 17 October 2025?", "prob": 0.202, "resolution": "YES", "error": 0.798, "bettors": 30, "avg_bet": 86.1, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will Apple (NASDAQ: AAPL) close higher at the end of April than March?", "prob": 0.79, "resolution": "NO", "error": 0.79, "bettors": 36, "avg_bet": 85.1, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will a major tech company release a generative video editor by the end of 2024?", "prob": 0.212, "resolution": "YES", "error": 0.7884, "bettors": 22, "avg_bet": 115.9, "category": "Other", "market_type": "Small-bet"}, {"question": "Will the German economy enter a recession during 2023?", "prob": 0.22, "resolution": "YES", "error": 0.78, "bettors": 23, "avg_bet": 66.4, "category": "Economics", "market_type": "Retail Flood"}, {"question": "Will we ever have 8+ people at the San Mateo science fiction reading group, by e", "prob": 0.22, "resolution": "YES", "error": 0.78, "bettors": 9, "avg_bet": 42.7, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will the April jobs report be above 200,000?", "prob": 0.23, "resolution": "YES", "error": 0.77, "bettors": 14, "avg_bet": 17.6, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will I win a bet with my friend?", "prob": 0.762, "resolution": "NO", "error": 0.7616, "bettors": 3, "avg_bet": 37.0, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will Trump get audibly booed at the Super Bowl?", "prob": 0.24, "resolution": "YES", "error": 0.76, "bettors": 31, "avg_bet": 31.5, "category": "Politics", "market_type": "Retail Flood"}, {"question": "Will the Dow Jones (DJI) close higher on October 11th than on October 10th? [\\u1e40ana Leaderboard]", "prob": 0.24, "resolution": "YES", "error": 0.76, "bettors": 13, "avg_bet": 104.8, "category": "Other", "market_type": "Small-bet"}, {"question": "Will Claude AI experience no partial or major outage in the next 7 days, accordi", "prob": 0.749, "resolution": "NO", "error": 0.7489, "bettors": 19, "avg_bet": 15.3, "category": "AI/Tech", "market_type": "Retail Flood"}, {"question": "Will the Dow Jones (DJI) close higher on August 3rd than it closed on August 2nd?   [\\u1e40ana Leaderboard]", "prob": 0.74, "resolution": "NO", "error": 0.74, "bettors": 21, "avg_bet": 73.0, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will Zvi Mowshowitz earn ElonBucks by 3/1/25?", "prob": 0.272, "resolution": "YES", "error": 0.7279, "bettors": 21, "avg_bet": 116.0, "category": "Elon/Tesla", "market_type": "Small-bet"}, {"question": "Will direct arbitrage between manifold markets and the US stock market be possible by July 2024  (Subsidized 250M)", "prob": 0.719, "resolution": "NO", "error": 0.719, "bettors": 20, "avg_bet": 77.3, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will the AP Compsci midterm be postponed again?", "prob": 0.356, "resolution": "YES", "error": 0.6438, "bettors": 11, "avg_bet": 20.3, "category": "AI/Tech", "market_type": "Retail Flood"}, {"question": "Will there be an El Nino weather event in Australia in 2023?", "prob": 0.36, "resolution": "YES", "error": 0.64, "bettors": 8, "avg_bet": 36.3, "category": "Climate", "market_type": "Retail Flood"}, {"question": "Will AntiHunter treasury exceed $500K FMV by Feb 19, 2026?", "prob": 0.594, "resolution": "NO", "error": 0.5944, "bettors": 2, "avg_bet": 15.0, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will Elon Musk be out of Trump\'s administration before any of Trump\'s cabinet?", "prob": 0.413, "resolution": "YES", "error": 0.5875, "bettors": 19, "avg_bet": 35.9, "category": "Politics", "market_type": "Retail Flood"}, {"question": "Will I write an essay about my crush for mcsp 2026?", "prob": 0.562, "resolution": "NO", "error": 0.5623, "bettors": 8, "avg_bet": 50.2, "category": "Other", "market_type": "Retail Flood"}, {"question": "GPT4-Resolving:   Was Tesla the most active stock on NASDAQ by Dollar Volume during regular trading hours?", "prob": 0.56, "resolution": "NO", "error": 0.56, "bettors": 1, "avg_bet": 13.0, "category": "AI/Tech", "market_type": "Retail Flood"}, {"question": "Will the S&P 500 open at its ATH today?", "prob": 0.514, "resolution": "NO", "error": 0.5143, "bettors": 5, "avg_bet": 35.8, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will I eat a sandwhich at lunch by tommorow", "prob": 0.487, "resolution": "YES", "error": 0.513, "bettors": 5, "avg_bet": 10.6, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will the temperature in Central Park September 11th at 3:51pm be in the 79-81\\u00b0 range?", "prob": 0.51, "resolution": "NO", "error": 0.51, "bettors": 20, "avg_bet": 35.2, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will Tesla stock be lower on August 10, 2025, than on August 7, 2025?", "prob": 0.501, "resolution": "NO", "error": 0.5009, "bettors": 48, "avg_bet": 156.4, "category": "Elon/Tesla", "market_type": "Small-bet"}, {"question": "Will Sky Sports Halo be discontinued?", "prob": 0.5, "resolution": "YES", "error": 0.4995, "bettors": 2, "avg_bet": 10.5, "category": "Other", "market_type": "Retail Flood"}, {"question": "Daily Coinflip", "prob": 0.501, "resolution": "YES", "error": 0.4988, "bettors": 34, "avg_bet": 259.6, "category": "AI/Tech", "market_type": "Large-bet"}, {"question": "Daily Coin Flip - Day 614", "prob": 0.505, "resolution": "YES", "error": 0.495, "bettors": 13, "avg_bet": 33.7, "category": "AI/Tech", "market_type": "Retail Flood"}, {"question": "Will Meta Platforms\\u2019 stock price on 7 October 2025 be lower than on 26 September", "prob": 0.508, "resolution": "YES", "error": 0.4922, "bettors": 31, "avg_bet": 137.9, "category": "Other", "market_type": "Small-bet"}, {"question": "Will anyone on LessWrong say they\'ve moved funds from a crypto exchange because ", "prob": 0.49, "resolution": "NO", "error": 0.49, "bettors": 6, "avg_bet": 22.5, "category": "Crypto", "market_type": "Retail Flood"}, {"question": "Will an AI Agent contribute to my codebase?", "prob": 0.518, "resolution": "YES", "error": 0.482, "bettors": 8, "avg_bet": 44.5, "category": "AI/Tech", "market_type": "Retail Flood"}, {"question": "Will the US Treasury Series I Bond fixed rate go up [or remain unchanged]?", "prob": 0.56, "resolution": "YES", "error": 0.44, "bettors": 9, "avg_bet": 28.0, "category": "AI/Tech", "market_type": "Retail Flood"}, {"question": "Will we wind up having to rip up the floor and/or ceiling?", "prob": 0.422, "resolution": "NO", "error": 0.4224, "bettors": 2, "avg_bet": 15.0, "category": "Other", "market_type": "Retail Flood"}, {"question": "will \\u2018I Like It\\u2019 be sung during the Super Bowl half time show?", "prob": 0.404, "resolution": "NO", "error": 0.4041, "bettors": 2, "avg_bet": 67.0, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will a new species, previously unknown to science, be discovered in Lake Vostok ", "prob": 0.4, "resolution": "NO", "error": 0.4, "bettors": 10, "avg_bet": 20.8, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will Joe Biden drop out of the US Election 2024 due to health concerns?", "prob": 0.37, "resolution": "NO", "error": 0.37, "bettors": 33, "avg_bet": 174.1, "category": "Politics", "market_type": "Small-bet"}, {"question": "Will I still believe that 30% of the American Economy was firewood in 1830 after", "prob": 0.364, "resolution": "NO", "error": 0.364, "bettors": 9, "avg_bet": 20.1, "category": "Economics", "market_type": "Retail Flood"}, {"question": "Will Taybor Pepper be selected for the 2023 NFL Pro Bowl games?", "prob": 0.36, "resolution": "NO", "error": 0.36, "bettors": 58, "avg_bet": 241.2, "category": "Sports", "market_type": "Large-bet"}, {"question": "Will Portland, Oregon have a daily high temperature of 104F (40C) or greater in 2024? \\ud83c\\udf21", "prob": 0.64, "resolution": "YES", "error": 0.36, "bettors": 30, "avg_bet": 41.2, "category": "AI/Tech", "market_type": "Retail Flood"}, {"question": "Will the Nasdaq Composite (IXIC) close higher on July 26th than it closed on July 25th?", "prob": 0.36, "resolution": "NO", "error": 0.36, "bettors": 20, "avg_bet": 86.9, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will US based crypto exchanges stop doing business with Russia?", "prob": 0.65, "resolution": "YES", "error": 0.3502, "bettors": 19, "avg_bet": 48.6, "category": "Crypto", "market_type": "Retail Flood"}, {"question": "Will there be a major doping disqualification at the 2026 Winter Olympics?", "prob": 0.34, "resolution": "NO", "error": 0.34, "bettors": 39, "avg_bet": 41.6, "category": "Other", "market_type": "Retail Flood"}, {"question": "Hamas accepts Trump\'s proposed peace deal?", "prob": 0.328, "resolution": "NO", "error": 0.3281, "bettors": 544, "avg_bet": 410.1, "category": "Politics", "market_type": "Large-bet"}, {"question": "Rosal\\u00eda will perform with Bad Bunny at the Super Bowl", "prob": 0.321, "resolution": "NO", "error": 0.3212, "bettors": 4, "avg_bet": 45.8, "category": "Other", "market_type": "Retail Flood"}, {"question": "By the next UK election, will a sitting MP publicly question US military presenc", "prob": 0.68, "resolution": "YES", "error": 0.3205, "bettors": 7, "avg_bet": 346.5, "category": "Politics", "market_type": "Large-bet"}, {"question": "Will I win my sport bet made by dice?", "prob": 0.688, "resolution": "YES", "error": 0.3121, "bettors": 7, "avg_bet": 46.0, "category": "Other", "market_type": "Retail Flood"}, {"question": "NYT/MSFT v. OpenAI: Current copyright law obsolete and non enforceable. Will hav", "prob": 0.311, "resolution": "NO", "error": 0.3109, "bettors": 5, "avg_bet": 21.1, "category": "AI/Tech", "market_type": "Retail Flood"}, {"question": "Will I leave the house before 11 a.m. every day in April?", "prob": 0.31, "resolution": "NO", "error": 0.31, "bettors": 8, "avg_bet": 21.7, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will NOAA publish a CEI for 2023 of 30+%?", "prob": 0.69, "resolution": "YES", "error": 0.31, "bettors": 8, "avg_bet": 11.8, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will palladium prices increase by more than 10% by February 2026 due to supply deficits?", "prob": 0.711, "resolution": "YES", "error": 0.2891, "bettors": 1, "avg_bet": 20.0, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will the Texas Legislature commission a study on split-rate property tax before 2025?", "prob": 0.286, "resolution": "NO", "error": 0.2859, "bettors": 14, "avg_bet": 63.1, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will the La Rioja currency the \\"chacho\\" default in December?", "prob": 0.28, "resolution": "NO", "error": 0.2799, "bettors": 2, "avg_bet": 350.0, "category": "Other", "market_type": "Large-bet"}, {"question": "Will I get into Oxford?", "prob": 0.272, "resolution": "NO", "error": 0.2719, "bettors": 20, "avg_bet": 89.0, "category": "Other", "market_type": "Retail Flood"}, {"question": "Temperature in New York City on Christmas Day plus 2*die roll > 46 Farenheit", "prob": 0.734, "resolution": "YES", "error": 0.2656, "bettors": 13, "avg_bet": 34.5, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will I have at least 10% of a watermelon in my house at market close?", "prob": 0.26, "resolution": "NO", "error": 0.26, "bettors": 9, "avg_bet": 16.2, "category": "Elon/Tesla", "market_type": "Retail Flood"}, {"question": "Will Ukraine establish the base along the left bank of the Dnipro River before December 2023?", "prob": 0.74, "resolution": "YES", "error": 0.26, "bettors": 61, "avg_bet": 205.7, "category": "AI/Tech", "market_type": "Large-bet"}, {"question": "QUESTION Will the temperature in Chelyabinsk be above -10\\u00b0C at 15:00 UTC on February 12, 2026?", "prob": 0.253, "resolution": "NO", "error": 0.2534, "bettors": 10, "avg_bet": 44.2, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will the Dow Jones (DJI) close higher on Mon. November 27th than on Fri. November 24th? {DAILY}", "prob": 0.25, "resolution": "NO", "error": 0.25, "bettors": 19, "avg_bet": 104.9, "category": "AI/Tech", "market_type": "Small-bet"}, {"question": "Will the Dow Jones (DJI) close higher on Fri. December 8th than on Thu. December 7th? {DAILY}", "prob": 0.76, "resolution": "YES", "error": 0.24, "bettors": 15, "avg_bet": 37.9, "category": "AI/Tech", "market_type": "Retail Flood"}, {"question": "Will big floods engulf London and Europe in 2024?", "prob": 0.238, "resolution": "NO", "error": 0.2377, "bettors": 8, "avg_bet": 19.3, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will the Miami Hurricanes be the winner of the CFP football?", "prob": 0.22, "resolution": "NO", "error": 0.22, "bettors": 18, "avg_bet": 41.4, "category": "Sports", "market_type": "Retail Flood"}, {"question": "Red Scare podcast hosts interview a major political figure before September 1, 2", "prob": 0.21, "resolution": "NO", "error": 0.2104, "bettors": 11, "avg_bet": 38.3, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will GPT-5 earn Bronze on IMO 2025?", "prob": 0.201, "resolution": "NO", "error": 0.2014, "bettors": 15, "avg_bet": 45.4, "category": "AI/Tech", "market_type": "Retail Flood"}, {"question": "Google releases an AI world model game by June 1st 2026?", "prob": 0.8, "resolution": "YES", "error": 0.2, "bettors": 12, "avg_bet": 124.6, "category": "AI/Tech", "market_type": "Small-bet"}, {"question": "Chet Holmgren shot blocked in game 6 of NBA finals?", "prob": 0.8, "resolution": "YES", "error": 0.2, "bettors": 6, "avg_bet": 65.2, "category": "Sports", "market_type": "Retail Flood"}, {"question": "Will the current Israeli coalition survive until January 2023?", "prob": 0.194, "resolution": "NO", "error": 0.1944, "bettors": 7, "avg_bet": 54.9, "category": "Geopolitics", "market_type": "Retail Flood"}, {"question": "Is the Letitia James mortgage fraud story legit?", "prob": 0.191, "resolution": "NO", "error": 0.1914, "bettors": 10, "avg_bet": 271.4, "category": "Other", "market_type": "Large-bet"}, {"question": "Will Destiny be banned or further limited on Twitter / X by the end of 2024?", "prob": 0.181, "resolution": "NO", "error": 0.1813, "bettors": 17, "avg_bet": 16.9, "category": "Elon/Tesla", "market_type": "Retail Flood"}, {"question": "Will the Dow Jones (DJI) close higher on Wed. November 29th than on Tue. November 28th? {DAILY}", "prob": 0.82, "resolution": "YES", "error": 0.18, "bettors": 14, "avg_bet": 45.0, "category": "AI/Tech", "market_type": "Retail Flood"}, {"question": "Will Claude Resolve this Market YES? (Hard Mode)", "prob": 0.171, "resolution": "NO", "error": 0.1707, "bettors": 109, "avg_bet": 291.0, "category": "AI/Tech", "market_type": "Large-bet"}, {"question": "Will China Move up in the World Happiness Report", "prob": 0.833, "resolution": "YES", "error": 0.1669, "bettors": 9, "avg_bet": 27.0, "category": "Other", "market_type": "Retail Flood"}, {"question": "Temperature in New York City on Christmas Day > 39 Farenheit?", "prob": 0.835, "resolution": "YES", "error": 0.1651, "bettors": 14, "avg_bet": 24.4, "category": "Other", "market_type": "Retail Flood"}, {"question": "Due to instability in the middle east, will the price of crude oil surpass 90 dollars per barrel by December 1st?", "prob": 0.84, "resolution": "YES", "error": 0.16, "bettors": 25, "avg_bet": 147.3, "category": "Other", "market_type": "Small-bet"}, {"question": "Will \\"Propaganda or Science: A Look at Open Source ...\\" make the top fifty posts", "prob": 0.158, "resolution": "NO", "error": 0.1577, "bettors": 1, "avg_bet": 40.2, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will America send troops to Iran within this week?", "prob": 0.15, "resolution": "NO", "error": 0.15, "bettors": 34, "avg_bet": 35.0, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will Congressional stock trades outperform the S&P 500 by more than 10 percentage points in 2023?", "prob": 0.15, "resolution": "NO", "error": 0.15, "bettors": 125, "avg_bet": 81.0, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will there be internet shutdown in Uganda post elections?", "prob": 0.853, "resolution": "YES", "error": 0.147, "bettors": 10, "avg_bet": 39.7, "category": "Politics", "market_type": "Retail Flood"}, {"question": "Will Binance be the largest crypto exchange market at the end of 2024?", "prob": 0.856, "resolution": "YES", "error": 0.1445, "bettors": 13, "avg_bet": 37.0, "category": "Crypto", "market_type": "Retail Flood"}, {"question": "Will the Dow Jones Industrial Average (DJI) close higher on July 3rd than it closed on June 30th?", "prob": 0.86, "resolution": "YES", "error": 0.14, "bettors": 19, "avg_bet": 180.5, "category": "Other", "market_type": "Small-bet"}, {"question": "Will the NY Yankees win the World Series in 2024?", "prob": 0.13, "resolution": "NO", "error": 0.1297, "bettors": 22, "avg_bet": 87.7, "category": "Other", "market_type": "Retail Flood"}, {"question": "Tesla closes above 500.00 in 2025?", "prob": 0.118, "resolution": "NO", "error": 0.1183, "bettors": 6, "avg_bet": 67.2, "category": "Elon/Tesla", "market_type": "Retail Flood"}, {"question": "Will the Nasdaq Composite (IXIC) close higher on October 11th than it closed on October 10th?  [\\u1e40ana Leaderboard]", "prob": 0.89, "resolution": "YES", "error": 0.11, "bettors": 18, "avg_bet": 157.0, "category": "Other", "market_type": "Small-bet"}, {"question": "Will a regulator take action against a Kalshi sports market in 2025?", "prob": 0.899, "resolution": "YES", "error": 0.1014, "bettors": 7, "avg_bet": 304.7, "category": "AI/Tech", "market_type": "Large-bet"}, {"question": "Reserve Bank of Australia cuts rates by 50+ basis points at next meeting?", "prob": 0.101, "resolution": "NO", "error": 0.101, "bettors": 10, "avg_bet": 251.1, "category": "Other", "market_type": "Large-bet"}, {"question": "Will the CDA be part of the next governing coalition of the Netherlands?", "prob": 0.1, "resolution": "NO", "error": 0.1, "bettors": 24, "avg_bet": 83.2, "category": "Crypto", "market_type": "Retail Flood"}, {"question": "Will the Kenya Universal Basic Income experiment find that UBI significantly decreases earnings?", "prob": 0.1, "resolution": "NO", "error": 0.1, "bettors": 50, "avg_bet": 42.1, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will a software engineer get a job through the tech hiring startup Otherbranch.c", "prob": 0.1, "resolution": "NO", "error": 0.1, "bettors": 18, "avg_bet": 920.5, "category": "Geopolitics", "market_type": "Sophisticated"}, {"question": "Will Coin Center sue the U.S. Treasury (OFAC) over Tornado Cash sanctions?", "prob": 0.9, "resolution": "YES", "error": 0.1, "bettors": 10, "avg_bet": 69.7, "category": "Other", "market_type": "Retail Flood"}, {"question": "Elon Musk charged with Wisconsin state law violation in 2025?", "prob": 0.091, "resolution": "NO", "error": 0.091, "bettors": 21, "avg_bet": 29.7, "category": "Elon/Tesla", "market_type": "Retail Flood"}, {"question": "2026 Winter Olympics: Will Mikaela Shiffrin (USA) Win Gold in Women\\u2019s Slalom?", "prob": 0.912, "resolution": "YES", "error": 0.0883, "bettors": 14, "avg_bet": 47.0, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will Neuralink implant its device in five or more patients before the end of 2024?", "prob": 0.083, "resolution": "NO", "error": 0.0829, "bettors": 14, "avg_bet": 202.8, "category": "Other", "market_type": "Small-bet"}, {"question": "Will University High School beat Troy High School in Science Olympiad at Socal S", "prob": 0.08, "resolution": "NO", "error": 0.08, "bettors": 7, "avg_bet": 56.1, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will at least 3 US banks be taken over by the Federal Government before April 1st?", "prob": 0.08, "resolution": "NO", "error": 0.08, "bettors": 38, "avg_bet": 76.1, "category": "Economics", "market_type": "Retail Flood"}, {"question": "Will Jason Kelce be sitting in a box with Taylor Swift at the 2023 Super Bowl?", "prob": 0.92, "resolution": "YES", "error": 0.08, "bettors": 41, "avg_bet": 159.5, "category": "Other", "market_type": "Small-bet"}, {"question": "Will market B get more traders than market A?", "prob": 0.922, "resolution": "YES", "error": 0.0782, "bettors": 19, "avg_bet": 39.9, "category": "Other", "market_type": "Retail Flood"}, {"question": "In an apocalyptic world, would you kill a zombie baby?", "prob": 0.924, "resolution": "YES", "error": 0.0758, "bettors": 9, "avg_bet": 41.2, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will I get a cast recording of the Rocket Boys musical by the end of 2025?", "prob": 0.075, "resolution": "NO", "error": 0.0752, "bettors": 3, "avg_bet": 73.8, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will Argentina\'s economy be broadly considered to be in better shape November 19", "prob": 0.926, "resolution": "YES", "error": 0.0745, "bettors": 37, "avg_bet": 78.5, "category": "Economics", "market_type": "Retail Flood"}, {"question": "Will Tenobrus\' account be private for the rest of August?", "prob": 0.074, "resolution": "NO", "error": 0.0745, "bettors": 3, "avg_bet": 96.7, "category": "Other", "market_type": "Small-bet"}, {"question": "Will Elon Musk sell more than $1Bln of Tesla stock during Q1 2023?", "prob": 0.07, "resolution": "NO", "error": 0.07, "bettors": 54, "avg_bet": 113.3, "category": "Elon/Tesla", "market_type": "Small-bet"}, {"question": "Will Elon Musk step down from his current roles in X (Twitter) this year", "prob": 0.07, "resolution": "NO", "error": 0.07, "bettors": 13, "avg_bet": 62.5, "category": "Elon/Tesla", "market_type": "Retail Flood"}, {"question": "Will the VVD exclude forming a government with PVV for the 2025 elections?", "prob": 0.937, "resolution": "YES", "error": 0.0627, "bettors": 7, "avg_bet": 112.6, "category": "Politics", "market_type": "Small-bet"}, {"question": "Any of these 30 LLM-generated disasters happens in the USA in 2024", "prob": 0.062, "resolution": "NO", "error": 0.0618, "bettors": 4, "avg_bet": 67.5, "category": "AI/Tech", "market_type": "Retail Flood"}, {"question": "Will Elon Musk suspend or resign from his Department of Government Efficiency role by October 15?", "prob": 0.939, "resolution": "YES", "error": 0.0612, "bettors": 16, "avg_bet": 43.6, "category": "Elon/Tesla", "market_type": "Retail Flood"}, {"question": "Will another person self-immolate in the United States for a political purpose b", "prob": 0.94, "resolution": "YES", "error": 0.06, "bettors": 96, "avg_bet": 258.4, "category": "Other", "market_type": "Large-bet"}, {"question": "Before 2026, will a neuroimaging study provide explicit support to symmetry theo", "prob": 0.06, "resolution": "NO", "error": 0.06, "bettors": 7, "avg_bet": 114.0, "category": "Other", "market_type": "Small-bet"}, {"question": "Will 10Y US Treasury Yields make new lows of the year before 31 Dec 2023?", "prob": 0.06, "resolution": "NO", "error": 0.06, "bettors": 12, "avg_bet": 80.5, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will an AI agent run a profitable (>10% ROI) prediction market portfolio by end ", "prob": 0.94, "resolution": "YES", "error": 0.06, "bettors": 14, "avg_bet": 54.8, "category": "AI/Tech", "market_type": "Retail Flood"}, {"question": "ECB cuts interest rates at September 11, 2025 meeting", "prob": 0.06, "resolution": "NO", "error": 0.06, "bettors": 10, "avg_bet": 130.1, "category": "Other", "market_type": "Small-bet"}, {"question": "Will the ROAD to Housing Act become law in 2025?", "prob": 0.053, "resolution": "NO", "error": 0.053, "bettors": 10, "avg_bet": 390.6, "category": "Other", "market_type": "Large-bet"}, {"question": "Will any commercial carbon\\u2013cement supercapacitors as a scalable bulk energy storage solution become operational by 2025?", "prob": 0.052, "resolution": "NO", "error": 0.0516, "bettors": 17, "avg_bet": 38.2, "category": "Other", "market_type": "Retail Flood"}, {"question": "Will the Washington Capitals beat Carolina Hurricanes on Jan 5? (Live Action Spo", "prob": 0.0, "resolution": "NO", "error": 0.0, "bettors": 18, "avg_bet": 733.4, "category": "Climate", "market_type": "Sophisticated"}, {"question": "Sport 2023: Will Philadelphia Eagles win Super Bowl?", "prob": 0.0, "resolution": "NO", "error": 0.0, "bettors": 25, "avg_bet": 668.7, "category": "Other", "market_type": "Sophisticated"}, {"question": "Will the WHO declare a new global health emergency due to an emerging infectious", "prob": 0.0, "resolution": "NO", "error": 0.0, "bettors": 21, "avg_bet": 445.2, "category": "Other", "market_type": "Large-bet"}, {"question": "\\ud83c\\udfc8 2023 NCAAF: Will Georgia Tech beat Virginia?", "prob": 1.0, "resolution": "YES", "error": 0.0, "bettors": 18, "avg_bet": 435.7, "category": "Other", "market_type": "Large-bet"}, {"question": "Will Robert F. Kennedy Jr. Achieve Ballot Access in All 50 States for the 2024 U", "prob": 0.0, "resolution": "NO", "error": 0.0, "bettors": 124, "avg_bet": 690.9, "category": "Other", "market_type": "Sophisticated"}, {"question": "\\ud83c\\udfc8 2023 NCAAF: Will Texas Tech beat #7 Texas?", "prob": 0.0, "resolution": "NO", "error": 0.0, "bettors": 16, "avg_bet": 860.7, "category": "Other", "market_type": "Sophisticated"}, {"question": "Will Trump ever earn more than 46% of voters in the Economist\'s polls for Februa", "prob": 1.0, "resolution": "YES", "error": 0.0, "bettors": 6, "avg_bet": 510.4, "category": "Politics", "market_type": "Sophisticated"}, {"question": "\\ud83c\\udfc0 Will #6 Texas Tech beat #11 NC State in the South region? (Men\'s March Madness", "prob": 0.0, "resolution": "NO", "error": 0.0, "bettors": 14, "avg_bet": 1365.0, "category": "Other", "market_type": "Sophisticated"}, {"question": "\\ud83c\\udfc8 2023 NCAAF: Will #1 Georgia beat Georgia Tech?", "prob": 1.0, "resolution": "YES", "error": 0.0, "bettors": 14, "avg_bet": 954.2, "category": "Other", "market_type": "Sophisticated"}, {"question": "Will Trump earn more than 45% of voters in the Economist\'s polls for February?", "prob": 1.0, "resolution": "YES", "error": 0.0, "bettors": 7, "avg_bet": 736.9, "category": "Politics", "market_type": "Sophisticated"}, {"question": "\\ud83c\\udfc8 2023 NCAAF: Will Texas Tech defeat Kansas?", "prob": 1.0, "resolution": "YES", "error": 0.0, "bettors": 13, "avg_bet": 645.8, "category": "Other", "market_type": "Sophisticated"}, {"question": "\\ud83c\\udfc8 2023 NCAAF: Will Syracuse beat Georgia Tech?", "prob": 0.0, "resolution": "NO", "error": 0.0, "bettors": 11, "avg_bet": 277.9, "category": "Other", "market_type": "Large-bet"}, {"question": "Every g7 economy will have negative growth q1 2023", "prob": 0.0, "resolution": "NO", "error": 0.0, "bettors": 10, "avg_bet": 868.3, "category": "Economics", "market_type": "Sophisticated"}, {"question": "Will Biden ever be ahead of Trump in the Economist\'s polls for February?", "prob": 0.0, "resolution": "NO", "error": 0.0, "bettors": 15, "avg_bet": 452.8, "category": "Politics", "market_type": "Large-bet"}, {"question": "Will the US economy grow in 2023?", "prob": 1.0, "resolution": "YES", "error": 0.0, "bettors": 560, "avg_bet": 221.3, "category": "Economics", "market_type": "Large-bet"}, {"question": "\\ud83c\\udfc8 2023 NCAAF: Will Georgia Tech beat Clemson?", "prob": 0.0, "resolution": "NO", "error": 0.0, "bettors": 10, "avg_bet": 475.7, "category": "Other", "market_type": "Sophisticated"}, {"question": "\\ud83c\\udfc8 2024 NCAAF: Will Florida State defeat Georgia Tech?", "prob": 0.0, "resolution": "NO", "error": 0.0, "bettors": 10, "avg_bet": 159.2, "category": "Other", "market_type": "Small-bet"}, {"question": "Will the next Speaker of the House be elected on the first ballot?", "prob": 0.0, "resolution": "NO", "error": 0.0, "bettors": 637, "avg_bet": 489.4, "category": "Other", "market_type": "Sophisticated"}, {"question": "\\ud83c\\udfc8 2023 NCAAF: Will Texas Tech beat BYU?", "prob": 0.0, "resolution": "NO", "error": 0.0, "bettors": 10, "avg_bet": 456.3, "category": "Other", "market_type": "Sophisticated"}, {"question": "Will Romina Pourmokhtari still be the Minister for Climate and the Environment i", "prob": 1.0, "resolution": "YES", "error": 0.0, "bettors": 10, "avg_bet": 671.7, "category": "Climate", "market_type": "Sophisticated"}, {"question": "Will Manifund\'s loan to @MarcusAbramovitch\'s crypto fund be repaid without incid", "prob": 1.0, "resolution": "YES", "error": 0.0, "bettors": 92, "avg_bet": 5603.8, "category": "Crypto", "market_type": "Sophisticated"}, {"question": "American politics 2023: Republicans retain their majority in the Virginia House ", "prob": 0.0, "resolution": "NO", "error": 0.0, "bettors": 22, "avg_bet": 540.8, "category": "Politics", "market_type": "Sophisticated"}, {"question": "Will Atalanta beat Sporting Lisbon in today\'s fixture on Europa League?", "prob": 1.0, "resolution": "YES", "error": 0.0, "bettors": 15, "avg_bet": 614.4, "category": "Other", "market_type": "Sophisticated"}, {"question": "Sport 2023: Will Kansas City Chiefs win Super Bowl?", "prob": 1.0, "resolution": "YES", "error": 0.0, "bettors": 30, "avg_bet": 505.7, "category": "Other", "market_type": "Sophisticated"}, {"question": "Sport 2023: Celtics win NBA finals?", "prob": 0.0, "resolution": "NO", "error": 0.0, "bettors": 40, "avg_bet": 4756.9, "category": "Sports", "market_type": "Sophisticated"}, {"question": "Sport 2023: Will Manchester City win the UEFA Champions League?", "prob": 1.0, "resolution": "YES", "error": 0.0, "bettors": 61, "avg_bet": 1709.3, "category": "Other", "market_type": "Sophisticated"}, {"question": "Will my own fun Crypto Token hit 0,001$ before 2025? (300 mana subsidy)", "prob": 1.0, "resolution": "YES", "error": 0.0, "bettors": 6, "avg_bet": 4909.8, "category": "Crypto", "market_type": "Sophisticated"}, {"question": "Will X start a crypto trading platform this year?", "prob": 0.0, "resolution": "NO", "error": 0.0, "bettors": 12, "avg_bet": 5915.2, "category": "Crypto", "market_type": "Sophisticated"}, {"question": "Will total crypto market cap be above 1T on years end, according to coinmarketca", "prob": 0.0, "resolution": "NO", "error": 0.0, "bettors": 16, "avg_bet": 722.0, "category": "Crypto", "market_type": "Sophisticated"}, {"question": "Conditional on Manifold *not* integrating with crypto in any way, will it have a", "prob": 0.0, "resolution": "NO", "error": 0.0, "bettors": 23, "avg_bet": 439.9, "category": "Crypto", "market_type": "Large-bet"}, {"question": "Will FTX dump all its accumulated crypto on CEXes in Nov 2023", "prob": 0.0, "resolution": "NO", "error": 0.0, "bettors": 23, "avg_bet": 504.0, "category": "Crypto", "market_type": "Sophisticated"}, {"question": "American politics 2023: Donald Trump indicted on federal charges?", "prob": 1.0, "resolution": "YES", "error": 0.0, "bettors": 53, "avg_bet": 1042.0, "category": "Politics", "market_type": "Sophisticated"}, {"question": "Will total crypto market cap be above $1.2T on june 20th, according to coinmarke", "prob": 0.0, "resolution": "NO", "error": 0.0, "bettors": 34, "avg_bet": 1123.4, "category": "Crypto", "market_type": "Sophisticated"}, {"question": "Sport 2023: Will Buffalo Bills win Super Bowl?", "prob": 0.0, "resolution": "NO", "error": 0.0, "bettors": 14, "avg_bet": 442.7, "category": "Other", "market_type": "Large-bet"}, {"question": "Will anyone run for political office with the promise of outsourcing all their d", "prob": 1.0, "resolution": "YES", "error": 0.0, "bettors": 125, "avg_bet": 424.2, "category": "Other", "market_type": "Large-bet"}, {"question": "Will there be another >$100M crypto hack by 1. November?", "prob": 1.0, "resolution": "YES", "error": 0.0, "bettors": 44, "avg_bet": 1300.4, "category": "Crypto", "market_type": "Sophisticated"}, {"question": "American politics 2023: Dianne Feinstein is still in office at year\\u2019s end?", "prob": 0.0, "resolution": "NO", "error": 0.0, "bettors": 87, "avg_bet": 528.0, "category": "Other", "market_type": "Sophisticated"}, {"question": "American politics 2023: Hunter Biden indicted on federal charges?", "prob": 1.0, "resolution": "YES", "error": 0.0, "bettors": 68, "avg_bet": 1072.6, "category": "Politics", "market_type": "Sophisticated"}, {"question": "Will crypto flip into \\"fear\\" on the fear & greed index before February 21 2024?", "prob": 0.0, "resolution": "NO", "error": 0.0, "bettors": 68, "avg_bet": 1394.0, "category": "Crypto", "market_type": "Sophisticated"}, {"question": "Will \\u201cBuild, Baby, Build: The Science and Ethics of Housing\\u201d by Bryan Caplan and", "prob": 0.0, "resolution": "NO", "error": 0.0, "bettors": 8, "avg_bet": 127.1, "category": "Crypto", "market_type": "Small-bet"}]')

class MarketRecord(BaseModel):
    question:    str
    prob:        float
    resolution:  str
    error:       float
    bettors:     int
    avg_bet:     float
    category:    str
    market_type: str

class MarketsResponse(BaseModel):
    markets: list[MarketRecord]
    total:   int
    note:    str

@app.get("/markets", response_model=MarketsResponse)
def markets(
    category:    str | None = Query(default=None, description="Filter by category"),
    market_type: str | None = Query(default=None, description="Filter by market_type"),
    min_error:   float      = Query(default=0.0,  description="Minimum calibration error"),
    max_error:   float      = Query(default=1.0,  description="Maximum calibration error"),
    limit:       int        = Query(default=200,  description="Max records to return"),
) -> MarketsResponse:
    """Return curated sample of 200 resolved Manifold markets with classification."""
    data = _MARKETS_DATA
    if category:
        data = [m for m in data if m["category"].lower() == category.lower()]
    if market_type:
        data = [m for m in data if m["market_type"].lower() == market_type.lower()]
    data = [m for m in data if min_error <= m["error"] <= max_error]
    data = data[:limit]
    return MarketsResponse(
        markets=data,
        total=len(data),
        note="200 representative records from 4,714-market dataset — top 60 by error, bottom 40, random 100 mid-range.",
    )

# AWS Lambda handler
handler = Mangum(app) if Mangum is not None else None
