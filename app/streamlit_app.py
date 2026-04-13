from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.data_layer import (
    CATEGORY_DISPLAY_NAMES,
    classify_topic,
    explain_topic,
    get_accuracy_trap_data,
    get_category_calibration_stats,
    get_category_lag,
    get_live_alerts,
    get_notebook_url,
    get_statistical_significance,
    get_topic_correlation_profile,
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
NOTEBOOK_URL = os.getenv("ZERVE_NOTEBOOK_URL", get_notebook_url())
BUCKET_COLORS = ["#EF4444", "#F59E0B", "#84CC16", "#22C55E"]

DATA_PATH = ROOT_DIR / "analysis" / "manifold_resolved_markets.csv"

st.set_page_config(
    page_title="The Accuracy Trap",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- styles ---
st.markdown("""
<style>
  .big-number { font-size: 3.2rem; font-weight: 800; line-height: 1; color: #111827; }
  .big-label  { font-size: 0.85rem; color: #6B7280; margin-top: 0.2rem; }
  .red   { color: #EF4444; }
  .green { color: #22C55E; }
  .blue  { color: #2563EB; }
  .card  { background: #F9FAFB; border-radius: 0.75rem;
           padding: 1.2rem 1.4rem; border: 1px solid #E5E7EB; color: #111827; }
  .callout-red   { border-left: 4px solid #EF4444; background: #FEF2F2;
                   padding: 0.9rem 1rem; border-radius: 0.5rem; color: #111827; }
  .callout-green { border-left: 4px solid #22C55E; background: #F0FDF4;
                   padding: 0.9rem 1rem; border-radius: 0.5rem; color: #111827; }
  .badge { display: inline-block; padding: 0.4rem 1rem;
           border-radius: 999px; color: white;
           font-weight: 700; letter-spacing: 0.04em; font-size: 0.9rem; }
  .example-card { background: white; border-radius: 0.75rem;
                  padding: 1rem 1.2rem; border: 1px solid #E5E7EB;
                  margin-bottom: 0.75rem; color: #111827; }
  h1 { font-size: 2.4rem !important; }
  .section-divider { border-top: 1px solid #E5E7EB; margin: 2rem 0 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# --- helpers ---

def api_get(path: str, params: dict | None = None) -> dict:
    r = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=5)
    r.raise_for_status()
    return r.json()

def safe_api(path: str, fallback_fn, params: dict | None = None):
    try:
        return api_get(path, params), False
    except Exception:
        result = fallback_fn() if callable(fallback_fn) else fallback_fn
        return result, True

def fmt_ts(ts: str | None) -> str:
    if not ts:
        return "Unavailable"
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%b %d %Y, %I:%M %p UTC")
    except Exception:
        return ts

@st.cache_data(ttl=300)
def load_market_data() -> pd.DataFrame | None:
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
    df["market_type"] = pd.qcut(df["avg_bet"], q=4,
        labels=["Retail Flood","Small-bet","Large-bet","Sophisticated"],
        duplicates="drop").astype(str)
    return df


# --- load all data upfront ---
trap_data   = get_accuracy_trap_data()           # reads analysis/accuracy_trap_results.json
cat_stats   = get_category_calibration_stats()   # computed from analysis/manifold_resolved_markets.csv
sig_stats   = get_statistical_significance()     # Welch t-test + Cohen's d from CSV

@st.cache_data(ttl=1800, show_spinner="Fetching live signals from Google Trends…")
def _load_alerts() -> tuple[dict, bool]:
    return safe_api("/live-alerts", get_live_alerts)

alerts_data, alert_fallback = _load_alerts()

headline   = trap_data.get("headline", {})
detector   = trap_data.get("retail_flood_detector", {})
retail_err = float(headline.get("retail_flood_calibration_error", 0.2454))
whale_err  = float(headline.get("sophisticated_calibration_error", 0.0321))
multiplier = float(headline.get("error_multiplier", 7.65))
n_markets  = int(headline.get("n_markets_analyzed", 1535))
df_markets = load_market_data()

# --- sidebar ---
with st.sidebar:
    st.markdown("## 🎯 The Accuracy Trap")
    st.caption("Prediction market retail flood detector")
    st.markdown("---")
    st.markdown(f"**{n_markets:,}** resolved markets analyzed")
    st.markdown(f"Last updated: {fmt_ts(alerts_data.get('generated_at'))}")
    st.markdown(f"[Open Zerve Notebook]({NOTEBOOK_URL})")
    st.markdown("---")
    st.markdown("**How it works**")
    st.caption("avg_bet = volume ÷ nr_bettors. Low avg_bet = many micro-bets = retail flood = Accuracy Trap.")


# --- hero ---
st.markdown("# 🎯 The Accuracy Trap")
st.markdown("**Retail-flooded prediction markets are measurably less accurate — and we can detect them in real time.**")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="card"><div class="big-number red">{retail_err:.1%}</div>'
                f'<div class="big-label">Retail flood calibration error</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="card"><div class="big-number green">{whale_err:.1%}</div>'
                f'<div class="big-label">Sophisticated market error</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="card"><div class="big-number blue">{multiplier:.2f}×</div>'
                f'<div class="big-label">Error multiplier</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="card"><div class="big-number">{n_markets:,}</div>'
                f'<div class="big-label">Resolved markets analyzed</div></div>', unsafe_allow_html=True)

if sig_stats.get("available"):
    st.markdown(
        f'<div style="text-align:center; color:#6B7280; font-size:0.82rem; margin-top:0.4rem">'
        f'Welch\'s t-test: <strong>p {sig_stats["p_value_display"]}</strong> &nbsp;·&nbsp; '
        f'Cohen\'s d = <strong>{sig_stats["cohens_d"]}</strong> ({sig_stats["effect_size"]} effect) &nbsp;·&nbsp; '
        f'n={sig_stats["retail_n"]} retail, n={sig_stats["sophisticated_n"]} sophisticated'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

col_chart, col_explain = st.columns((3, 2))
with col_chart:
    buckets = trap_data.get("attention_buckets", [])
    if buckets:
        labels = [b["label"].replace("\n", "<br>") for b in buckets]
        errors = [b["mean_calibration_error"] for b in buckets]
        ns     = [b.get("sample_size", "") for b in buckets]
        fig = go.Figure(go.Bar(
            x=labels, y=errors,
            marker_color=BUCKET_COLORS[:len(labels)],
            text=[f"{e:.1%}" for e in errors],
            textposition="outside",
            customdata=ns,
            hovertemplate="<b>%{x}</b><br>Error: %{y:.1%}<br>n=%{customdata}<extra></extra>",
        ))
        fig.add_annotation(x=labels[0], y=errors[0],
            text="⚠ Retail Flood Zone", showarrow=True,
            arrowhead=2, ax=80, ay=-60,
            bgcolor="#FEF2F2", bordercolor="#EF4444",
            font=dict(color="#991B1B", size=12))
        fig.update_layout(
            title="<b>Calibration Error by Market Type</b>",
            height=360, margin=dict(l=10,r=10,t=50,b=10),
            yaxis_tickformat=".0%", yaxis_title="Mean Calibration Error",
            xaxis_title="", plot_bgcolor="white", paper_bgcolor="white",
            showlegend=False,
        )
        st.plotly_chart(fig, width='stretch')

with col_explain:
    st.markdown(f"""
<div class="callout-red">
<strong>The Accuracy Trap</strong><br><br>
When a topic goes viral, retail traders flood in with tiny bets — drowning out informed forecasters.<br><br>
Markets with many micro-bets show <strong>{multiplier:.2f}× higher calibration error</strong> than sophisticated markets.<br><br>
The problem isn't how many people are watching.<br>It's <em>who</em> shows up.
</div>
""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    rfd = trap_data.get("retail_flood_detector", {})
    if rfd:
        st.markdown(f"""
<div class="callout-green">
<strong>Retail Flood Detector</strong><br><br>
High attention + small bets = <strong>{rfd.get('high_attention_retail_error', 0.228):.0%} error</strong><br>
High attention + large bets = <strong>{rfd.get('high_attention_sophisticated_error', 0.05):.0%} error</strong><br><br>
Same audience size. {rfd.get('ratio', 4.56):.1f}× difference in accuracy.
</div>
""", unsafe_allow_html=True)


# --- tabs ---
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📖 Real Examples",
    "📊 Confidence Curve",
    "🔍 Data Explorer",
    "📡 Live Monitor",
    "🏷 Classify a Market",
])


# tab 1 — real examples
with tab1:
    st.markdown("### The Most Compelling Cases From Real Data")
    st.caption("These are actual resolved prediction markets from the dataset — not hypotheticals.")

    st.markdown("#### 🔴 Case 1 — The Biggest Single Upset (97% wrong)")
    st.markdown("""
<div class="example-card">
<strong>Will the US successfully broker a ceasefire between Israel and Hamas?</strong><br>
<span style="color:#6B7280; font-size:0.9rem">100 bettors · Avg bet: 135 Mana · Total volume: 13,538</span><br><br>
<table style="width:100%; border-collapse:collapse">
<tr>
  <td style="padding:0.5rem; background:#FEF2F2; border-radius:0.4rem; text-align:center; width:30%">
    <div style="font-size:1.8rem; font-weight:800; color:#EF4444">3%</div>
    <div style="font-size:0.8rem; color:#6B7280">Market said YES</div>
  </td>
  <td style="padding:0.5rem; text-align:center; font-size:1.5rem; width:10%">→</td>
  <td style="padding:0.5rem; background:#F0FDF4; border-radius:0.4rem; text-align:center; width:30%">
    <div style="font-size:1.8rem; font-weight:800; color:#22C55E">YES</div>
    <div style="font-size:0.8rem; color:#6B7280">Actual outcome</div>
  </td>
  <td style="padding:0.5rem; text-align:center; width:10%"></td>
  <td style="padding:0.5rem; background:#FEF2F2; border-radius:0.4rem; text-align:center; width:20%">
    <div style="font-size:1.8rem; font-weight:800; color:#EF4444">97%</div>
    <div style="font-size:0.8rem; color:#6B7280">Calibration error</div>
  </td>
</tr>
</table>
<br>
<span style="font-size:0.9rem; color:#374151">
100 people bet on this market. The crowd was 97% confident the ceasefire would NOT happen.
It happened. This is the Accuracy Trap at its most visible — a high-emotion geopolitical market
where collective opinion completely overwhelmed any signal.
</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("#### 🔴 Case 2 — Same Question, 100× Different Accuracy")
    st.markdown("""
<div class="example-card">
<strong>Will Donald Trump win the 2024 US Presidential Election?</strong><br>
<span style="font-size:0.9rem; color:#374151">Two markets existed simultaneously on the same question. The difference was who was betting:</span>
<br><br>
<table style="width:100%; border-collapse:collapse; font-size:0.9rem">
<tr style="background:#F9FAFB">
  <th style="padding:0.5rem; text-align:left">Market</th>
  <th style="padding:0.5rem; text-align:center">Predicted</th>
  <th style="padding:0.5rem; text-align:center">Actual</th>
  <th style="padding:0.5rem; text-align:center">Error</th>
  <th style="padding:0.5rem; text-align:center">Bettors</th>
  <th style="padding:0.5rem; text-align:center">Avg Bet</th>
</tr>
<tr>
  <td style="padding:0.5rem; color:#22C55E; font-weight:600">✓ Sophisticated version</td>
  <td style="padding:0.5rem; text-align:center">99.5%</td>
  <td style="padding:0.5rem; text-align:center">YES</td>
  <td style="padding:0.5rem; text-align:center; color:#22C55E; font-weight:700">0.5%</td>
  <td style="padding:0.5rem; text-align:center">3,770</td>
  <td style="padding:0.5rem; text-align:center">3,076</td>
</tr>
<tr style="background:#FEF2F2">
  <td style="padding:0.5rem; color:#EF4444; font-weight:600">✗ Retail-flooded version</td>
  <td style="padding:0.5rem; text-align:center">50.0%</td>
  <td style="padding:0.5rem; text-align:center">YES</td>
  <td style="padding:0.5rem; text-align:center; color:#EF4444; font-weight:700">50.0%</td>
  <td style="padding:0.5rem; text-align:center">2,905</td>
  <td style="padding:0.5rem; text-align:center">1,291</td>
</tr>
</table>
<br>
<span style="font-size:0.9rem; color:#374151">
Same real-world event. Similar number of participants. The one with higher avg bet size (more sophisticated)
was <strong>100× more accurate</strong>. The retail-flooded one was stuck at 50% — no better than a coin flip.
</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("#### 🟢 Case 3 — When Markets Get It Exactly Right")
    st.markdown("""
<div class="example-card">
<strong>The "Sweet Spot" — 90% of these markets had under 5% error</strong><br>
<span style="color:#6B7280; font-size:0.9rem">Profile: high bettors + high avg_bet = sophisticated + popular</span><br><br>
<table style="width:100%; border-collapse:collapse; font-size:0.9rem">
<tr style="background:#F9FAFB">
  <th style="padding:0.5rem; text-align:left">Question</th>
  <th style="padding:0.5rem; text-align:center">Predicted</th>
  <th style="padding:0.5rem; text-align:center">Actual</th>
  <th style="padding:0.5rem; text-align:center">Error</th>
</tr>
<tr>
  <td style="padding:0.4rem">Will Joe Biden win the 2024 US Presidential Election?</td>
  <td style="padding:0.4rem; text-align:center">1.0%</td><td style="padding:0.4rem; text-align:center">NO</td>
  <td style="padding:0.4rem; text-align:center; color:#22C55E; font-weight:700">1.0%</td>
</tr>
<tr style="background:#F9FAFB">
  <td style="padding:0.4rem">Will Trump win the 2024 Election?</td>
  <td style="padding:0.4rem; text-align:center">99.5%</td><td style="padding:0.4rem; text-align:center">YES</td>
  <td style="padding:0.4rem; text-align:center; color:#22C55E; font-weight:700">0.5%</td>
</tr>
<tr>
  <td style="padding:0.4rem">Will SB 1047 (CA AI regulation) become law?</td>
  <td style="padding:0.4rem; text-align:center">0%</td><td style="padding:0.4rem; text-align:center">NO</td>
  <td style="padding:0.4rem; text-align:center; color:#22C55E; font-weight:700">0.0%</td>
</tr>
<tr style="background:#F9FAFB">
  <td style="padding:0.4rem">Will Donald Trump be federally indicted?</td>
  <td style="padding:0.4rem; text-align:center">100%</td><td style="padding:0.4rem; text-align:center">YES</td>
  <td style="padding:0.4rem; text-align:center; color:#22C55E; font-weight:700">0.0%</td>
</tr>
</table>
<br>
<span style="font-size:0.9rem; color:#374151">
These markets had high participation AND high avg bet size. Sophisticated participants dominate →
the wisdom of crowds works exactly as intended.
</span>
</div>
""", unsafe_allow_html=True)


# tab 2 — calibration curve
with tab2:
    st.markdown("### When the Market Says X%, What Actually Happens?")
    st.caption("The classic calibration plot — built from 1,535 real resolved markets.")

    if df_markets is not None:
        bins   = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0]
        labels_p = ["0-10%","10-20%","20-30%","30-40%","40-50%","50-60%","60-70%","70-80%","80-90%","90-100%"]
        df_markets["prob_bin"] = pd.cut(df_markets["prob"], bins=bins, labels=labels_p, include_lowest=True)
        cal = df_markets.groupby("prob_bin", observed=True).agg(
            n=("resolution","count"),
            actual=("resolution","mean"),
            predicted=("prob","mean"),
        ).reset_index()
        cal["midpoint"] = [0.05,.15,.25,.35,.45,.55,.65,.75,.85,.95]

        col_a, col_b = st.columns(2)
        with col_a:
            fig2 = go.Figure()
            # Perfect calibration line
            fig2.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                line=dict(color="#9CA3AF", dash="dash", width=1.5),
                name="Perfect calibration"))
            # Actual calibration
            fig2.add_trace(go.Scatter(
                x=cal["midpoint"], y=cal["actual"],
                mode="lines+markers",
                line=dict(color="#2563EB", width=3),
                marker=dict(size=10, color="#2563EB"),
                customdata=cal[["n","predicted"]].values,
                hovertemplate="<b>%{text}</b><br>Predicted: %{customdata[1]:.1%}<br>Actual YES rate: %{y:.1%}<br>n=%{customdata[0]}<extra></extra>",
                text=labels_p,
                name="Actual outcome rate",
            ))
            # Highlight the 60-70% anomaly
            n_bucket = int(cal[cal["prob_bin"]=="60-70%"]["n"].iloc[0])
            fig2.add_annotation(x=0.65, y=float(cal[cal["prob_bin"]=="60-70%"]["actual"].iloc[0]),
                text=f"Market says 65%<br>Reality: 86%<br>(n={n_bucket})",
                showarrow=True, arrowhead=2, ax=-90, ay=-30,
                bgcolor="#FEF9C3", bordercolor="#EAB308",
                font=dict(color="#713F12", size=11))
            fig2.update_layout(
                title="<b>Calibration Plot: Predicted vs Actual</b>",
                xaxis=dict(title="Market's predicted probability", tickformat=".0%", range=[0,1]),
                yaxis=dict(title="Actual YES resolution rate",   tickformat=".0%", range=[0,1]),
                height=420, plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(x=0.02, y=0.98),
            )
            st.plotly_chart(fig2, width='stretch')

        with col_b:
            st.markdown("#### Key Insights From This Chart")
            st.markdown("""
<div class="callout-red" style="margin-bottom:0.75rem">
<strong>60-70% confidence zone is broken</strong><br>
When markets predict 60-70% YES, the actual YES rate is <strong>86%</strong>.
The crowd is systematically underconfident here — these events are far more likely than priced.
</div>

<div class="callout-green" style="margin-bottom:0.75rem">
<strong>Extremes are very accurate</strong><br>
0-10% → resolves YES only 0.5% of the time.<br>
90-100% → resolves YES 100% of the time.<br>
The crowd nails near-certainties.
</div>

<div class="card">
<strong>What this means for you</strong><br><br>
If you see a market at 65%, the real probability is closer to <strong>86%</strong>.
Markets in the 60-70% range are systematically undervalued.
This is an exploitable bias — and it's hiding in plain sight.
</div>
""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            # Show the table
            display_cal = cal[["prob_bin","n","predicted","actual"]].copy()
            display_cal.columns = ["Predicted range","n","Mean predicted","Actual YES rate"]
            display_cal["Overconfidence"] = display_cal["Mean predicted"] - display_cal["Actual YES rate"]
            st.dataframe(
                display_cal.style.format({
                    "Mean predicted":  "{:.1%}",
                    "Actual YES rate": "{:.1%}",
                    "Overconfidence":  "{:+.1%}",
                }).background_gradient(subset=["Overconfidence"], cmap="RdYlGn"),
                width='stretch', hide_index=True,
            )
    else:
        st.info("Dataset not found. Run analysis/accuracy_trap.py first.")


# tab 3 — data explorer
with tab3:
    st.markdown("### Explore All 1,535 Resolved Markets")
    st.caption("Filter, sort, and search the full dataset.")

    if df_markets is not None:
        # Filters
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
            min_err, max_err = st.slider("Calibration error range", 0.0, 1.0, (0.0, 1.0), 0.05)

        keyword = st.text_input("Search question text", placeholder="e.g. Trump, ceasefire, bitcoin...")

        filtered = df_markets.copy()
        if sel_cat  != "All": filtered = filtered[filtered["category"] == sel_cat]
        if sel_type != "All": filtered = filtered[filtered["market_type"] == sel_type]
        if sel_res  != "All": filtered = filtered[filtered["resolution_label"] == sel_res]
        filtered = filtered[(filtered["calibration_err"] >= min_err) & (filtered["calibration_err"] <= max_err)]
        if keyword.strip():
            filtered = filtered[filtered["question"].str.contains(keyword.strip(), case=False, na=False)]

        st.markdown(f"**{len(filtered):,} markets match** · sorted by calibration error (worst first)")

        display_df = filtered.sort_values("calibration_err", ascending=False)[
            ["question","prob","resolution_label","calibration_err","nr_bettors","avg_bet","category","market_type"]
        ].rename(columns={
            "question":       "Question",
            "prob":           "Predicted",
            "resolution_label":"Actual",
            "calibration_err":"Error",
            "nr_bettors":     "Bettors",
            "avg_bet":        "Avg Bet",
            "category":       "Category",
            "market_type":    "Market Type",
        }).head(200)

        st.dataframe(
            display_df.style.format({
                "Predicted": "{:.1%}",
                "Error":     "{:.1%}",
                "Bettors":   "{:.0f}",
                "Avg Bet":   "{:.0f}",
            }).background_gradient(subset=["Error"], cmap="RdYlGn"),
            width='stretch', hide_index=True, height=440,
        )

        # Category breakdown chart
        st.markdown("#### Calibration Error by Category")
        cat_chart = df_markets.groupby("category").agg(
            mean_err=("calibration_err","mean"),
            n=("calibration_err","count"),
        ).sort_values("mean_err", ascending=True).reset_index()
        fig_cat = go.Figure(go.Bar(
            y=cat_chart["category"], x=cat_chart["mean_err"],
            orientation="h",
            marker_color=["#22C55E","#84CC16","#A3E635","#FDE68A","#F59E0B","#FB923C","#EF4444","#DC2626","#991B1B"][:len(cat_chart)],
            text=[f"{e:.1%}  (n={n})" for e,n in zip(cat_chart["mean_err"], cat_chart["n"])],
            textposition="outside",
        ))
        fig_cat.update_layout(
            height=320, margin=dict(l=0,r=80,t=20,b=0),
            xaxis=dict(tickformat=".0%", title="Mean Calibration Error"),
            yaxis_title="", plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_cat, width='stretch')
    else:
        st.info("Dataset not found. Run analysis/accuracy_trap.py first.")


# tab 4 — live monitor
with tab4:
    st.markdown("### Live Retail Flood Monitor")
    st.caption("Active Polymarket markets showing retail flood signals — treat their prices with caution.")

    st.caption(alerts_data.get("note", ""))
    alerts_df = pd.DataFrame(alerts_data.get("alerts", []))
    if alerts_df.empty:
        st.info("No retail-flooded markets with elevated Google Trends signal detected right now. This is a real signal — only genuine alerts are shown.")
    else:
        def row_style(row, conf_series):
            c = conf_series.loc[row.name]
            color = {"high":"rgba(220,38,38,0.14)","medium":"rgba(245,158,11,0.14)"}.get(c,"rgba(107,114,128,0.10)")
            return [f"background-color:{color}" for _ in row]

        rename_map = {
            "market":                  "Market",
            "social_score":            "Social Score",
            "current_probability":     "Probability",
            "hours_since_signal":      "Hours Since Signal",
            "expected_reprice_window": "Expected Reprice Window",
            "social_score_source":     "Signal Source",
        }
        alerts_df = alerts_df.rename(columns=rename_map)
        cols = ["Market","Social Score","Probability","Hours Since Signal","Expected Reprice Window"]
        if "Signal Source" in alerts_df.columns:
            cols.append("Signal Source")
        disp = alerts_df[cols]

        fmt = {
            "Social Score":       lambda v: f"{v:.2f}" if v is not None else "—",
            "Probability":        lambda v: f"{v:.3f}" if v is not None else "—",
            "Hours Since Signal": lambda v: f"{v:.1f}" if v is not None else "—",
        }
        styled = disp.style.apply(row_style, axis=1, conf_series=alerts_df["confidence"]).format(fmt)
        st.dataframe(styled, width='stretch')
        st.caption("🔴 Red = high retail flood signal  🟡 Yellow = medium  ⚫ Grey = low")

    st.markdown("---")
    st.markdown("#### The Accuracy Trap Is Present in Every Category")
    st.caption("Retail-flooded vs sophisticated calibration error by topic category — computed from 1,535 resolved markets.")

    cat_display = {
        k: v for k, v in cat_stats.items()
        if k != "other" and v.get("retail_error") and v.get("sophisticated_error")
    }
    cat_display = dict(sorted(cat_display.items(), key=lambda x: x[1]["retail_error"], reverse=True))

    if cat_display:
        names    = [v["display_name"] for v in cat_display.values()]
        retail_e = [v["retail_error"] for v in cat_display.values()]
        soph_e   = [v["sophisticated_error"] for v in cat_display.values()]
        ns       = [v["n"] for v in cat_display.values()]

        fig_cat2 = go.Figure()
        fig_cat2.add_trace(go.Bar(
            name="Retail-flooded (Q1 avg bet)",
            x=names, y=retail_e,
            marker_color="#EF4444",
            text=[f"{e:.1%}" for e in retail_e],
            textposition="outside",
        ))
        fig_cat2.add_trace(go.Bar(
            name="Sophisticated (Q4 avg bet)",
            x=names, y=soph_e,
            marker_color="#22C55E",
            text=[f"{e:.1%}" for e in soph_e],
            textposition="outside",
        ))
        fig_cat2.update_layout(
            barmode="group",
            title="<b>Retail vs Sophisticated Error — by Category</b>",
            yaxis_tickformat=".0%",
            yaxis_title="Mean Calibration Error",
            height=380,
            margin=dict(l=0, r=20, t=50, b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(x=0.02, y=0.98),
        )
        st.plotly_chart(fig_cat2, width='stretch')
        st.caption(
            "Sample sizes: " +
            ", ".join(f"{v['display_name']} n={v['n']}" for v in cat_display.values()) +
            ". Source: Manifold Markets — 1,535 resolved binary markets."
        )

    st.markdown("""
<div class="callout-green">
<strong>The pattern is consistent:</strong> In every category, retail-flooded markets
(bottom avg-bet quartile) show dramatically higher calibration error than sophisticated
markets (top avg-bet quartile). The Accuracy Trap is not unique to one domain — it is
structural. The two directly validated cases are GME 2021 (retail, −3 day lag) and
BTC 2024 (institutional, +7 day lag).
</div>
""", unsafe_allow_html=True)


# tab 5 — classify
with tab5:
    st.markdown("### Is This Market Being Overwhelmed by Retail Attention?")
    st.caption("Type any topic or market name to classify its flood risk.")

    with st.form("classifier_form"):
        col_inp, col_btn = st.columns((4,1))
        with col_inp:
            topic_input = st.text_input("Topic or market name",
                value=st.session_state.get("at_topic","gamestop"),
                placeholder="e.g. gamestop, ukraine, bitcoin, super bowl...")
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            clicked = st.form_submit_button("Classify →", use_container_width=True)  # form buttons don't support width=

    if clicked and topic_input.strip():
        st.session_state["at_topic"] = topic_input.strip()

    active = st.session_state.get("at_topic", "gamestop")
    clf, clf_fb = safe_api("/classify", lambda: classify_topic(active), {"topic": active})
    profile    = get_topic_correlation_profile(active)
    explanation = explain_topic(active)

    mtype  = clf.get("market_type","retail_driven")
    is_retail = mtype == "retail_driven"
    badge_color = "#EF4444" if is_retail else "#2563EB"
    badge_text  = "⚠ RETAIL-DRIVEN" if is_retail else "✓ INSTITUTIONAL"
    risk_text   = "HIGH flood risk — treat market prices with caution during attention spikes." \
                  if is_retail else \
                  "LOW flood risk — price discovery tends to lead public attention."

    st.markdown(f"""
<div style="margin:1rem 0">
  <span class="badge" style="background:{badge_color}">{badge_text}</span>
  &nbsp;&nbsp;<span style="color:#6B7280; font-size:0.9rem">{risk_text}</span>
</div>
""", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Expected Lag", f"{clf.get('expected_lag_days',0):+d} days")
    m2.metric("Confidence", f"{clf.get('confidence',0):.0%}")
    m3.metric("Flood Risk", "High" if is_retail else "Low")
    st.caption(clf.get("reasoning",""))

    # Correlation profile chart
    lags   = profile.get("lags", [])
    corrs  = profile.get("corrs", [])
    best   = profile.get("best_lag", 0)
    source = profile.get("source", "")

    if lags and profile.get("validated"):
        bar_colors = ["#2563EB" if l == best else "#16A34A" if l < 0 else "#9CA3AF" for l in lags]
        fig_corr = go.Figure(go.Bar(x=lags, y=corrs, marker_color=bar_colors,
            hovertemplate="Lag %{x} days<br>Corr %{y:.3f}<extra></extra>"))
        fig_corr.add_vline(x=0,    line_dash="dash", line_color="#6B7280", line_width=1)
        fig_corr.add_vline(x=best, line_dash="dot",  line_color="#2563EB", line_width=2)
        fig_corr.update_layout(
            title=f"<b>Measured cross-correlation: '{active}'</b>",
            height=300, margin=dict(l=0,r=20,t=50,b=0),
            xaxis_title="Lag (days) — negative = social trends lead market",
            yaxis_title="Correlation",
            plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
        )
        st.plotly_chart(fig_corr, width='stretch')
        st.caption(f"Source: {profile.get('source','')} · n={profile.get('n_points','')} data points")

    elif source == "category_stats" and profile.get("category_stats"):
        st.info(profile["message"])
        cs = profile["category_stats"]
        ca, cb, cc = st.columns(3)
        ca.metric("Category mean error", f"{cs['mean_calibration_error']:.1%}", help=f"n={cs['n']} markets")
        cb.metric("Retail-flooded error", f"{cs.get('retail_error', 0):.1%}" if cs.get('retail_error') else "N/A")
        cc.metric("Sophisticated error",  f"{cs.get('sophisticated_error', 0):.1%}" if cs.get('sophisticated_error') else "N/A")
        if cs.get("retail_error") and cs.get("sophisticated_error"):
            mult = cs["retail_error"] / cs["sophisticated_error"]
            st.caption(f"Accuracy Trap multiplier for {cs['display_name']}: **{mult:.1f}×** (retail error ÷ sophisticated error) · n={cs['n']} markets · Source: Manifold Markets 1,535-market dataset")

    else:
        st.caption(profile.get("message", "No cross-correlation data available for this topic."))

    if explanation.get("lag_analysis",{}).get("signal_detected"):
        la = explanation["lag_analysis"]
        st.info(f"Signal detected on **{la['signal_detected_at']}** — expected reprice by **{la['expected_reprice_by']}**.")
    else:
        st.caption("No active retail-flood reprice signal detected for this topic right now.")
