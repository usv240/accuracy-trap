# The Accuracy Trap — ZerveHack 2026 Submission

## One-Line Pitch
Prediction markets are 7.65× less accurate when retail traders flood in — and we built a real-time detector for it.

---

## 300-Word Devpost Description

Prediction markets are supposed to be the most accurate forecasting tool humans have built. When people risk real money, they research carefully. The crowd aggregates information. The result should beat polls, pundits, and experts.

We found a systematic failure mode: the Accuracy Trap.

When a topic goes viral — an election, a ceasefire, a meme stock — uninformed retail traders flood in with tiny bets. They don't research. They bet on vibes. Their volume drowns out the small number of careful, informed forecasters who were already in the market. The price stops reflecting reality.

We quantified this across 1,535 resolved binary markets from Manifold Markets. Our key metric is simple: `avg_bet = total_volume ÷ unique_bettors`. Low avg bet = retail flood. High avg bet = sophisticated participants.

The finding is stark. Markets in the lowest avg-bet quartile show **24.5% mean calibration error**. Markets in the top quartile show **3.2%**. That's a **7.65× gap** — statistically significant at p < 0.001, Cohen's d = 1.285 (large effect).

The obvious objection: maybe viral topics are just harder to predict. We controlled for this. An OLS regression shows log(avg_bet) predicts calibration error independently of log(nr_bettors) — controlling for crowd size, composition still matters at p < 0.001. And a direct cross-validation at the same attention level (same nr_bettors) shows a 4.56× gap by bet-size alone.

The strongest single example: the Trump 2024 election. Two markets on the same event, same day. The sophisticated version (avg bet: 3,076 Mana) predicted 99.5% and was 0.5% wrong. The retail-flooded version (avg bet: 1,291 Mana) was stuck at 50% — a coin flip. Same event. 100× accuracy difference.

We turned this into a live detector. Enter any topic — it's classified as retail-driven or institutional, with expected reprice lag, confidence score, and active Polymarket alerts scored by Google Trends 7-day momentum.

**The Accuracy Trap is structural, measurable, and predictable.**

---

## Key Numbers

| Metric | Value |
|---|---|
| Markets analyzed | 1,535 resolved binary markets |
| Retail flood calibration error | 24.5% |
| Sophisticated market calibration error | 3.2% |
| Error multiplier | **7.65×** |
| Statistical significance | p < 0.001 |
| Effect size | Cohen's d = 1.285 (Large) |
| OLS: log(avg_bet) p-value | < 0.001 (controlling for crowd size) |
| Cross-validation ratio | 4.56× (same attention, different composition) |

---

## Data Sources

- **Manifold Markets API** — 1,535 resolved binary markets (no auth required)
- **Polymarket Gamma API** — live active market monitoring
- **Google Trends (pytrends)** — 7-day social momentum scoring
- **Yahoo Finance (yfinance)** — GME and BTC price cross-correlation

---

## Tech Stack

- **Zerve notebook** — full reproducible analysis pipeline
- **FastAPI** — 6-endpoint backend with live data
- **Streamlit** — interactive 5-tab dashboard
- **scipy** — Welch's t-test, Cohen's d, OLS regression
- **Plotly** — calibration curves, correlation charts

---

## Video Script Outline (2 min)

1. **(0:00–0:20)** The promise of prediction markets. The flaw nobody talks about.
2. **(0:20–0:45)** Show the 7.65× chart. Explain avg_bet in one sentence.
3. **(0:45–1:10)** The Trump dual-market case — same event, 100× accuracy difference.
4. **(1:10–1:35)** Live monitor demo — Tab 4, classify a topic.
5. **(1:35–2:00)** The OLS regression — this isn't just attention. It's who shows up.
