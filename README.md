# The Accuracy Trap

> **Prediction markets are 7.65× less accurate when everyone is watching.**

We analyzed 1,535 resolved binary prediction markets and found that markets flooded with retail traders — many small bets per participant — have **24.5% mean calibration error** compared to **3.2%** in markets dominated by large bets. The accuracy gap is monotonic across all four quartiles, statistically significant (p < 0.001, Cohen's d = 1.285), and caused entirely by *who* bets, not how many people are watching.

The metric is simple: `avg_bet = volume ÷ unique_bettors`. Low avg bet = retail flood = Accuracy Trap.

![Accuracy Trap Curve](analysis/accuracy_trap_curve.png)

---

## Key Finding

| Market Type | Calibration Error | Median Avg Bet |
|---|---|---|
| Micro-bet (Retail flood) | **24.5%** | 42 Mana |
| Small-bet | 9.7% | 113 Mana |
| Large-bet | 4.9% | 240 Mana |
| Whale-bet (Sophisticated) | **3.2%** | 617 Mana |

**Cross-validation:** at the same attention level, markets with small bets show 22.8% error vs 5.0% for large bets — a 4.56× gap. Attention isn't the driver. Composition is.

---

## Real Examples

**The most extreme case:** 100 bettors on whether the US would broker an Israel-Hamas ceasefire. The crowd said 3% chance. It happened. **97% calibration error.**

**Same question, two markets — Trump 2024 election:**
- Sophisticated version (avg bet: 3,076 Mana) → predicted 99.5%, resolved YES → **0.5% error**
- Retail-flooded version (avg bet: 1,291 Mana) → predicted 50.0%, resolved YES → **50% error**

Same event. Same day. 100× difference in accuracy.

---

## Project Structure

```
├── analysis/
│   ├── zerve_notebook.py              # Zerve submission notebook
│   ├── accuracy_trap.py               # Metaculus-based validation analysis
│   ├── manifold_resolved_markets.csv  # 1,535 resolved Manifold markets
│   └── accuracy_trap_results.json     # Pre-computed calibration numbers
├── api/
│   ├── main.py                        # FastAPI — 6 endpoints
│   ├── data_layer.py                  # Data access: CSV + JSON + live APIs
│   └── models.py                      # Pydantic response models
├── app/
│   └── streamlit_app.py               # Interactive 5-tab dashboard
└── requirements.txt
```

---

## Quick Start

```bash
pip install -r requirements.txt

# API server (port 8000)
uvicorn api.main:app --reload

# Dashboard (separate terminal)
streamlit run app/streamlit_app.py
```

The dashboard works standalone — it reads directly from the CSV and JSON files. The FastAPI server is optional; the app falls back to local data if no server is running.

---

## API

| Endpoint | Description |
|---|---|
| `GET /health` | Health check |
| `GET /accuracy-trap` | Full calibration curve from 1,535 markets |
| `GET /lag?category=sports` | Calibration stats by category |
| `GET /classify?topic=gamestop` | Classify retail vs institutional |
| `GET /live-alerts` | Live retail flood signals (Polymarket + Google Trends) |
| `GET /explain?topic=gamestop` | Full topic analysis with social signal |

**Example:**
```bash
curl "http://localhost:8000/classify?topic=israel%20hamas%20ceasefire"
# → {"market_type": "retail_driven", "expected_lag_days": -3, "confidence": 0.68}

curl "http://localhost:8000/accuracy-trap"
# → {"headline": {"error_multiplier": 7.65, "n_markets_analyzed": 1535, ...}}
```

---

## How It Works

**Step 1 — Data:** Fetched 1,535 resolved binary markets from Manifold Markets (no auth required). Each market has a final community probability and a YES/NO outcome.

**Step 2 — Calibration error:** `|resolutionProbability − outcome|` per market.

**Step 3 — Market type:** `avg_bet = volume ÷ unique_bettors`. Quartile split:
- Q1 (avg_bet < 78.5) → retail flood
- Q4 (avg_bet > 368) → sophisticated

**Step 4 — Live detector:** For active Polymarket markets, we pull 7-day Google Trends momentum to flag markets where retail attention is rising before prices reprice.

---

## Data Sources

- [Manifold Markets API](https://api.manifold.markets/v0) — resolved binary markets, no auth required
- [Polymarket Gamma API](https://gamma-api.polymarket.com) — live active markets
- [Google Trends via pytrends](https://github.com/GeneralMills/pytrends) — 7-day social momentum
- [Yahoo Finance via yfinance](https://github.com/ranaroussi/yfinance) — price history for GME/BTC cross-correlation

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `http://localhost:8000` | FastAPI base URL for the dashboard |
| `METACULUS_TOKEN` | *(empty)* | Optional Metaculus API token for `accuracy_trap.py` |
| `ZERVE_NOTEBOOK_URL` | `https://app.zerve.ai/` | Link shown in the sidebar |

---

Built for [ZerveHack 2026](https://zervehack.devpost.com/).
