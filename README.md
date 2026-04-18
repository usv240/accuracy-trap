# The Accuracy Trap

> Prediction markets are **10.97× less accurate** when retail traders flood in — and I built a real-time detector for it.

Built for [ZerveHack 2026](https://zervehack.devpost.com/).

**Live app:** [accuracy-trap.hub.zerve.cloud](https://accuracy-trap.hub.zerve.cloud)  
**Live API:** [Lambda endpoint](https://pil57aej3zgm64vkhospsca4dq0vvnff.lambda-url.us-east-1.on.aws/docs)

---

## What I Found

Prediction markets are supposed to be the most accurate forecasting tool we have. When people risk real money, they're supposed to research carefully.

I found a systematic failure mode. When a topic goes viral, uninformed retail traders flood in with tiny bets — drowning out the small group of careful, informed forecasters who were already there. The price stops reflecting reality. I call it the **Accuracy Trap**.

The signal is one metric: `avg_bet = total_volume ÷ unique_bettors`.

Across **4,714 resolved binary markets** from Manifold Markets:

| Market type | Calibration error | Median avg bet |
|---|---|---|
| Micro-bet (Retail flood) | **22.3%** | 52 Mana |
| Small-bet | 7.8% | 136 Mana |
| Large-bet | 3.6% | 297 Mana |
| Whale-bet (Sophisticated) | **2.0%** | 720 Mana |

That's a **10.97× accuracy gap** between the worst and best quartile — confirmed at **p < 0.001**, Cohen's d = 1.256 (large effect by any standard).

The obvious objection: maybe viral topics are just harder to predict. I controlled for this. OLS regression shows `log(avg_bet)` predicts calibration error independently of `log(nr_bettors)` — controlling for crowd size, composition still drives accuracy at p < 0.001, t = −35.07. The driver is **who bets**, not how many are watching.

---

## The Best Example

**Same event. Same day. Same question.**

> *Will Donald Trump win the 2024 US Presidential Election?*

Two markets existed simultaneously on Manifold. The only difference: who was betting.

| Version | Avg bet | Prediction | Outcome | Error |
|---|---|---|---|---|
| Retail-flooded | 1,291 Mana | 50% YES | YES | **50%** — coin flip |
| Sophisticated | 3,076 Mana | 99.5% YES | YES | **0.5%** — near-perfect |

Same event. 100× difference in accuracy.

---

## How I Detect It Live

1. Enter any market topic
2. Keyword + category classifier tags it as **retail-driven** or **institutional-driven**
3. For retail-driven topics: Google Trends 7-day momentum scores live Polymarket markets
4. High social score + low avg bet = Accuracy Trap in progress → treat price with caution

Validated cross-correlations:
- **GME 2021**: Google Trends spike *preceded* market reprice by 3 days (Pearson corr = −0.604)
- **BTC 2024**: Market price *led* Google Trends by 7 days (corr = +0.707)

---

## Key Statistics

| Metric | Value |
|---|---|
| Markets analyzed | 4,714 resolved binary markets |
| Retail flood calibration error | 22.3% (95% CI: 21.0%–23.5%) |
| Sophisticated calibration error | 2.0% (95% CI: 1.6%–2.4%) |
| Error multiplier | **10.97×** |
| Welch's t-test | p < 0.001, t = 30.498 |
| Cohen's d | **1.256** (large effect) |
| OLS: log(avg_bet) | β = −0.0673, t = −35.07, p < 0.001 |
| OLS: log(nr_bettors) | β = −0.0098, t = −4.61, p < 0.001 |
| OLS R² | 0.253 (n = 4,714) |
| Polymarket cross-validation | 299 markets, $116.9M USDC — same pattern |

---

## Project Structure

```
├── analysis/
│   ├── accuracy_trap.py                   # Manifold data fetcher + calibration analysis
│   ├── accuracy_trap_curve.png            # The main chart
│   ├── accuracy_trap_results.json         # Pre-computed results (4,714 markets)
│   ├── manifold_resolved_markets.csv      # Full dataset
│   ├── polymarket_validation.py           # Real-money cross-validation
│   ├── polymarket_validation_results.json # Polymarket tier summary
│   └── zerve_notebook.py                  # Zerve canvas notebook (full pipeline)
├── api/
│   └── main_zerve.py                      # FastAPI on AWS Lambda — 7 endpoints, self-contained
├── app/
│   └── zerve_deploy.py                    # Streamlit app hosted on Zerve
├── deploy_lambda.sh                       # Redeploy the Lambda
├── test_zerve_api.py                      # API health check (18/18 endpoints green)
└── test_zerve_ui.py                       # Playwright UI test (screenshots + checks)
```

---

## Running It Locally

```bash
pip install streamlit plotly pandas numpy requests scipy pytrends

# Dashboard (reads from Lambda API automatically)
streamlit run app/zerve_deploy.py

# Re-run the full analysis
python analysis/accuracy_trap.py

# Test the live Lambda API
python test_zerve_api.py
# → Passed: 18/18
```

---

## API Endpoints

Hosted on AWS Lambda (Function URL, no auth required):

| Endpoint | Description |
|---|---|
| `GET /health` | Health check |
| `GET /accuracy-trap` | Full calibration curve + headline stats |
| `GET /lag?category=sports` | Calibration breakdown by topic category |
| `GET /classify?topic=gamestop` | Retail vs institutional classification |
| `GET /markets` | 200 representative resolved markets (filterable) |
| `GET /live-alerts` | Active Polymarket markets scored by social momentum |
| `GET /explain?topic=gamestop` | Full topic analysis with signal explanation |

```bash
# Try it
curl "https://pil57aej3zgm64vkhospsca4dq0vvnff.lambda-url.us-east-1.on.aws/classify?topic=gamestop"
# → {"market_type": "retail_driven", "expected_lag_days": -3, "confidence": 0.76, ...}
```

---

## Data Sources

- [Manifold Markets API](https://api.manifold.markets/v0) — 4,714 resolved binary markets, no auth required
- [Polymarket Gamma API](https://gamma-api.polymarket.com) — live market monitoring + closed market validation
- [Wikipedia Pageviews API](https://wikimedia.org/api/rest_v1/) — real-time social momentum (free, no key)
- [Google Trends via pytrends](https://github.com/GeneralMills/pytrends) — 7-day search momentum (fallback)
- [Yahoo Finance via yfinance](https://github.com/ranaroussi/yfinance) — GME/BTC price history for cross-correlation

---

*The Accuracy Trap is structural, measurable, and predictable.*
