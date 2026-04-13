from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api.data_layer import classify_topic, explain_topic, get_available_categories, get_category_lag, get_live_alerts
from api.models import AccuracyTrapResponse, ClassifyResponse, ExplainResponse, HealthResponse, LagResponse, LiveAlertsResponse

app = FastAPI(
    title="The Accuracy Trap API",
    description="Prediction market retail-flood detector and live monitoring API.",
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
    return HealthResponse(status="ok")


@app.get("/accuracy-trap", response_model=AccuracyTrapResponse)
def accuracy_trap() -> dict:
    """Return the Accuracy Trap calibration curve from real market analysis."""
    from api.data_layer import get_accuracy_trap_data

    return get_accuracy_trap_data()


@app.get("/lag", response_model=LagResponse)
def lag(category: str = Query(..., description="Prediction market category")) -> LagResponse:
    normalized = category.lower().strip()
    if normalized not in get_available_categories():
        raise HTTPException(
            status_code=404,
            detail=f"Unknown category '{category}'. Valid categories: {', '.join(get_available_categories())}",
        )
    data = get_category_lag(normalized)
    return LagResponse(**data)


@app.get("/classify", response_model=ClassifyResponse)
def classify(topic: str = Query(..., min_length=1, description="Topic or market name")) -> ClassifyResponse:
    cleaned = topic.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="topic must not be blank")

    result = classify_topic(cleaned)
    return ClassifyResponse(
        topic=result["topic"],
        market_type=result["market_type"],
        expected_lag_days=result["expected_lag_days"],
        confidence=result["confidence"],
        reasoning=result["reasoning"],
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
