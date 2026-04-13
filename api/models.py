from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


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
    headline:             dict[str, Any]
    attention_buckets:    list[AttentionBucket]
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


class HealthResponse(BaseModel):
    status: Literal["ok"]
