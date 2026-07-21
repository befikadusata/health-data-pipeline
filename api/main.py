"""Serving API: loads the current registered models the same way the `score`
DAG task does (models.registry.load_latest_sklearn_model), then runs live
inference for a single facility's most recent report - it does not just read
back scored_reports, since the brief's point is proving the registry load
path works outside the DAG too.

Models are loaded once at startup (not per-request): reloading a scikit-learn
model from MLflow on every call would make latency depend on the tracking
store, not the request.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models.anomaly import build_reasons
from models.features import (
    ANOMALY_FIELDS,
    build_anomaly_features,
    build_forecast_dataset,
    load_facilities,
    load_monthly_reports,
)
from models.mlflow_utils import configure_mlflow
from models.registry import load_latest_sklearn_model
from warehouse.db import get_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("health_pipeline.api")

MODELS: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_mlflow()
    anomaly_model, anomaly_version = load_latest_sklearn_model("health-anomaly-detector")
    forecast_model, forecast_version = load_latest_sklearn_model("health-suppression-forecaster")
    MODELS["anomaly_model"] = anomaly_model
    MODELS["anomaly_version"] = str(anomaly_version)
    MODELS["forecast_model"] = forecast_model
    MODELS["forecast_version"] = str(forecast_version)
    logger.info(
        "loaded models: anomaly v%s, forecast v%s",
        anomaly_version,
        forecast_version,
    )
    yield
    MODELS.clear()


app = FastAPI(title="health-pipeline-api", lifespan=lifespan)


class AnomalyReason(BaseModel):
    field: str
    z_score: float
    direction: str


class FacilityScoreResponse(BaseModel):
    facility_id: str
    report_month: str
    is_anomaly: bool
    anomaly_score: float
    anomaly_reasons: list[AnomalyReason]
    forecast_next_quarter_suppression_pct: float | None
    model_version_anomaly: str
    model_version_forecast: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models_loaded": bool(MODELS)}


@app.get("/facilities/{facility_id}/score", response_model=FacilityScoreResponse)
def score_facility(facility_id: str) -> FacilityScoreResponse:
    if not MODELS:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    engine = get_engine()
    df = load_monthly_reports(engine)
    facility_df = df[df["facility_id"] == facility_id].reset_index(drop=True)
    if facility_df.empty:
        raise HTTPException(
            status_code=404, detail=f"No monthly_reports rows for facility_id={facility_id}"
        )

    anomaly_model = MODELS["anomaly_model"]
    features, z_scores = build_anomaly_features(facility_df)
    try:
        raw_scores = anomaly_model.decision_function(features[ANOMALY_FIELDS])
        predictions = anomaly_model.predict(features[ANOMALY_FIELDS])
    except (ValueError, KeyError) as exc:
        logger.exception("anomaly model inference failed for facility_id=%s", facility_id)
        raise HTTPException(
            status_code=500,
            detail="Anomaly model inference failed - possible feature/schema mismatch",
        ) from exc
    flagged = pd.Series(predictions == -1, index=facility_df.index)
    reasons_by_row = build_reasons(z_scores, flagged)

    latest_pos = facility_df["report_month"].idxmax()
    report_month = facility_df.loc[latest_pos, "report_month"]

    facilities = load_facilities(engine)
    forecast_dataset = build_forecast_dataset(facility_df, facilities, require_target=False)
    forecast_value = None
    if not forecast_dataset.empty:
        match = forecast_dataset[forecast_dataset["report_month"] == report_month]
        if not match.empty:
            feature_cols = forecast_dataset.attrs["feature_cols"]
            try:
                forecast_value = float(
                    MODELS["forecast_model"].predict(match[feature_cols])[0]
                )
            except (ValueError, KeyError) as exc:
                logger.exception(
                    "forecast model inference failed for facility_id=%s", facility_id
                )
                raise HTTPException(
                    status_code=500,
                    detail="Forecast model inference failed - possible feature/schema mismatch",
                ) from exc

    return FacilityScoreResponse(
        facility_id=facility_id,
        report_month=report_month.date().isoformat(),
        is_anomaly=bool(flagged.loc[latest_pos]),
        anomaly_score=float(-raw_scores[latest_pos]),
        anomaly_reasons=reasons_by_row.loc[latest_pos],
        forecast_next_quarter_suppression_pct=forecast_value,
        model_version_anomaly=MODELS["anomaly_version"],
        model_version_forecast=MODELS["forecast_version"],
    )
