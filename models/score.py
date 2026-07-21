"""Scoring DAG task: loads the *registered* anomaly + forecast models from
MLflow (models.registry.load_latest_sklearn_model - the same load path
api/main.py uses), scores every row in the clean fact table, and upserts the
results into scored_reports.

Unlike ingest/validate, score always recomputes over the *entire*
monthly_reports table each run, not just the current run's batch: the anomaly
detector's features are z-scores relative to a facility's own history, so one
newly-arrived row shifts the mean/std for every other row from that facility,
not just the new one. This stays idempotent via ON CONFLICT DO UPDATE on
(facility_id, report_month), so re-running for any period is always safe.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy.dialects.postgresql import insert

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
from warehouse.models import ScoredReport

SCORED_COLUMNS = [
    "facility_id",
    "report_month",
    "is_anomaly",
    "anomaly_score",
    "anomaly_reasons",
    "forecast_next_quarter_suppression_pct",
]


def score_all(engine, anomaly_model, forecast_model) -> pd.DataFrame:
    df = load_monthly_reports(engine)
    facilities = load_facilities(engine)

    features, z_scores = build_anomaly_features(df)
    raw_scores = anomaly_model.decision_function(features[ANOMALY_FIELDS])
    predictions = anomaly_model.predict(features[ANOMALY_FIELDS])

    scored = df[["facility_id", "report_month"]].copy()
    scored["is_anomaly"] = predictions == -1
    scored["anomaly_score"] = -raw_scores
    scored["anomaly_reasons"] = build_reasons(z_scores, scored["is_anomaly"])

    forecast_dataset = build_forecast_dataset(df, facilities, require_target=False)
    feature_cols = forecast_dataset.attrs["feature_cols"]
    forecast_dataset = forecast_dataset.copy()
    forecast_dataset["forecast_next_quarter_suppression_pct"] = forecast_model.predict(
        forecast_dataset[feature_cols]
    )

    scored = scored.merge(
        forecast_dataset[
            ["facility_id", "report_month", "forecast_next_quarter_suppression_pct"]
        ],
        on=["facility_id", "report_month"],
        how="left",
    )
    return scored


def upsert_scored(
    engine,
    run_id: str,
    dag_logical_date: str,
    scored: pd.DataFrame,
    anomaly_version: str,
    forecast_version: str,
) -> int:
    if scored.empty:
        return 0

    rows = scored[SCORED_COLUMNS].to_dict(orient="records")
    for row in rows:
        row["run_id"] = run_id
        row["dag_logical_date"] = dag_logical_date
        row["model_version_anomaly"] = str(anomaly_version)
        row["model_version_forecast"] = str(forecast_version)
        row["is_anomaly"] = bool(row["is_anomaly"])
        row["anomaly_score"] = float(row["anomaly_score"])
        forecast_val = row["forecast_next_quarter_suppression_pct"]
        row["forecast_next_quarter_suppression_pct"] = (
            None if pd.isna(forecast_val) else float(forecast_val)
        )

    update_cols = [c for c in rows[0] if c not in ("facility_id", "report_month")]
    stmt = insert(ScoredReport).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["facility_id", "report_month"],
        set_={c: getattr(stmt.excluded, c) for c in update_cols},
    )
    with engine.begin() as conn:
        conn.execute(stmt)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score monthly_reports with registered models")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dag-logical-date", default=date.today().isoformat())
    parser.add_argument("--output-file", help="also write the JSON result here")
    args = parser.parse_args()

    configure_mlflow()
    anomaly_model, anomaly_version = load_latest_sklearn_model("health-anomaly-detector")
    forecast_model, forecast_version = load_latest_sklearn_model("health-suppression-forecaster")

    engine = get_engine()
    scored = score_all(engine, anomaly_model, forecast_model)
    n = upsert_scored(
        engine, args.run_id, args.dag_logical_date, scored, anomaly_version, forecast_version
    )

    summary = {
        "run_id": args.run_id,
        "rows_scored": n,
        "n_flagged_anomaly": int(scored["is_anomaly"].sum()),
        "n_forecasted": int(scored["forecast_next_quarter_suppression_pct"].notna().sum()),
        "anomaly_model_version": str(anomaly_version),
        "forecast_model_version": str(forecast_version),
    }
    print(json.dumps(summary, indent=2))
    if args.output_file:
        Path(args.output_file).write_text(json.dumps(summary))


if __name__ == "__main__":
    main()
