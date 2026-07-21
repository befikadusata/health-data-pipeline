"""Forecasting model: lag-feature gradient boosting for suppression_pct three
months ahead, per facility (chosen over Prophet - lighter dependency footprint,
faster in CI/Docker, and pairs cleanly with SHAP's TreeExplainer).

Backtest is a time-based split (train on earlier months, test on the most
recent months) - never a random split, which would leak future information
into training via a facility's own nearby months. Evaluated against a naive
carry-forward baseline (predict no change from the current value) since
beating that baseline is the actual bar for a 3-month clinical forecast to be
useful, not an abstract R^2.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from data_gen.config import SEED
from models.features import build_forecast_dataset, load_facilities, load_monthly_reports
from models.mlflow_utils import configure_mlflow
from warehouse.db import get_engine

TEST_MONTHS = 4  # most recent N distinct feature-row months held out for backtest
OUTPUT_DIR = "models/output"


def time_based_split(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    months = sorted(dataset["report_month"].unique())
    cutoff = months[-TEST_MONTHS]
    train = dataset[dataset["report_month"] < cutoff]
    test = dataset[dataset["report_month"] >= cutoff]
    return train, test


def train_and_evaluate() -> dict:
    configure_mlflow()
    engine = get_engine()
    df = load_monthly_reports(engine)
    facilities = load_facilities(engine)

    dataset = build_forecast_dataset(df, facilities)
    feature_cols = dataset.attrs["feature_cols"]

    train, test = time_based_split(dataset)

    model = GradientBoostingRegressor(random_state=SEED, n_estimators=150, max_depth=3)
    model.fit(train[feature_cols], train["target_suppression_pct"])

    predictions = model.predict(test[feature_cols])
    naive_predictions = test["suppression_pct"]  # carry-forward: assume no change

    mae_model = mean_absolute_error(test["target_suppression_pct"], predictions)
    mae_naive = mean_absolute_error(test["target_suppression_pct"], naive_predictions)
    improvement_pct = (mae_naive - mae_model) / mae_naive * 100 if mae_naive else 0.0

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test[feature_cols])
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = (
        pd.Series(mean_abs_shap, index=feature_cols).sort_values(ascending=False).round(4)
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    feature_importance.to_csv(f"{OUTPUT_DIR}/forecast_shap_importance.csv", header=["mean_abs_shap"])

    forecast_out = test[["facility_id", "report_month", "target_suppression_pct"]].copy()
    forecast_out["predicted_suppression_pct"] = predictions
    forecast_out["report_month"] = forecast_out["report_month"].dt.date.astype(str)
    forecast_out.to_csv(f"{OUTPUT_DIR}/forecast_predictions.csv", index=False)

    eval_metrics = {
        "train_rows": len(train),
        "test_rows": len(test),
        "test_month_cutoff": str(sorted(dataset["report_month"].unique())[-TEST_MONTHS].date()),
        "mae_model": round(float(mae_model), 4),
        "mae_naive_baseline": round(float(mae_naive), 4),
        "improvement_over_baseline_pct": round(float(improvement_pct), 2),
        "top_features_by_shap": feature_importance.head(5).to_dict(),
    }
    with open(f"{OUTPUT_DIR}/forecast_eval.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)

    with mlflow.start_run(run_name="forecast-gradient-boosting") as run:
        mlflow.log_params(
            {
                "model_type": "GradientBoostingRegressor",
                "n_estimators": 150,
                "max_depth": 3,
                "random_state": SEED,
                "horizon_months": 3,
                "features": feature_cols,
                "test_months_held_out": TEST_MONTHS,
            }
        )
        mlflow.log_metrics(
            {
                "mae_model": eval_metrics["mae_model"],
                "mae_naive_baseline": eval_metrics["mae_naive_baseline"],
                "improvement_over_baseline_pct": eval_metrics["improvement_over_baseline_pct"],
                "train_rows": eval_metrics["train_rows"],
                "test_rows": eval_metrics["test_rows"],
            }
        )
        mlflow.log_artifact(f"{OUTPUT_DIR}/forecast_eval.json")
        mlflow.log_artifact(f"{OUTPUT_DIR}/forecast_predictions.csv")
        mlflow.log_artifact(f"{OUTPUT_DIR}/forecast_shap_importance.csv")
        # Logged but not registered here - see models/anomaly.py's train step
        # for why train/register are kept as separate DAG tasks.
        mlflow.sklearn.log_model(model, name="model")
        eval_metrics["mlflow_run_id"] = run.info.run_id

    return eval_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the forecasting model")
    parser.add_argument("--output-file", help="also write the JSON result here")
    args = parser.parse_args()

    metrics = train_and_evaluate()
    print(json.dumps(metrics, indent=2))
    if args.output_file:
        Path(args.output_file).write_text(json.dumps(metrics))
