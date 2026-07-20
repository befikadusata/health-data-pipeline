"""Feature engineering shared by the anomaly detector and the forecaster.

Both read from monthly_reports (the clean fact table) - never raw or
quarantined data. Anomaly features are facility-normalized (z-scores relative
to that facility's own history) so clinics of very different sizes are
compared on a level footing rather than the model just learning "big clinics
look different from small ones."
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text

ANOMALY_FIELDS = ["patients_tested", "suppression_pct", "drug_stock_level", "reporting_delay_days"]
FORECAST_TARGET = "suppression_pct"
FORECAST_HORIZON_MONTHS = 3
FORECAST_LAGS = [1, 2, 3]


def load_monthly_reports(engine) -> pd.DataFrame:
    df = pd.read_sql(text("SELECT * FROM monthly_reports"), engine)
    df["report_month"] = pd.to_datetime(df["report_month"])
    return df.sort_values(["facility_id", "report_month"]).reset_index(drop=True)


def load_facilities(engine) -> pd.DataFrame:
    return pd.read_sql(text("SELECT * FROM facilities"), engine)


def build_anomaly_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (features, z_scores) both indexed like df. z_scores are kept
    separately (not just as the feature matrix) because they double as the
    plain-language "why flagged" explanation for the dashboard."""
    z_scores = pd.DataFrame(index=df.index)
    for field in ANOMALY_FIELDS:
        grouped = df.groupby("facility_id")[field]
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        z_scores[field] = ((df[field] - mean) / std).fillna(0.0)
    return z_scores.copy(), z_scores


def build_forecast_dataset(
    df: pd.DataFrame, facilities: pd.DataFrame
) -> pd.DataFrame:
    """One row per (facility_id, report_month) with lag/rolling features and a
    forward-looking target. Rows without enough history for the lags, or
    without a target that far in the future, are dropped (both are true NaNs,
    not injected data-quality issues) - annotate the drop counts in the eval
    report shown in models/forecast.py so the built-in class carve-outs stay
    visible rather than silently shrinking the dataset."""
    df = df.merge(
        facilities[["facility_id", "baseline_patient_volume", "region"]], on="facility_id"
    )
    df["month_of_year"] = df["report_month"].dt.month

    out = df.copy()
    grouped = out.groupby("facility_id")
    for lag in FORECAST_LAGS:
        out[f"suppression_pct_lag_{lag}"] = grouped["suppression_pct"].shift(lag)
        out[f"patients_tested_lag_{lag}"] = grouped["patients_tested"].shift(lag)
    out["suppression_pct_roll3_mean"] = grouped["suppression_pct"].transform(
        lambda s: s.shift(1).rolling(3).mean()
    )
    out["target_suppression_pct"] = grouped["suppression_pct"].shift(-FORECAST_HORIZON_MONTHS)

    feature_cols = (
        ["suppression_pct", "patients_tested"]
        + [f"suppression_pct_lag_{lag}" for lag in FORECAST_LAGS]
        + [f"patients_tested_lag_{lag}" for lag in FORECAST_LAGS]
        + ["suppression_pct_roll3_mean", "month_of_year", "baseline_patient_volume"]
    )
    required = feature_cols + ["target_suppression_pct"]
    out = out.dropna(subset=required).reset_index(drop=True)
    out.attrs["feature_cols"] = feature_cols
    return out
