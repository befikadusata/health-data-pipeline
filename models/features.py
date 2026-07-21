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
MIN_HISTORY_MONTHS = 2  # a z-score needs at least 2 points to have a defined std
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
    plain-language "why flagged" explanation for the dashboard.

    A facility with fewer than MIN_HISTORY_MONTHS rows has an undefined
    z-score (no real history to compare against) - those get filled to 0.0
    like any other zero-std case so the model still has a numeric feature to
    fit/predict on, but that 0.0 means "can't judge yet", not "confirmed
    normal". Callers that report is_anomaly to a user should treat rows
    flagged via z_scores.attrs["insufficient_history"] as unscored rather
    than trusting a clean read.
    """
    z_scores = pd.DataFrame(index=df.index)
    history_count = df.groupby("facility_id")["facility_id"].transform("count")
    for field in ANOMALY_FIELDS:
        grouped = df.groupby("facility_id")[field]
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        z_scores[field] = ((df[field] - mean) / std).fillna(0.0)
    z_scores.attrs["insufficient_history"] = history_count < MIN_HISTORY_MONTHS
    return z_scores.copy(), z_scores


def build_forecast_dataset(
    df: pd.DataFrame, facilities: pd.DataFrame, require_target: bool = True
) -> pd.DataFrame:
    """One row per (facility_id, report_month) with lag/rolling features and a
    forward-looking target. Rows without enough history for the lags are
    dropped (a true NaN, not an injected data-quality issue) - annotate the
    drop counts in the eval report shown in models/forecast.py so the
    built-in class carve-outs stay visible rather than silently shrinking the
    dataset.

    require_target=False is for scoring (models/score.py, api/main.py): the
    whole point of scoring the most recent months is that the actual 3-months-
    ahead value doesn't exist yet, so rows can't be dropped for lacking it."""
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
    required = feature_cols + (["target_suppression_pct"] if require_target else [])
    out = out.dropna(subset=required).reset_index(drop=True)
    out.attrs["feature_cols"] = feature_cols
    return out
