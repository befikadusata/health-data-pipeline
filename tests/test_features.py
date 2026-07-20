import numpy as np
import pandas as pd
import pytest

from models.features import ANOMALY_FIELDS, build_anomaly_features, build_forecast_dataset


def _monthly_reports(n_months=6, facility_id="CL-001", suppression_values=None):
    months = pd.date_range("2024-01-01", periods=n_months, freq="MS")
    if suppression_values is None:
        suppression_values = [80.0] * n_months
    return pd.DataFrame(
        {
            "facility_id": [facility_id] * n_months,
            "report_month": months,
            "patients_tested": [100] * n_months,
            "suppression_pct": suppression_values,
            "drug_stock_level": [50.0] * n_months,
            "reporting_delay_days": [5] * n_months,
        }
    )


def test_build_anomaly_features_zscore_matches_manual_computation():
    values = [80.0, 82.0, 78.0, 95.0]  # last value is the outlier
    df = _monthly_reports(n_months=4, suppression_values=values)
    _, z_scores = build_anomaly_features(df)

    mean = np.mean(values)
    std = np.std(values, ddof=1)
    expected_last = (values[-1] - mean) / std
    assert z_scores["suppression_pct"].iloc[-1] == pytest.approx(expected_last, abs=1e-9)


def test_build_anomaly_features_zero_std_does_not_divide_by_zero():
    """A facility reporting the same value every month has std=0 - the z-score
    should come back as 0.0 (no signal), not NaN or inf."""
    df = _monthly_reports(n_months=3, suppression_values=[80.0, 80.0, 80.0])
    _, z_scores = build_anomaly_features(df)
    assert (z_scores["suppression_pct"] == 0.0).all()
    assert set(z_scores.columns) == set(ANOMALY_FIELDS)


def test_build_forecast_dataset_require_target_false_keeps_recent_rows():
    df = _monthly_reports(n_months=6)
    facilities = pd.DataFrame(
        [{"facility_id": "CL-001", "baseline_patient_volume": 100, "region": "North"}]
    )

    with_target = build_forecast_dataset(df, facilities, require_target=True)
    without_target = build_forecast_dataset(df, facilities, require_target=False)

    # The most recent rows have no 3-month-ahead actual yet, so they only
    # survive when require_target=False - the whole point of scoring.
    assert len(without_target) > len(with_target)
    assert without_target["target_suppression_pct"].isna().any()
    assert not with_target["target_suppression_pct"].isna().any()
