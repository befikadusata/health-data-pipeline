import pandas as pd

from models.anomaly import Z_REASON_THRESHOLD, build_reasons


def test_build_reasons_only_includes_flagged_rows_over_threshold():
    z_scores = pd.DataFrame(
        {
            "patients_tested": [0.1, Z_REASON_THRESHOLD + 1],
            "suppression_pct": [Z_REASON_THRESHOLD + 1, 0.1],
            "drug_stock_level": [0.0, 0.0],
            "reporting_delay_days": [0.0, 0.0],
        }
    )
    flagged = pd.Series([False, True])

    reasons = build_reasons(z_scores, flagged)

    # Row 0 exceeds threshold on suppression_pct but isn't flagged - no reason.
    assert reasons.loc[0] == []
    # Row 1 is flagged and exceeds threshold on patients_tested.
    assert reasons.loc[1] == [
        {"field": "patients_tested", "z_score": Z_REASON_THRESHOLD + 1, "direction": "high"}
    ]


def test_build_reasons_reports_direction_from_sign():
    z_scores = pd.DataFrame(
        {
            "patients_tested": [-(Z_REASON_THRESHOLD + 1)],
            "suppression_pct": [0.0],
            "drug_stock_level": [0.0],
            "reporting_delay_days": [0.0],
        }
    )
    reasons = build_reasons(z_scores, pd.Series([True]))
    assert reasons.loc[0][0]["direction"] == "low"
