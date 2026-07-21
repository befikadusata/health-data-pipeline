import pandas as pd
import pytest

import models.anomaly as anomaly_module
from models.anomaly import Z_REASON_THRESHOLD, build_reasons, evaluate_against_ground_truth


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


def test_evaluate_against_ground_truth_confusion_matrix(tmp_path, monkeypatch):
    # Candidates: (A, Jan), (B, Feb), (C, Jan). (C, Jan) never reaches
    # monthly_reports, exercising the "removed upstream by validation" count.
    ground_truth = pd.DataFrame(
        {
            "facility_id": ["A", "A", "B", "B", "C"],
            "report_month": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-01-01", "2024-02-01", "2024-01-01"]
            ),
            "is_anomaly_candidate": [True, False, False, True, True],
        }
    )
    gt_path = tmp_path / "ground_truth_anomalies.csv"
    ground_truth.to_csv(gt_path, index=False)
    monkeypatch.setattr(anomaly_module, "GROUND_TRUTH_PATH", str(gt_path))

    # (A, Jan): candidate, flagged      -> TP
    # (A, Feb): not candidate, clear    -> TN
    # (B, Jan): not candidate, flagged  -> FP
    # (B, Feb): candidate, missed       -> FN
    scored = pd.DataFrame(
        {
            "facility_id": ["A", "A", "B", "B"],
            "report_month": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-01-01", "2024-02-01"]
            ),
            "is_anomaly": [True, False, True, False],
        }
    )

    metrics = evaluate_against_ground_truth(scored)

    assert metrics["total_candidates_injected"] == 3
    assert metrics["candidates_removed_upstream_by_validation"] == 1
    assert metrics["candidates_present_in_monthly_reports"] == 2
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_negatives"] == 1
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["false_positive_rate"] == pytest.approx(0.5)
