import pandas as pd
import pytest

from validation.checks import (
    check_completeness,
    check_duplicates,
    check_ranges,
    check_referential_integrity,
    classify_batch,
)


def _row(facility_id="CL-001", report_month="2024-01-01", **overrides):
    base = {
        "facility_id": facility_id,
        "report_month": pd.Timestamp(report_month),
        "patients_tested": 100,
        "suppression_pct": 90.0,
        "drug_stock_level": 50.0,
        "reporting_delay_days": 5,
    }
    base.update(overrides)
    return base


def test_check_referential_integrity_flags_unknown_facility():
    df = pd.DataFrame([_row(facility_id="CL-001"), _row(facility_id="CL-999")])
    ok = check_referential_integrity(df, known_facility_ids={"CL-001"})
    assert ok.tolist() == [True, False]


def test_check_duplicates_flags_both_rows_of_a_conflicting_pair():
    df = pd.DataFrame(
        [
            _row(report_month="2024-01-01", patients_tested=100),
            _row(report_month="2024-01-01", patients_tested=200),
            _row(report_month="2024-02-01"),
        ]
    )
    dup = check_duplicates(df)
    assert dup.tolist() == [True, True, False]


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("suppression_pct", -1.0, True),
        ("suppression_pct", 101.0, True),
        ("suppression_pct", 50.0, False),
        ("patients_tested", -5, True),
        ("reporting_delay_days", -1, True),
    ],
)
def test_check_ranges_flags_out_of_range_values(field, value, expected):
    df = pd.DataFrame([_row(**{field: value})])
    violations = check_ranges(df)
    assert bool(violations[f"range_{field}"].iloc[0]) is expected


def test_check_completeness_reports_missing_facility_months():
    known = {"CL-001", "CL-002"}
    df = pd.DataFrame(
        [
            _row(facility_id="CL-001", report_month="2024-01-01"),
            _row(facility_id="CL-001", report_month="2024-02-01"),
            _row(facility_id="CL-002", report_month="2024-01-01"),
        ]
    )
    gaps = check_completeness(df, known)
    assert gaps.to_dict("records") == [
        {"facility_id": "CL-002", "report_month": pd.Timestamp("2024-02-01")}
    ]


def test_classify_batch_priority_orphan_beats_range_violation():
    """A row that's both an unknown facility and out-of-range should carry the
    orphan reason, not the range one - classify_batch's documented priority
    order (orphan > duplicate > range > outlier)."""
    df = pd.DataFrame([_row(facility_id="CL-999", suppression_pct=-1.0)])
    _, quarantined, _ = classify_batch(df, known_facility_ids={"CL-001"})
    assert quarantined["quarantine_reason"].tolist() == ["unknown_facility_id"]


def test_classify_batch_clean_row_passes_through():
    df = pd.DataFrame([_row()])
    clean, quarantined, _ = classify_batch(df, known_facility_ids={"CL-001"})
    assert len(clean) == 1
    assert quarantined.empty
