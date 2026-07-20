"""Semantic data-quality checks for a batch of raw monthly reports.

Structural checks (not-null, uniqueness, FK) are enforced by the warehouse
schema itself (see warehouse/models.py). This module handles the checks that
need judgment: referential integrity against the *current* facility roster,
duplicate submissions, indicator range violations, statistical outliers, and
reporting completeness.

Pure functions operate on DataFrames so they're testable without a database;
validation/run.py wires them to the warehouse.
"""

from __future__ import annotations

import pandas as pd

RANGE_RULES = {
    "patients_tested": (0, None),
    "suppression_pct": (0, 100),
    "drug_stock_level": (0, 100),
    "reporting_delay_days": (0, None),
}

# robust z-score threshold (using median/MAD) for flagging statistical outliers
# within a facility's own history in the batch. Deliberately conservative - this
# is a deterministic guardrail, distinct from the IsolationForest model trained
# later, so it should only catch clear-cut spikes.
OUTLIER_Z_THRESHOLD = 5.0
OUTLIER_FIELDS = ["patients_tested", "suppression_pct", "drug_stock_level"]


def check_referential_integrity(df: pd.DataFrame, known_facility_ids: set[str]) -> pd.Series:
    """Returns a boolean Series, True where facility_id is a known facility."""
    return df["facility_id"].isin(known_facility_ids)


def check_duplicates(df: pd.DataFrame) -> pd.Series:
    """Returns a boolean Series, True where the (facility_id, report_month) key
    appears more than once in the batch. All rows sharing a duplicated key are
    quarantined together - conflicting duplicates aren't auto-resolved, since
    guessing which value is correct is worse than surfacing both for review."""
    key = list(zip(df["facility_id"], df["report_month"]))
    counts = pd.Series(key).value_counts()
    dup_keys = set(counts[counts > 1].index)
    return pd.Series(key, index=df.index).isin(dup_keys)


def check_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a DataFrame of boolean columns, one per rule, True = violation."""
    violations = pd.DataFrame(index=df.index)
    for field, (low, high) in RANGE_RULES.items():
        col = df[field]
        bad = pd.Series(False, index=df.index)
        if low is not None:
            bad |= col < low
        if high is not None:
            bad |= col > high
        violations[f"range_{field}"] = bad
    return violations


def check_statistical_outliers(df: pd.DataFrame) -> pd.Series:
    """Robust z-score (median/MAD) per facility, per field. True = outlier on
    at least one monitored field. MAD=0 (e.g. a facility with too few points,
    or genuinely constant values) is treated as "not enough signal" rather
    than flagging everything, to avoid false positives on short histories."""
    is_outlier = pd.Series(False, index=df.index)
    for field in OUTLIER_FIELDS:
        grouped = df.groupby("facility_id")[field]
        median = grouped.transform("median")
        mad = grouped.transform(lambda s: (s - s.median()).abs().median())
        robust_z = (df[field] - median).abs() / (mad.replace(0, float("nan")) * 1.4826)
        is_outlier |= (robust_z > OUTLIER_Z_THRESHOLD).fillna(False)
    return is_outlier


def check_completeness(df: pd.DataFrame, known_facility_ids: set[str]) -> pd.DataFrame:
    """Facility-months present in the batch's overall month range but absent
    for a specific facility. Not a row-level rejection (there's no row to
    quarantine) - reported separately for the monitoring/completeness section."""
    expected_months = sorted(df["report_month"].unique())
    present = df.groupby("facility_id")["report_month"].apply(set)

    missing_rows = []
    for facility_id in sorted(known_facility_ids):
        facility_months = present.get(facility_id, set())
        for month in expected_months:
            if month not in facility_months:
                missing_rows.append({"facility_id": facility_id, "report_month": month})
    return pd.DataFrame(missing_rows, columns=["facility_id", "report_month"])


def classify_batch(
    df: pd.DataFrame, known_facility_ids: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Runs all row-level checks and splits the batch into (clean, quarantined,
    completeness_gaps). Each quarantined row gets exactly one `quarantine_reason`
    - checks are evaluated in priority order (orphan > duplicate > range >
    statistical outlier) and the first violation wins, keeping the reason
    unambiguous even when a row fails more than one check."""
    df = df.reset_index(drop=True)

    referential_ok = check_referential_integrity(df, known_facility_ids)
    is_duplicate = check_duplicates(df)
    range_violations = check_ranges(df)
    is_statistical_outlier = check_statistical_outliers(df)

    reason = pd.Series(pd.NA, index=df.index, dtype="object")
    reason = reason.mask(~referential_ok, "unknown_facility_id")
    reason = reason.mask(reason.isna() & is_duplicate, "duplicate_facility_month")
    for col in range_violations.columns:
        reason = reason.mask(reason.isna() & range_violations[col], col)
    reason = reason.mask(reason.isna() & is_statistical_outlier, "statistical_outlier")

    quarantined_mask = reason.notna()
    quarantined = df.loc[quarantined_mask].copy()
    quarantined["quarantine_reason"] = reason.loc[quarantined_mask]
    clean = df.loc[~quarantined_mask].copy()

    completeness_gaps = check_completeness(df, known_facility_ids)

    return clean, quarantined, completeness_gaps


def build_report(
    batch_size: int,
    clean: pd.DataFrame,
    quarantined: pd.DataFrame,
    completeness_gaps: pd.DataFrame,
    run_id: str,
    dag_logical_date: str,
) -> dict:
    reason_counts = (
        quarantined["quarantine_reason"].value_counts().to_dict() if len(quarantined) else {}
    )
    quarantine_rate = len(quarantined) / batch_size if batch_size else 0.0

    return {
        "run_id": run_id,
        "dag_logical_date": dag_logical_date,
        "batch_size": batch_size,
        "clean_count": len(clean),
        "quarantined_count": len(quarantined),
        "quarantine_rate": round(quarantine_rate, 4),
        "quarantine_reasons": reason_counts,
        "completeness_gap_count": len(completeness_gaps),
        "completeness_gaps_sample": completeness_gaps.head(20).astype(str).to_dict(orient="records"),
        # threshold a production alert would fire on - see docs/monitoring.md
        "alert_quarantine_rate_exceeded": quarantine_rate > 0.10,
    }
