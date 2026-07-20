"""Synthetic health-program data generator.

Produces three artifacts under data_gen/output/:
  - facilities.csv            : dimension table, ~N_FACILITIES clinics
  - raw_monthly_reports.csv   : the "raw extract" ingestion reads — includes
                                injected duplicates and outliers, excludes
                                injected missing months (they're simply absent,
                                as a real non-reporting clinic would be)
  - ground_truth_anomalies.csv: ledger of every injected issue, used later to
                                evaluate the anomaly detector and to sanity-check
                                the validation suite's completeness checks.

Everything is driven by data_gen.config so the generator, the eval script, and
the docs stay in sync. Re-running with the same SEED reproduces byte-identical
output.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from data_gen.config import (
    DUPLICATE_RATE,
    MISSING_MONTH_RATE,
    N_FACILITIES,
    N_MONTHS,
    NORMAL_DELAY_DAYS,
    OUTLIER_RATE,
    OUTPUT_DIR,
    REGIONS,
    SEED,
    SEVERE_DELAY_DAYS,
    SEVERE_DELAY_RATE,
    START_MONTH,
)


def month_range(start: date, n: int) -> list[date]:
    return [start + relativedelta(months=i) for i in range(n)]


def generate_facilities(rng: np.random.Generator) -> pd.DataFrame:
    facility_ids = [f"CL-{i:03d}" for i in range(1, N_FACILITIES + 1)]
    regions = rng.choice(REGIONS, size=N_FACILITIES)
    ownership = rng.choice(["public", "ngo", "private"], size=N_FACILITIES, p=[0.6, 0.3, 0.1])
    # baseline patient volume per clinic - lognormal so a few clinics are much larger
    baseline_size = rng.lognormal(mean=4.5, sigma=0.6, size=N_FACILITIES).round().astype(int)
    baseline_size = np.clip(baseline_size, 20, 2000)
    opened_offsets = rng.integers(365, 365 * 12, size=N_FACILITIES)  # days before START_MONTH
    opened_dates = [START_MONTH - relativedelta(days=int(d)) for d in opened_offsets]

    return pd.DataFrame(
        {
            "facility_id": facility_ids,
            "facility_name": [f"{r} Community Health Center {i+1}" for i, r in enumerate(regions)],
            "region": regions,
            "ownership_type": ownership,
            "baseline_patient_volume": baseline_size,
            "opened_date": opened_dates,
        }
    )


@dataclass
class CleanReports:
    df: pd.DataFrame


def generate_clean_reports(facilities: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    months = month_range(START_MONTH, N_MONTHS)
    rows = []

    for _, fac in facilities.iterrows():
        baseline = fac["baseline_patient_volume"]
        # persistent per-facility random effects so clinics have stable "personality"
        suppression_baseline = rng.normal(78, 6)
        stock_level = rng.normal(85, 8)  # start point for the stock random walk

        for t, month in enumerate(months):
            seasonal = 1.0 + 0.12 * np.sin(2 * np.pi * (t % 12) / 12)
            growth = 1.0 + 0.01 * t  # slow programmatic growth over 2 years
            patients_tested = baseline * seasonal * growth * rng.normal(1.0, 0.08)
            patients_tested = max(0, round(patients_tested))

            suppression_trend = 0.15 * t  # slow improvement over time (program effect)
            suppression_pct = suppression_baseline + suppression_trend + rng.normal(0, 3)
            suppression_pct = float(np.clip(suppression_pct, 0, 100))

            # mean-reverting random walk for drug stock, clipped to a valid pct range
            stock_level += rng.normal(0, 6)
            stock_level = 0.9 * stock_level + 0.1 * 85  # mean reversion toward 85
            drug_stock_level = float(np.clip(stock_level, 0, 100))

            reporting_delay_days = int(rng.integers(NORMAL_DELAY_DAYS[0], NORMAL_DELAY_DAYS[1] + 1))

            rows.append(
                {
                    "facility_id": fac["facility_id"],
                    "report_month": month,
                    "patients_tested": patients_tested,
                    "suppression_pct": round(suppression_pct, 1),
                    "drug_stock_level": round(drug_stock_level, 1),
                    "reporting_delay_days": reporting_delay_days,
                }
            )

    return pd.DataFrame(rows)


def inject_issues(
    clean: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (raw_reports, ground_truth_ledger).

    Applied independently, in this order, over the full facility-month grid:
      1. missing_month   - row dropped entirely (candidate pool shrinks after this)
      2. duplicate       - a second, conflicting row appended for that facility-month
      3. outlier_spike   - one indicator overwritten with an extreme value
      4. severe_delay    - reporting_delay_days overwritten with a large value

    outlier_spike and severe_delay are drawn from rows not already marked
    duplicate/missing, so each injected issue type stays cleanly separable -
    useful both for validation-suite testing and for anomaly-detector eval.
    """
    df = clean.copy()
    n = len(df)
    ledger_rows = []

    missing_mask = rng.random(n) < MISSING_MONTH_RATE
    for _, row in df.loc[missing_mask].iterrows():
        ledger_rows.append(
            _ledger_row(row, "missing_month", is_anomaly_candidate=False, detail="row absent from raw extract")
        )

    remaining = df.loc[~missing_mask].copy()

    dup_mask = rng.random(len(remaining)) < DUPLICATE_RATE
    dup_source = remaining.loc[dup_mask].copy()
    duplicate_rows = []
    for _, row in dup_source.iterrows():
        perturbed = row.copy()
        perturbed["patients_tested"] = max(0, round(row["patients_tested"] * rng.normal(1.0, 0.2)))
        perturbed["suppression_pct"] = float(
            np.clip(row["suppression_pct"] + rng.normal(0, 8), 0, 100)
        )
        duplicate_rows.append(perturbed)
        ledger_rows.append(
            _ledger_row(row, "duplicate_facility_id", is_anomaly_candidate=False, detail="original of a conflicting duplicate pair")
        )
        ledger_rows.append(
            _ledger_row(perturbed, "duplicate_facility_id", is_anomaly_candidate=False, detail="conflicting duplicate submission")
        )

    # rows chosen for duplication keep their original, unmodified copy in the raw
    # extract too (that's what makes it a duplicate) - they're just excluded from
    # the outlier/delay injection pool below so issue types stay separable.
    dup_originals_unmodified = dup_source
    non_dup_pool = remaining.loc[~dup_mask].copy()

    outlier_mask = rng.random(len(non_dup_pool)) < OUTLIER_RATE
    outlier_idx = non_dup_pool.loc[outlier_mask].index
    fields = ["patients_tested", "suppression_pct", "drug_stock_level"]
    for idx in outlier_idx:
        field = rng.choice(fields)
        if field == "patients_tested":
            non_dup_pool.loc[idx, field] = round(non_dup_pool.loc[idx, field] * rng.choice([6, 8, 10]))
        elif field == "suppression_pct":
            non_dup_pool.loc[idx, field] = rng.choice([1.5, 3.0, 98.5, 99.9])
        else:  # drug_stock_level
            non_dup_pool.loc[idx, field] = rng.choice([0.0, 1.0])
        ledger_rows.append(
            _ledger_row(
                non_dup_pool.loc[idx], "outlier_spike", is_anomaly_candidate=True, detail=f"field={field}"
            )
        )

    delay_pool_mask = ~non_dup_pool.index.isin(outlier_idx)
    delay_candidates = non_dup_pool.loc[delay_pool_mask]
    severe_delay_mask = rng.random(len(delay_candidates)) < SEVERE_DELAY_RATE
    severe_idx = delay_candidates.loc[severe_delay_mask].index
    for idx in severe_idx:
        non_dup_pool.loc[idx, "reporting_delay_days"] = int(
            rng.integers(SEVERE_DELAY_DAYS[0], SEVERE_DELAY_DAYS[1] + 1)
        )
        ledger_rows.append(
            _ledger_row(non_dup_pool.loc[idx], "severe_delay", is_anomaly_candidate=True, detail="reporting_delay_days overwritten")
        )

    raw = pd.concat(
        [non_dup_pool, dup_originals_unmodified, pd.DataFrame(duplicate_rows)], ignore_index=True
    )
    raw = raw.sample(frac=1, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)

    ledger = pd.DataFrame(ledger_rows)
    return raw, ledger


def _ledger_row(row: pd.Series, issue_type: str, is_anomaly_candidate: bool, detail: str) -> dict:
    return {
        "facility_id": row["facility_id"],
        "report_month": row["report_month"],
        "issue_type": issue_type,
        "is_anomaly_candidate": is_anomaly_candidate,
        "detail": detail,
    }


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    facilities = generate_facilities(rng)
    clean_reports = generate_clean_reports(facilities, rng)
    raw_reports, ground_truth = inject_issues(clean_reports, rng)

    facilities.to_csv(f"{OUTPUT_DIR}/facilities.csv", index=False)
    raw_reports.to_csv(f"{OUTPUT_DIR}/raw_monthly_reports.csv", index=False)
    ground_truth.to_csv(f"{OUTPUT_DIR}/ground_truth_anomalies.csv", index=False)

    print(f"facilities:            {len(facilities)}")
    print(f"clean facility-months: {len(clean_reports)}")
    print(f"raw rows (post-inject):{len(raw_reports)}")
    print(f"ground-truth issues:   {len(ground_truth)}")
    print(ground_truth["issue_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
