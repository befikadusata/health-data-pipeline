"""Integration tests against a real Postgres for the pipeline's headline
guarantee: every DAG task's write is upsert-idempotent, so re-running
ingest/validate/score for any period never creates duplicate rows and always
reflects the latest run's values. Unit tests exercise the pure classification
logic (validation/checks.py, models/features.py) against synthetic
DataFrames; these hit the actual SQL (see tests/conftest.py for the database
these use and why).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from sqlalchemy import text

from models.score import upsert_scored
from validation.run import upsert_clean
from warehouse.ingest import clear_raw_landing_zone, load_facilities, load_raw_reports


def _facilities_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "facility_id": "CL-TEST-1",
                "facility_name": "Test Clinic 1",
                "region": "Test",
                "ownership_type": "public",
                "baseline_patient_volume": 100,
                "opened_date": date(2020, 1, 1),
            }
        ]
    )


def _raw_reports_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "facility_id": "CL-TEST-1",
                "report_month": date(2024, 1, 1),
                "patients_tested": 100,
                "suppression_pct": 80.0,
                "drug_stock_level": 50.0,
                "reporting_delay_days": 5,
            },
            {
                "facility_id": "CL-TEST-1",
                "report_month": date(2024, 2, 1),
                "patients_tested": 110,
                "suppression_pct": 82.0,
                "drug_stock_level": 55.0,
                "reporting_delay_days": 4,
            },
        ]
    )


def test_ingest_full_refresh_is_idempotent_across_runs(clean_db):
    """warehouse/ingest.py: the reports CSV is a full-history snapshot
    reloaded on every DAG run, not an incremental feed - re-ingesting it
    under a new run_id must replace the raw landing zone, not accumulate a
    second full copy on top of it."""
    engine = clean_db
    load_facilities(engine, _facilities_df())

    clear_raw_landing_zone(engine)
    load_raw_reports(
        engine, _raw_reports_df(), run_id="run-A", dag_logical_date=date(2024, 1, 1),
        source_file="test.csv",
    )
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM raw_monthly_reports")).scalar() == 2

    clear_raw_landing_zone(engine)
    load_raw_reports(
        engine, _raw_reports_df(), run_id="run-B", dag_logical_date=date(2024, 2, 1),
        source_file="test.csv",
    )
    with engine.connect() as conn:
        count = conn.execute(text("SELECT count(*) FROM raw_monthly_reports")).scalar()
        run_ids = [
            r[0] for r in conn.execute(text("SELECT DISTINCT run_id FROM raw_monthly_reports"))
        ]
    assert count == 2
    assert run_ids == ["run-B"]


def test_upsert_scored_is_idempotent_and_updates_in_place(clean_db):
    """models/score.py's upsert_scored: re-scoring the same facility-month
    (a later `score` run) must update the existing row in place, never insert
    a second row for the same natural key."""
    engine = clean_db
    load_facilities(engine, _facilities_df())
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO monthly_reports "
                "(facility_id, report_month, patients_tested, suppression_pct, "
                "drug_stock_level, reporting_delay_days, run_id, dag_logical_date) "
                "VALUES (:facility_id, :report_month, 100, 80.0, 50.0, 5, 'seed', :report_month)"
            ),
            {"facility_id": "CL-TEST-1", "report_month": date(2024, 1, 1)},
        )

    scored_v1 = pd.DataFrame(
        [
            {
                "facility_id": "CL-TEST-1",
                "report_month": date(2024, 1, 1),
                "is_anomaly": False,
                "anomaly_score": 0.1,
                "anomaly_reasons": [],
                "forecast_next_quarter_suppression_pct": 81.0,
            }
        ]
    )
    n = upsert_scored(engine, "run-A", "2024-01-01", scored_v1, "1", "1")
    assert n == 1
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT is_anomaly, anomaly_score, run_id FROM scored_reports")
        ).fetchone()
    assert row.is_anomaly is False
    assert float(row.anomaly_score) == pytest.approx(0.1)
    assert row.run_id == "run-A"

    scored_v2 = scored_v1.copy()
    scored_v2["is_anomaly"] = True
    scored_v2["anomaly_score"] = 3.5
    upsert_scored(engine, "run-B", "2024-02-01", scored_v2, "2", "2")

    with engine.connect() as conn:
        count = conn.execute(text("SELECT count(*) FROM scored_reports")).scalar()
        row = conn.execute(
            text("SELECT is_anomaly, anomaly_score, run_id FROM scored_reports")
        ).fetchone()
    assert count == 1
    assert row.is_anomaly is True
    assert float(row.anomaly_score) == pytest.approx(3.5)
    assert row.run_id == "run-B"


def test_upsert_clean_retracts_previously_clean_rows_now_quarantined(clean_db):
    """validation/run.py's upsert_clean: if a re-validation of the same
    run_id now quarantines a key it previously promoted to monthly_reports,
    that row must be retracted - otherwise stricter re-validation could never
    undo an earlier wrong "clean" classification (see upsert_clean's own
    docstring)."""
    engine = clean_db
    load_facilities(engine, _facilities_df())

    clean_v1 = pd.DataFrame(
        [
            {
                "facility_id": "CL-TEST-1",
                "report_month": date(2024, 1, 1),
                "patients_tested": 100,
                "suppression_pct": 80.0,
                "drug_stock_level": 50.0,
                "reporting_delay_days": 5,
                "run_id": "run-A",
                "dag_logical_date": date(2024, 1, 1),
            }
        ]
    )
    upsert_clean(engine, "run-A", clean_v1, pd.DataFrame())
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM monthly_reports")).scalar() == 1

    quarantined_v2 = pd.DataFrame(
        [
            {
                "facility_id": "CL-TEST-1",
                "report_month": date(2024, 1, 1),
                "patients_tested": 100,
                "suppression_pct": 80.0,
                "drug_stock_level": 50.0,
                "reporting_delay_days": 5,
                "quarantine_reason": "statistical_outlier",
                "run_id": "run-A",
                "dag_logical_date": date(2024, 1, 1),
            }
        ]
    )
    upsert_clean(engine, "run-A", pd.DataFrame(), quarantined_v2)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM monthly_reports")).scalar() == 0
