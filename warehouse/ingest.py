"""Loads the data_gen CSV outputs into the warehouse.

This simulates the `ingest` DAG task: facilities are upserted (idempotent on
facility_id). The reports CSV is a full-history snapshot (not an incremental
per-period extract), so the raw landing zone is a full-refresh table: each
ingest replaces its entire contents with the latest load, tagged with
run_id/dag_logical_date for lineage. Loading the same snapshot on every run
without this refresh would otherwise leave every prior run's full copy of the
raw rows sitting in the table forever. No cleaning happens here - that's the
validate task's job (see validation/checks.py).
"""

from __future__ import annotations

import argparse
import uuid
from datetime import date

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

from warehouse.db import get_engine
from warehouse.models import Facility, RawMonthlyReport


def load_facilities(engine, facilities_df: pd.DataFrame) -> int:
    rows = facilities_df.to_dict(orient="records")
    if not rows:
        return 0
    stmt = insert(Facility).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["facility_id"],
        set_={
            "facility_name": stmt.excluded.facility_name,
            "region": stmt.excluded.region,
            "ownership_type": stmt.excluded.ownership_type,
            "baseline_patient_volume": stmt.excluded.baseline_patient_volume,
            "opened_date": stmt.excluded.opened_date,
        },
    )
    with engine.begin() as conn:
        conn.execute(stmt)
    return len(rows)


def load_raw_reports(
    engine,
    raw_df: pd.DataFrame,
    run_id: str,
    dag_logical_date: date,
    source_file: str,
) -> int:
    df = raw_df.copy()
    df["run_id"] = run_id
    df["dag_logical_date"] = dag_logical_date
    df["source_file"] = source_file
    rows = df.to_dict(orient="records")
    if not rows:
        return 0
    with engine.begin() as conn:
        conn.execute(RawMonthlyReport.__table__.insert(), rows)
    return len(rows)


def clear_raw_landing_zone(engine) -> None:
    """Full-refresh for the raw landing zone: the reports CSV is always the
    complete history snapshot, so every ingest clears *all* prior rows (not
    just this run_id's) before loading, rather than accumulating a redundant
    full copy of the same ~1,200 rows on top of every previous run's."""
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM raw_monthly_reports"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest data_gen CSVs into the warehouse")
    parser.add_argument("--facilities-csv", default="data_gen/output/facilities.csv")
    parser.add_argument("--reports-csv", default="data_gen/output/raw_monthly_reports.csv")
    parser.add_argument("--dag-logical-date", default=date.today().isoformat())
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    run_id = args.run_id or str(uuid.uuid4())
    logical_date = date.fromisoformat(args.dag_logical_date)

    engine = get_engine()

    facilities_df = pd.read_csv(args.facilities_csv, parse_dates=["opened_date"])
    facilities_df["opened_date"] = facilities_df["opened_date"].dt.date
    n_fac = load_facilities(engine, facilities_df)

    raw_df = pd.read_csv(args.reports_csv, parse_dates=["report_month"])
    raw_df["report_month"] = raw_df["report_month"].dt.date

    clear_raw_landing_zone(engine)
    n_raw = load_raw_reports(engine, raw_df, run_id, logical_date, args.reports_csv)

    print(f"run_id={run_id}")
    print(f"facilities upserted: {n_fac}")
    print(f"raw reports loaded:  {n_raw}")


if __name__ == "__main__":
    main()
