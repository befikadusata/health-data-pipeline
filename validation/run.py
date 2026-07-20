"""Wires validation/checks.py to the warehouse - the `validate` DAG task.

Reads a batch from raw_monthly_reports (by run_id), classifies every row,
upserts clean rows into monthly_reports (idempotent on facility_id +
report_month), inserts quarantined rows into quarantined_reports, and writes
a structured report (JSON + HTML) describing what happened.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date

import pandas as pd
from sqlalchemy import ARRAY, Date, String, bindparam, text
from sqlalchemy.dialects.postgresql import insert

from validation.checks import build_report, classify_batch
from warehouse.db import get_engine
from warehouse.models import MonthlyReport, QuarantinedReport

REPORT_DIR = "validation/output"

MONTHLY_REPORT_COLUMNS = [
    "facility_id",
    "report_month",
    "patients_tested",
    "suppression_pct",
    "drug_stock_level",
    "reporting_delay_days",
    "run_id",
    "dag_logical_date",
]

QUARANTINE_COLUMNS = [
    "facility_id",
    "report_month",
    "patients_tested",
    "suppression_pct",
    "drug_stock_level",
    "reporting_delay_days",
    "quarantine_reason",
    "run_id",
    "dag_logical_date",
]


def load_batch(engine, run_id: str) -> pd.DataFrame:
    return pd.read_sql(
        text("SELECT * FROM raw_monthly_reports WHERE run_id = :run_id"),
        engine,
        params={"run_id": run_id},
    )


def load_known_facility_ids(engine) -> set[str]:
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT facility_id FROM facilities")).fetchall()
    return {r[0] for r in rows}


def upsert_clean(engine, run_id: str, clean: pd.DataFrame, quarantined: pd.DataFrame) -> int:
    """Idempotent upsert on (facility_id, report_month). Also retracts any key
    that this run previously promoted to monthly_reports but now quarantines -
    otherwise re-validating with stricter rules couldn't undo an earlier
    (wrong) clean classification, breaking the re-run-safety invariant."""
    with engine.begin() as conn:
        if not quarantined.empty:
            conn.execute(
                text(
                    "DELETE FROM monthly_reports mr "
                    "USING unnest(:facility_ids, :report_months) AS t(facility_id, report_month) "
                    "WHERE mr.run_id = :run_id "
                    "AND mr.facility_id = t.facility_id AND mr.report_month = t.report_month"
                ).bindparams(
                    bindparam("facility_ids", type_=ARRAY(String)),
                    bindparam("report_months", type_=ARRAY(Date)),
                ),
                {
                    "run_id": run_id,
                    "facility_ids": quarantined["facility_id"].tolist(),
                    "report_months": quarantined["report_month"].tolist(),
                },
            )

        if clean.empty:
            return 0

        rows = clean[MONTHLY_REPORT_COLUMNS].to_dict(orient="records")
        update_cols = [
            c for c in MONTHLY_REPORT_COLUMNS if c not in ("facility_id", "report_month")
        ]
        stmt = insert(MonthlyReport).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["facility_id", "report_month"],
            set_={c: getattr(stmt.excluded, c) for c in update_cols},
        )
        conn.execute(stmt)
    return len(rows) if not clean.empty else 0


def replace_quarantine(engine, run_id: str, quarantined: pd.DataFrame) -> int:
    """Idempotency for quarantine: re-validating the same run_id replaces its
    quarantined rows rather than appending duplicates on top."""
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM quarantined_reports WHERE run_id = :run_id"), {"run_id": run_id}
        )
        if quarantined.empty:
            return 0
        rows = quarantined[QUARANTINE_COLUMNS].to_dict(orient="records")
        conn.execute(QuarantinedReport.__table__.insert(), rows)
    return len(quarantined)


def render_html(report: dict) -> str:
    reasons_rows = "".join(
        f"<tr><td>{reason}</td><td>{count}</td></tr>"
        for reason, count in report["quarantine_reasons"].items()
    )
    gaps_rows = "".join(
        f"<tr><td>{g['facility_id']}</td><td>{g['report_month']}</td></tr>"
        for g in report["completeness_gaps_sample"]
    )
    alert_banner = (
        '<p style="color:#b00"><strong>ALERT: quarantine rate exceeds 10% threshold</strong></p>'
        if report["alert_quarantine_rate_exceeded"]
        else ""
    )
    return f"""<!doctype html>
<html><head><title>Validation report - {report['run_id']}</title></head>
<body style="font-family: sans-serif; max-width: 720px; margin: 2rem auto;">
<h1>Data quality validation report</h1>
<p>Run: <code>{report['run_id']}</code> &middot; Logical date: {report['dag_logical_date']}</p>
{alert_banner}
<ul>
<li>Batch size: {report['batch_size']}</li>
<li>Clean: {report['clean_count']}</li>
<li>Quarantined: {report['quarantined_count']} ({report['quarantine_rate']:.1%})</li>
<li>Completeness gaps (facility-months with no report at all): {report['completeness_gap_count']}</li>
</ul>
<h2>Quarantine reasons</h2>
<table border="1" cellpadding="4"><tr><th>Reason</th><th>Count</th></tr>{reasons_rows}</table>
<h2>Completeness gaps (sample)</h2>
<table border="1" cellpadding="4"><tr><th>Facility</th><th>Report month</th></tr>{gaps_rows}</table>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semantic validation on a raw batch")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dag-logical-date", default=date.today().isoformat())
    args = parser.parse_args()

    engine = get_engine()
    batch = load_batch(engine, args.run_id)
    if batch.empty:
        raise SystemExit(f"No raw_monthly_reports rows found for run_id={args.run_id}")

    known_facility_ids = load_known_facility_ids(engine)
    clean, quarantined, completeness_gaps = classify_batch(batch, known_facility_ids)

    n_clean = upsert_clean(engine, args.run_id, clean, quarantined)
    n_quarantined = replace_quarantine(engine, args.run_id, quarantined)

    report = build_report(
        batch_size=len(batch),
        clean=clean,
        quarantined=quarantined,
        completeness_gaps=completeness_gaps,
        run_id=args.run_id,
        dag_logical_date=args.dag_logical_date,
    )

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(f"{REPORT_DIR}/{args.run_id}_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(f"{REPORT_DIR}/{args.run_id}_report.html", "w") as f:
        f.write(render_html(report))

    print(f"clean upserted:      {n_clean}")
    print(f"quarantined:         {n_quarantined}")
    print(f"completeness gaps:   {report['completeness_gap_count']}")
    print(f"quarantine rate:     {report['quarantine_rate']:.1%}")
    if report["alert_quarantine_rate_exceeded"]:
        print("ALERT: quarantine rate exceeds 10% threshold")


if __name__ == "__main__":
    main()
