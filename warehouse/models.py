"""SQLAlchemy ORM schema for the health-program warehouse.

Layout (see project-brief.md for the rationale):
  facilities          - dimension table
  raw_monthly_reports - append-only ingestion landing zone; no uniqueness
                        constraint, so injected duplicates land as literal
                        duplicate rows for the validate task to catch
  quarantined_reports - rows the validate task rejected, with a reason
  monthly_reports     - clean fact table; structural constraints enforced here
                        (not-null, range checks, FK, unique on natural key)
  scored_reports      - model output, keyed the same way as monthly_reports so
                        it upserts idempotently
"""

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    Numeric,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Facility(Base):
    __tablename__ = "facilities"

    facility_id = Column(String, primary_key=True)
    facility_name = Column(String, nullable=False)
    region = Column(String, nullable=False)
    ownership_type = Column(String, nullable=False)
    baseline_patient_volume = Column(Integer, nullable=False)
    opened_date = Column(Date, nullable=False)


class RawMonthlyReport(Base):
    """Raw ingestion landing zone. facility_id is intentionally not FK-enforced
    here - an unknown facility_id is exactly the kind of issue the validate
    task should catch and quarantine, not something the DB should reject on
    ingest."""

    __tablename__ = "raw_monthly_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    facility_id = Column(String, nullable=False)
    report_month = Column(Date, nullable=False)
    patients_tested = Column(Integer)
    suppression_pct = Column(Numeric)
    drug_stock_level = Column(Numeric)
    reporting_delay_days = Column(Integer)
    source_file = Column(String)
    run_id = Column(String)
    dag_logical_date = Column(Date)
    ingested_at = Column(DateTime, server_default=func.now())


class QuarantinedReport(Base):
    __tablename__ = "quarantined_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    facility_id = Column(String, nullable=False)
    report_month = Column(Date, nullable=False)
    patients_tested = Column(Integer)
    suppression_pct = Column(Numeric)
    drug_stock_level = Column(Numeric)
    reporting_delay_days = Column(Integer)
    quarantine_reason = Column(String, nullable=False)
    run_id = Column(String)
    dag_logical_date = Column(Date)
    quarantined_at = Column(DateTime, server_default=func.now())


class MonthlyReport(Base):
    """Clean fact table. Primary key on (facility_id, report_month) is what
    makes ingestion idempotent: re-running a period upserts in place."""

    __tablename__ = "monthly_reports"

    facility_id = Column(String, ForeignKey("facilities.facility_id"), primary_key=True)
    report_month = Column(Date, primary_key=True)
    patients_tested = Column(Integer, nullable=False)
    suppression_pct = Column(Numeric, nullable=False)
    drug_stock_level = Column(Numeric, nullable=False)
    reporting_delay_days = Column(Integer, nullable=False)
    run_id = Column(String)
    dag_logical_date = Column(Date)
    ingested_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        CheckConstraint("patients_tested >= 0", name="ck_monthly_reports_patients_nonneg"),
        CheckConstraint(
            "suppression_pct >= 0 AND suppression_pct <= 100",
            name="ck_monthly_reports_suppression_range",
        ),
        CheckConstraint("drug_stock_level >= 0", name="ck_monthly_reports_stock_nonneg"),
        CheckConstraint("reporting_delay_days >= 0", name="ck_monthly_reports_delay_nonneg"),
    )


class ScoredReport(Base):
    """Model output. run_id + dag_logical_date give basic lineage back to the
    MLflow run and DAG execution that produced each row. Same natural key as
    monthly_reports, so scoring is upsert-idempotent too."""

    __tablename__ = "scored_reports"

    facility_id = Column(String, primary_key=True)
    report_month = Column(Date, primary_key=True)
    run_id = Column(String, nullable=False)
    dag_logical_date = Column(Date)
    is_anomaly = Column(Boolean, nullable=False)
    anomaly_score = Column(Numeric, nullable=False)
    anomaly_reasons = Column(JSONB)
    forecast_next_quarter_suppression_pct = Column(Numeric)
    model_version_anomaly = Column(String)
    model_version_forecast = Column(String)
    scored_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        ForeignKeyConstraint(
            ["facility_id", "report_month"],
            ["monthly_reports.facility_id", "monthly_reports.report_month"],
        ),
    )
