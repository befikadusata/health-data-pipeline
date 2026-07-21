"""Shared fixtures for DB-backed integration tests.

Integration tests need a real Postgres - they exercise the warehouse's actual
upsert SQL (ON CONFLICT DO UPDATE, retraction deletes), which is exactly what
unit tests with synthetic DataFrames can't verify. They default to the
docker-compose `postgres` service (infra/docker-compose.yml, localhost:5442),
never the manually-run local-dev container on 5433, so they can never touch a
developer's own demo data. If that database isn't reachable, the tests that
depend on it are skipped rather than failing the whole suite.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, text

INTEGRATION_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+psycopg://health:health@localhost:5442/health_pipeline"
)

# Children first, so a plain (non-cascading) reader of this list would still
# be FK-safe; TRUNCATE ... CASCADE below doesn't strictly need the order but
# it documents the dependency chain.
TABLES_IN_FK_ORDER = [
    "scored_reports",
    "quarantined_reports",
    "monthly_reports",
    "raw_monthly_reports",
    "facilities",
]


@pytest.fixture(scope="session")
def pg_engine():
    engine = create_engine(INTEGRATION_DATABASE_URL, future=True)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 - any connection failure means "skip"
        pytest.skip(f"Postgres not reachable at {INTEGRATION_DATABASE_URL}: {exc}")
    yield engine
    engine.dispose()


@pytest.fixture
def clean_db(pg_engine):
    """Truncates every warehouse table before each test for isolation."""
    with pg_engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {', '.join(TABLES_IN_FK_ORDER)} RESTART IDENTITY CASCADE"))
    return pg_engine
