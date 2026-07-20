"""Database connection helpers, shared by Alembic, the DAG tasks, and the API."""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DEFAULT_DATABASE_URL = "postgresql+psycopg://health:health@localhost:5433/health_pipeline"


def get_database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def get_engine():
    return create_engine(get_database_url(), future=True)


SessionLocal = sessionmaker(bind=get_engine(), future=True)
