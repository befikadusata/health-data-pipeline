"""Single MLflow Model Registry load path, shared by the `score` DAG task
(models/score.py) and the serving API (api/main.py) - per the brief, both
must resolve "the current registered model" the exact same way.

Uses search_model_versions (returns every version regardless of stage) and
picks the highest version number, rather than the deprecated stage-based
get_latest_versions - this repo never assigns stages/aliases, so "latest
version number" is the only meaningful notion of "current" here.
"""

from __future__ import annotations

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


def load_latest_sklearn_model(name: str):
    """Returns (model, version) - version as the registry's string version
    number, for stamping into scored_reports / API responses as
    model_version_*."""
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise RuntimeError(f"No registered versions found for model '{name}'")
    latest = max(versions, key=lambda v: int(v.version))
    model = mlflow.sklearn.load_model(f"models:/{name}/{latest.version}")
    return model, latest.version
