"""Shared MLflow configuration.

Tracking store is SQLite-backed (not the plain file store) because the model
registry requires a database backend - this is the smallest setup that
supports register_model() locally.
"""

import os

import mlflow

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "health-pipeline"


def configure_mlflow() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
