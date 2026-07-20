"""`register` DAG task: promotes a specific MLflow run's logged model to the
Model Registry as a new version.

Kept as a separate task from `train` (models/anomaly.py, models/forecast.py)
per the brief's DAG shape: training produces a candidate model logged to a
run; registering is the deliberate promotion of that exact run's artifact to
the registry that models/registry.py's load-latest path (used by both the
`score` task and api/main.py) will pick up next.

A real production version of this task would gate promotion on the candidate
beating the currently-registered version's metrics - skipped here since nothing
in this repo tunes or competes models against each other, and adding a gate
with no real decision behind it would just be decoration.
"""

from __future__ import annotations

import argparse
import json

import mlflow

from models.mlflow_utils import configure_mlflow


def register_run_model(run_id: str, model_name: str) -> str:
    result = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=model_name)
    return result.version


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a trained run's model")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-name", required=True)
    args = parser.parse_args()

    configure_mlflow()
    version = register_run_model(args.run_id, args.model_name)
    print(json.dumps({"model_name": args.model_name, "run_id": args.run_id, "version": version}))


if __name__ == "__main__":
    main()
