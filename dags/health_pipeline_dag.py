"""health_pipeline DAG: ingest -> validate -> train -> register -> score -> publish.

Every task shells out to a dedicated Python virtualenv
(/opt/airflow/venvs/project-env, built in infra/Dockerfile.airflow) instead of
importing this project's packages into Airflow's own interpreter. Airflow 2.x
pins SQLAlchemy 1.4 internally; this project's warehouse layer needs
SQLAlchemy 2.0+ for its ORM usage (upsert() etc.). Installing both into one
environment would silently break Airflow's own ORM, so the PythonOperator
callables below only ever use stdlib (subprocess, json) - never this
project's modules directly.

Airflow's own {{ run_id }} is threaded through every task as *this project's*
run_id too, so a single value doubles as our warehouse's lineage/idempotency
key (see project-brief.md). Re-running (clearing) any task within a dag_run
reuses that same run_id, so ingest/validate/score upserts land in place
instead of duplicating.

train/register are split into two tasks (not one) so a training run that
produces a candidate model is a distinct, auditable step from promoting that
candidate into the registry - see models/register.py's docstring. In a more
mature setup train/register might run on a slower cadence than
ingest/validate/score via a separate DAG; kept as one DAG here since splitting
it would need a cross-DAG sensor for no real benefit at this scale.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import task
from airflow.models.dag import DAG
from airflow.utils.task_group import TaskGroup

PROJECT_DIR = "/opt/airflow/project"
PROJECT_PYTHON = "/opt/airflow/venvs/project-env/bin/python"

default_args = {
    "owner": "health-pipeline",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "depends_on_past": False,
}


def _trailing_json_block(stdout: str) -> str:
    """mlflow prints its own 'View run'/'View experiment' banners to stdout
    ahead of each module's own pretty-printed JSON result block, so the
    result isn't the whole of stdout - scan from the end for the last
    balanced {...} block instead."""
    depth = 0
    end = None
    for i in range(len(stdout) - 1, -1, -1):
        if stdout[i] == "}":
            if depth == 0:
                end = i + 1
            depth += 1
        elif stdout[i] == "{":
            depth -= 1
            if depth == 0:
                return stdout[i:end]
    return stdout


def _run_module(module: str, *args: str) -> str:
    """Runs `{PROJECT_PYTHON} -m {module} {args}` in the isolated venv and
    returns its trailing JSON result block, so callers that need structured
    output can json.loads() the return value."""
    result = subprocess.run(
        [PROJECT_PYTHON, "-m", module, *args],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"{module} failed with exit code {result.returncode}")
    return _trailing_json_block(result.stdout)


with DAG(
    dag_id="health_pipeline",
    description="ingest -> validate -> train -> register -> score -> publish",
    default_args=default_args,
    schedule="@monthly",
    start_date=datetime(2024, 1, 1),
    # Every task here is idempotent/upsert-based, so backfill (catchup=True)
    # would be safe - left off so the demo doesn't eagerly run ~24 months of
    # backfill the first time the scheduler sees this DAG.
    catchup=False,
    max_active_runs=1,
    tags=["health-pipeline"],
) as dag:

    # Params are named pipeline_run_id / logical_date_str, not run_id / ds -
    # those two are Airflow's own reserved TaskFlow context-injection names,
    # and declaring a parameter with those names makes Airflow try to
    # auto-inject them with a default, which breaks signature ordering
    # against the other non-default parameters.
    @task
    def ingest(pipeline_run_id: str, logical_date_str: str) -> None:
        _run_module(
            "warehouse.ingest", "--run-id", pipeline_run_id, "--dag-logical-date", logical_date_str
        )

    @task
    def validate(pipeline_run_id: str, logical_date_str: str) -> None:
        _run_module(
            "validation.run", "--run-id", pipeline_run_id, "--dag-logical-date", logical_date_str
        )

    with TaskGroup(group_id="train") as train_group:

        @task(task_id="train_anomaly")
        def train_anomaly() -> str:
            return json.loads(_run_module("models.anomaly"))["mlflow_run_id"]

        @task(task_id="train_forecast")
        def train_forecast() -> str:
            return json.loads(_run_module("models.forecast"))["mlflow_run_id"]

        anomaly_run_id = train_anomaly()
        forecast_run_id = train_forecast()

    with TaskGroup(group_id="register") as register_group:

        @task(task_id="register_anomaly")
        def register_anomaly(train_run_id: str) -> str:
            out = _run_module(
                "models.register",
                "--run-id",
                train_run_id,
                "--model-name",
                "health-anomaly-detector",
            )
            return json.loads(out)["version"]

        @task(task_id="register_forecast")
        def register_forecast(train_run_id: str) -> str:
            out = _run_module(
                "models.register",
                "--run-id",
                train_run_id,
                "--model-name",
                "health-suppression-forecaster",
            )
            return json.loads(out)["version"]

        register_anomaly(anomaly_run_id)
        register_forecast(forecast_run_id)

    @task
    def score(pipeline_run_id: str, logical_date_str: str) -> str:
        return _run_module(
            "models.score", "--run-id", pipeline_run_id, "--dag-logical-date", logical_date_str
        )

    @task
    def publish(pipeline_run_id: str, score_summary_json: str) -> None:
        """Consolidates this run's validation report + scoring summary into
        reports/latest_*.{json,html} - fixed paths a future dashboard can
        poll without needing to know a specific run_id."""
        reports_dir = Path(PROJECT_DIR) / "reports"
        reports_dir.mkdir(exist_ok=True)

        validation_report_path = (
            Path(PROJECT_DIR) / "validation" / "output" / f"{pipeline_run_id}_report.json"
        )
        validation_report = json.loads(validation_report_path.read_text())

        summary = {
            "run_id": pipeline_run_id,
            "validation": validation_report,
            "scoring": json.loads(score_summary_json),
        }
        (reports_dir / "latest_summary.json").write_text(json.dumps(summary, indent=2))

        html_src = Path(PROJECT_DIR) / "validation" / "output" / f"{pipeline_run_id}_report.html"
        (reports_dir / "latest_validation_report.html").write_text(html_src.read_text())
        print(json.dumps(summary, indent=2))

    ingest_task = ingest(pipeline_run_id="{{ run_id }}", logical_date_str="{{ ds }}")
    validate_task = validate(pipeline_run_id="{{ run_id }}", logical_date_str="{{ ds }}")
    score_task = score(pipeline_run_id="{{ run_id }}", logical_date_str="{{ ds }}")
    publish(pipeline_run_id="{{ run_id }}", score_summary_json=score_task)

    ingest_task >> validate_task >> train_group
    register_group >> score_task
