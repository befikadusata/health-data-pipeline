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
import os
import subprocess
import tempfile
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


def _run_module(module: str, *args: str, json_output: bool = False) -> str:
    """Runs `{PROJECT_PYTHON} -m {module} {args}` in the isolated venv.

    When json_output=True, passes --output-file so the module writes its
    structured result straight to a temp file that's read back directly -
    mlflow (and other libs) print their own banners to stdout too, so
    scraping the result out of stdout would be fragile.
    """
    output_path = None
    cmd = [PROJECT_PYTHON, "-m", module, *args]
    if json_output:
        fd, output_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        cmd += ["--output-file", output_path]

    try:
        result = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError(f"{module} failed with exit code {result.returncode}")
        return Path(output_path).read_text() if json_output else result.stdout
    finally:
        if output_path:
            Path(output_path).unlink(missing_ok=True)


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

    @task(retries=0)
    def check_quality_alert(pipeline_run_id: str) -> None:
        """Surfaces validate's alert_quarantine_rate_exceeded flag as a failed,
        clearly-visible task instead of a line in a report nobody reads. Kept
        as a sibling of train/score (not a blocking predecessor) so the
        pipeline's "quarantine + continue + alert" policy holds: a bad
        quarantine rate flags this run for attention without stopping
        training/scoring on the rows that did pass validation. retries=0
        because retrying doesn't change the underlying data-quality reading.
        """
        report_path = (
            Path(PROJECT_DIR) / "validation" / "output" / f"{pipeline_run_id}_report.json"
        )
        report = json.loads(report_path.read_text())
        if report["alert_quarantine_rate_exceeded"]:
            raise RuntimeError(
                f"Quarantine rate {report['quarantine_rate']:.1%} exceeds the alert "
                f"threshold for run {pipeline_run_id} - see "
                f"validation/output/{pipeline_run_id}_report.json"
            )

    with TaskGroup(group_id="train") as train_group:

        @task(task_id="train_anomaly")
        def train_anomaly() -> str:
            return json.loads(_run_module("models.anomaly", json_output=True))["mlflow_run_id"]

        @task(task_id="train_forecast")
        def train_forecast() -> str:
            return json.loads(_run_module("models.forecast", json_output=True))["mlflow_run_id"]

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
                json_output=True,
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
                json_output=True,
            )
            return json.loads(out)["version"]

        register_anomaly(anomaly_run_id)
        register_forecast(forecast_run_id)

    @task
    def score(pipeline_run_id: str, logical_date_str: str) -> str:
        return _run_module(
            "models.score",
            "--run-id",
            pipeline_run_id,
            "--dag-logical-date",
            logical_date_str,
            json_output=True,
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
    check_quality_alert_task = check_quality_alert(pipeline_run_id="{{ run_id }}")
    score_task = score(pipeline_run_id="{{ run_id }}", logical_date_str="{{ ds }}")
    publish(pipeline_run_id="{{ run_id }}", score_summary_json=score_task)

    ingest_task >> validate_task >> train_group
    validate_task >> check_quality_alert_task
    register_group >> score_task
