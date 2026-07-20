# Monitoring

## What's actually implemented today

- **Structured logging.** `api/main.py` uses Python's `logging` module (not `print`),
  configured at `INFO`. DAG task output is captured by Airflow's own task logs.
- **Health checks.** `api` exposes `/health` (also reports whether models are loaded);
  `dashboard` exposes Streamlit's built-in `/_stcore/health`; `mlflow` exposes `/health`.
  All three, plus `postgres` and `airflow-webserver`, have Docker Compose healthchecks
  wired to `depends_on: condition: service_healthy`, so the stack won't come up in a
  half-started state.
- **A structured validation report as a monitoring artifact.** Every `validate` task run
  produces `validation/output/{run_id}_report.json` (and an HTML twin) with
  `quarantine_rate`, a breakdown of `quarantine_reasons`, `completeness_gap_count`, and an
  `alert_quarantine_rate_exceeded` boolean (fires above a 10% quarantine rate — see
  `validation/checks.py::build_report`). Today this is a file, not a wired alert; see
  "gaps" below.
- **Airflow's own retry/failure surface.** Every task retries twice with a 5-minute
  delay (`dags/health_pipeline_dag.py`'s `default_args`) before a run shows as failed in
  the Airflow UI.

## What a production version would alert on

None of the below is wired to a real notification channel yet (no PagerDuty/Slack
integration exists in this repo) — this is the alerting design a production rollout
should implement, listed in priority order:

1. **Ingestion / DAG task failure.** Any task exhausting its retries should page — this
   is the DAG's core promise (`ingest → validate → train → register → score → publish`
   completing monthly). Airflow supports this natively via `on_failure_callback` or a
   dedicated alerting provider (Slack/PagerDuty operators).
2. **Validation quarantine-rate spike.** `alert_quarantine_rate_exceeded` in the
   validation report already computes this (>10% quarantined); a production version
   would route it to a real alert instead of leaving it in a JSON file, and would trend
   it over time (a one-off spike vs. a sustained shift in upstream data quality are very
   different problems).
3. **Anomaly rate drift.** The anomaly detector assumes ~5% of clinic-months warrant a
   look (`CONTAMINATION = 0.05` in `models/anomaly.py`) — this is a fixed business
   assumption, not something IsolationForest recalibrates on its own. If the flagged
   rate in `scored_reports` drifts materially from that baseline over several runs, it's
   a signal that either the underlying data distribution changed or the model needs
   retraining, and should alert.
4. **Forecast accuracy drift.** `models/forecast.py` backtests MAE against a naive
   last-value baseline at training time (currently ~15.75% improvement — see the
   README). A production version would track live MAE (once actuals catch up 3 months
   later) and alert if the model stops beating that baseline by a meaningful margin.
5. **Model registry promotion without a gate.** `models/register.py` promotes every
   training run's candidate unconditionally — its own docstring calls this out as a
   simplification for this repo, since nothing here tunes or competes models against
   each other. A production version should gate promotion on the candidate beating the
   currently-registered version's metrics, and alert if a promotion is skipped or a
   candidate regresses.
6. **API error rate / latency.** `api/main.py` has no metrics middleware yet; a
   production version would export request latency and error-rate metrics (e.g. via
   `prometheus-fastapi-instrumentator`) and alert on elevated 5xx rates or the
   `/health` check reporting `models_loaded: false`.

## Suggested tooling for a real rollout

- **Metrics/dashboards:** Prometheus + Grafana (or a managed equivalent), scraping the
  API and Airflow.
- **Alert routing:** Airflow's callback hooks and/or a dedicated alerting service
  (PagerDuty, Opsgenie) fed by both Airflow task failures and the validation report's
  `alert_quarantine_rate_exceeded` flag.
- **Drift detection:** track `scored_reports.is_anomaly` rate and forecast MAE as time
  series (even a scheduled query + Grafana panel would beat nothing); a more mature
  setup would use a dedicated drift-detection library (e.g. Evidently) scored against
  the same MLflow-logged baselines.
