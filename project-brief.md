# Project: health-data-pipeline

Infra-first ML platform demo. Goal: show production-grade SWE/MLOps skills, with two
competent-but-lean ML models on top. Target: 2 days (realistically 3–4 if setup fights
back — see "Degrade-gracefully ship order"). Audience: interview portfolio piece for an
AI/ML Engineer role focused on data-warehouse integration (health/NGO domain).

Emphasis split: ~70% infra polish, ~30% ML. Models should be simple and well-evaluated,
not novel. Depth should live in reliability, structure, and documentation.

**Guiding principle:** a reviewer trusts 6 solid components more than 10 shaky ones. This
is a credibility artifact — never demo a broken DAG or an API that can't load from the
registry. Cut depth before you cut correctness.

## Domain (synthetic data only — no real patient data)

Simulate a health-program data warehouse: ~50 fake clinics reporting monthly indicators
over 24 months (patients tested, viral load suppression %, drug stock level, reporting
delay in days). That's ~1,200 base rows — small on purpose. The point is reliability and
structure, not scale; call this out in "data limitations."

Inject realistic problems: missing months, duplicate facility IDs, outlier spikes,
delayed reports. **Keep a ground-truth record of every injected issue** (which rows, what
kind) — this is used to *evaluate* the anomaly detector later (see Model layer). Keep the
injection logic and the detector genuinely independent: don't let the detector see the
features that define the injected labels, or you're grading it on its own answer key.

Reproducibility: one `SEED` constant drives all randomness, referenced by name in the
README so a reviewer can re-run and get identical data.

## Components to build

1. **Data generator** — Python script producing the synthetic dataset with seasonality
   + injected data-quality issues, seeded for reproducibility. Emits both the dataset and
   a ground-truth anomaly ledger.
2. **Warehouse** — Postgres. Real schema (facilities, monthly_reports, plus
   `scored_reports` for model output). Migrations via Alembic. Data-quality enforced at
   two layers:
   - **Structural** (DB constraints): not-null, uniqueness, referential integrity,
     unique `(facility_id, report_month)`.
   - **Semantic** (validate task, see below): the interesting checks live here, not in
     the schema.
   Every scored row carries a `run_id` (traces to the MLflow run) and the DAG logical
   date, for basic lineage/provenance.
3. **Orchestration** — Airflow DAG:
   `ingest → validate → train → register → score → publish`.
   - Note train and score are **separate tasks** on purpose: training is a scheduled
     batch job that logs + registers a model; scoring pulls the *registered* model and
     writes flags/forecasts. The API loads the model the same way `score` does — one
     registry, one load path.
   - **Idempotency (concrete, not a buzzword):**
     - Ingest/score upsert on `(facility_id, report_month)` via
       `INSERT ... ON CONFLICT DO UPDATE` — a re-run overwrites, never duplicates.
     - Tasks are parameterized by the DAG logical date (`ds`), never `now()`, so
       backfills are safe.
     - Invariant to state in the README: *any task can be re-run for any period without
       corrupting the warehouse.*
   - Retries, sensible scheduling (monthly), clear task boundaries.
4. **Data-quality validation** (the `validate` task — called out as its own component
   because the JD names it explicitly)
   - A hand-rolled check suite (lighter and more controllable than Great Expectations for
     a 2-day build) covering the semantic checks: suppression % within [0,100],
     reporting delay ≥ 0, outlier spikes, month gaps, orphan facility IDs.
   - Emits a **structured validation report** (JSON + short HTML) as a DAG artifact.
   - Failure policy: **quarantine bad rows + continue + alert** (don't hard-fail the
     whole run on a few bad rows; do surface them). Having this opinion is itself signal.
5. **Model layer**
   - **Anomaly / data-quality model:** `sklearn.IsolationForest` (or z-score baseline) to
     flag suspicious clinic-months. **Evaluate it against the injected ground-truth
     labels** — report precision/recall and false-positive rate (e.g. "caught 87% of
     injected anomalies at X% FPR"). This turns "I ran IsolationForest" into "I built a
     detector and measured it."
   - **Forecasting model:** lag-feature gradient boosting (chosen over Prophet: lighter
     deps, faster CI/Docker, pairs cleanly with SHAP) for next-quarter caseload or
     suppression rate per clinic. Report a proper backtest metric (e.g. MAE vs. a naive
     last-value baseline).
   - Log every run to MLflow: params, metrics, artifacts, model registry entry.
   - **Explainability, split by model:**
     - Forecasting GBM → **SHAP** (clean, supervised).
     - Anomaly detector → **per-feature z-scores / "which indicator was abnormal and by
       how much"** rather than SHAP (IsolationForest is unsupervised; this is both more
       honest and more plain-language for the dashboard).
6. **Serving API** — FastAPI, loads the current registered model from MLflow (same load
   path as the `score` task), Pydantic request/response validation, `/health` endpoint,
   `model_version` in every response. Containerized with Docker.
7. **Dashboard** — Streamlit, single page: "clinics flagged this month and why," plain
   language, no jargon. "Why" comes from the anomaly detector's per-feature abnormality,
   not SHAP.
8. **CI/CD** — GitHub Actions: lint, run tests, build Docker image on push.
9. **IaC** — Minimal Terraform stub (e.g. container registry + placeholder service) —
   doesn't need to be deployed live, needs to exist and be coherent.
10. **Observability** — Structured logging (not print), health checks, and
    `docs/monitoring.md` describing what would be alerted on in production (model drift,
    spike in anomaly rate, ingestion failure, validation-quarantine rate).
11. **Docs**
    - `README.md` with assumptions, data limitations (incl. the ~1,200-row scale caveat),
      model limitations, reproducibility (the `SEED`), the idempotency invariant, and
      recommended next steps for a real production version. Reads like something you'd
      hand a non-technical stakeholder, plus a technical appendix.
    - `docs/architecture.md` — a diagram of the DAG + service topology (often the first
      thing a reviewer opens).
    - `docs/monitoring.md` — see Observability.

## Explicit non-goals

- No hyperparameter tuning / model novelty chasing.
- No real cloud deployment required — Terraform + Docker Compose is enough to reason
  about deployment, not necessarily execute it.
- No auth/user management — out of scope.
- No NLP/RAG component — not needed for this JD's core ask.
- **No streaming — batch, monthly only.** Scoped deliberately.

## Degrade-gracefully ship order

If time runs short, sacrifice depth in this order — and never violate the guiding
principle above:

1. Drop the Terraform stub first.
2. Then Streamlit polish (keep it functional, drop the nice-to-haves).
3. Then the CI matrix (keep a single lint+test+build path).

**Never ship:** a broken DAG, or an API that doesn't actually load from the registry.
Those two are the spine of the interview story.

## Suggested repo layout

```
health-data-pipeline/
├── data_gen/            # synthetic data generator + ground-truth anomaly ledger
├── warehouse/           # schema, migrations, structural constraints
├── validation/          # semantic data-quality check suite + report
├── dags/                # Airflow DAG(s)
├── models/              # training + scoring scripts, MLflow logging, SHAP
├── api/                 # FastAPI service
├── dashboard/           # Streamlit app
├── infra/               # Dockerfile(s), docker-compose.yml, terraform/
├── .github/workflows/   # CI/CD
├── docs/                # architecture.md, monitoring.md
└── README.md
```

## Build order (2 days, realistically 3–4)

**Day 1**: data generator (+ ground-truth ledger) → warehouse schema + structural
constraints → semantic validation suite → both models trained, evaluated, and logged to
MLflow.
**Day 2**: FastAPI serving from MLflow registry → Docker → Airflow DAG wiring it together
(`ingest → validate → train → register → score → publish`, idempotent) → Streamlit
dashboard → GitHub Actions → Terraform stub → README + architecture diagram.

> The hidden monster on Day 2 is the Docker Compose stack: Postgres + Airflow (webserver,
> scheduler, metadata DB, executor) + MLflow + API + Streamlit all talking to each other.
> Budget half a day just to get it healthy.

## Interview angle to keep in mind while building

Every component should map to a specific line in the target job description
(anomaly detection, forecasting, **data quality validation**, MLOps/model registry,
explainability for non-technical stakeholders, cloud/containerized deployment). Favor
clarity and correctness over cleverness — this is a credibility artifact, not a research
project.

Three things that most raise the ceiling in an interview:
1. train/score split with a single registry load path (MLOps maturity).
2. data-quality validation as a real, opinionated component with a report + failure
   policy (directly names a JD ask).
3. the anomaly detector *evaluated* against injected ground truth (turns a demo into a
   measurement).
