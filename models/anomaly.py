"""Anomaly / data-quality model: IsolationForest over facility-normalized
indicator z-scores, flagging suspicious clinic-months for the dashboard.

Two-layer design (documented here since it explains why the eval numbers look
the way they do): validation/checks.py already runs a deterministic z-score
guardrail on the *raw* batch and quarantines the most obvious univariate
spikes before they ever reach monthly_reports. This model runs downstream, on
the *clean* fact table, over a multivariate facility-normalized feature space.
Its job is the subtler cases: behavioral anomalies like severe reporting
delays that pass validation cleanly (delay >= 0 is a valid value, just an
unusual one for that facility), plus any obvious spikes that slipped through
validation's threshold.

Evaluated against the ground-truth injection ledger from data_gen, restricted
to the subset of injected anomaly candidates that actually survive into
monthly_reports (the rest were already caught upstream - see the eval report
for that split).

contamination is a fixed business assumption (expected ~5% of clinic-months
warrant a look), not tuned against the ground-truth labels - tuning against
the same labels used for evaluation would make the reported precision/recall
meaningless.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import IsolationForest

from data_gen.config import SEED
from models.features import ANOMALY_FIELDS, build_anomaly_features, load_monthly_reports
from models.mlflow_utils import configure_mlflow
from warehouse.db import get_engine

CONTAMINATION = 0.05
Z_REASON_THRESHOLD = 2.0
GROUND_TRUTH_PATH = "data_gen/output/ground_truth_anomalies.csv"
OUTPUT_DIR = "models/output"


def build_reasons(z_scores: pd.DataFrame, flagged_mask: pd.Series) -> pd.Series:
    """Plain-language reason list per flagged row: which fields were unusual
    for that facility, and in which direction - this is what the dashboard's
    "why" column renders, so it's evaluated against z-scores, not SHAP
    (IsolationForest is unsupervised; z-scores are more honest here)."""
    reasons = pd.Series([[] for _ in range(len(z_scores))], index=z_scores.index)
    for field in ANOMALY_FIELDS:
        z = z_scores[field]
        significant = (z.abs() > Z_REASON_THRESHOLD) & flagged_mask
        for idx in z_scores.index[significant]:
            reasons.at[idx].append(
                {
                    "field": field,
                    "z_score": round(float(z.loc[idx]), 2),
                    "direction": "high" if z.loc[idx] > 0 else "low",
                }
            )
    return reasons


def evaluate_against_ground_truth(scored: pd.DataFrame) -> dict:
    gt = pd.read_csv(GROUND_TRUTH_PATH, parse_dates=["report_month"])
    candidates = gt[gt["is_anomaly_candidate"]].copy()

    merged = scored.merge(
        candidates[["facility_id", "report_month"]].assign(is_candidate=True),
        on=["facility_id", "report_month"],
        how="left",
    )
    merged["is_candidate"] = merged["is_candidate"].fillna(False).astype(bool)

    total_candidates_injected = candidates.drop_duplicates(["facility_id", "report_month"]).shape[0]
    candidates_present = int(merged["is_candidate"].sum())
    removed_upstream = total_candidates_injected - candidates_present

    tp = int(((merged["is_candidate"]) & (merged["is_anomaly"])).sum())
    fp = int(((~merged["is_candidate"]) & (merged["is_anomaly"])).sum())
    fn = int(((merged["is_candidate"]) & (~merged["is_anomaly"])).sum())
    tn = int(((~merged["is_candidate"]) & (~merged["is_anomaly"])).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "total_candidates_injected": total_candidates_injected,
        "candidates_removed_upstream_by_validation": removed_upstream,
        "candidates_present_in_monthly_reports": candidates_present,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "false_positive_rate": round(fpr, 4),
    }


def train_and_evaluate() -> dict:
    configure_mlflow()
    engine = get_engine()
    df = load_monthly_reports(engine)

    features, z_scores = build_anomaly_features(df)

    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=SEED,
    )
    model.fit(features[ANOMALY_FIELDS])

    raw_scores = model.decision_function(features[ANOMALY_FIELDS])
    predictions = model.predict(features[ANOMALY_FIELDS])  # -1 = anomaly, 1 = normal

    scored = df[["facility_id", "report_month"]].copy()
    scored["is_anomaly"] = predictions == -1
    scored["anomaly_score"] = -raw_scores  # flip sign: higher = more anomalous, more intuitive
    scored["reasons"] = build_reasons(z_scores, scored["is_anomaly"])

    eval_metrics = evaluate_against_ground_truth(scored)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    scored_out = scored.copy()
    scored_out["report_month"] = scored_out["report_month"].dt.date.astype(str)
    scored_out["reasons"] = scored_out["reasons"].apply(json.dumps)
    scored_out.to_csv(f"{OUTPUT_DIR}/anomaly_scored.csv", index=False)
    with open(f"{OUTPUT_DIR}/anomaly_eval.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)

    with mlflow.start_run(run_name="anomaly-isolation-forest") as run:
        mlflow.log_params(
            {
                "model_type": "IsolationForest",
                "contamination": CONTAMINATION,
                "n_estimators": 200,
                "random_state": SEED,
                "features": ANOMALY_FIELDS,
                "z_reason_threshold": Z_REASON_THRESHOLD,
            }
        )
        mlflow.log_metrics(
            {
                "precision": eval_metrics["precision"],
                "recall": eval_metrics["recall"],
                "false_positive_rate": eval_metrics["false_positive_rate"],
                "n_flagged": int(scored["is_anomaly"].sum()),
                "n_rows_scored": len(scored),
                "candidates_removed_upstream_by_validation": eval_metrics[
                    "candidates_removed_upstream_by_validation"
                ],
            }
        )
        mlflow.log_artifact(f"{OUTPUT_DIR}/anomaly_eval.json")
        mlflow.log_artifact(f"{OUTPUT_DIR}/anomaly_scored.csv")
        # Logged but not registered here - `register` is a separate DAG task
        # (models/register.py) that promotes a specific run to the Model
        # Registry, so training a candidate and promoting it are distinct steps.
        mlflow.sklearn.log_model(model, name="model")
        eval_metrics["mlflow_run_id"] = run.info.run_id

    return eval_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the anomaly detector")
    parser.add_argument("--output-file", help="also write the JSON result here")
    args = parser.parse_args()

    metrics = train_and_evaluate()
    print(json.dumps(metrics, indent=2))
    if args.output_file:
        Path(args.output_file).write_text(json.dumps(metrics))
