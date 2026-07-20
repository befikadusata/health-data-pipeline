"""Streamlit dashboard: single page, "clinics flagged this month and why."

Reads directly from scored_reports + monthly_reports + facilities (same
DATABASE_URL as the DAG/API) rather than the DAG's reports/latest_summary.json,
so it always reflects whatever the warehouse currently holds, not just the
most recent DAG run's snapshot.

The "why" is the anomaly detector's per-feature z-scores (models/anomaly.py's
build_reasons), not SHAP - see project-brief.md's explainability split:
IsolationForest is unsupervised, so per-feature deviation-from-this-clinic's-
own-history is both more honest and more plain-language than SHAP would be.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from sqlalchemy import text

from warehouse.db import get_engine

FIELD_LABELS = {
    "patients_tested": "patients tested",
    "suppression_pct": "viral load suppression rate",
    "drug_stock_level": "drug stock level",
    "reporting_delay_days": "reporting delay",
}

st.set_page_config(page_title="Health Program Monitoring", page_icon="🏥", layout="wide")


@st.cache_resource
def _engine():
    return get_engine()


@st.cache_data(ttl=60)
def load_scored() -> pd.DataFrame:
    query = text(
        """
        SELECT
            s.facility_id,
            s.report_month,
            s.is_anomaly,
            s.anomaly_score,
            s.anomaly_reasons,
            s.forecast_next_quarter_suppression_pct,
            s.model_version_anomaly,
            s.model_version_forecast,
            f.facility_name,
            f.region,
            m.patients_tested,
            m.suppression_pct,
            m.drug_stock_level,
            m.reporting_delay_days
        FROM scored_reports s
        JOIN facilities f ON f.facility_id = s.facility_id
        JOIN monthly_reports m
            ON m.facility_id = s.facility_id AND m.report_month = s.report_month
        """
    )
    df = pd.read_sql(query, _engine())
    df["report_month"] = pd.to_datetime(df["report_month"])
    return df


def reasons_to_sentence(reasons: list[dict]) -> str:
    if not reasons:
        return "Flagged by the model, but no single indicator crossed the plain-language threshold."
    parts = []
    for r in reasons:
        label = FIELD_LABELS.get(r["field"], r["field"])
        direction = "much higher" if r["direction"] == "high" else "much lower"
        parts.append(f"**{label}** was {direction} than usual for this clinic")
    return "; ".join(parts) + "."


def main() -> None:
    st.title("Health Program Monitoring")
    st.caption("Clinics flagged this month, and why - plain language, no jargon.")

    try:
        df = load_scored()
    except Exception as exc:  # noqa: BLE001 - surface a friendly message, not a stack trace
        st.error(f"Couldn't reach the warehouse: {exc}")
        st.stop()

    if df.empty:
        st.info("No scored data yet - run the DAG's `score` task first.")
        st.stop()

    months = sorted(df["report_month"].unique(), reverse=True)
    selected_month = st.selectbox(
        "Report month",
        months,
        format_func=lambda d: pd.Timestamp(d).strftime("%B %Y"),
    )
    month_df = df[df["report_month"] == selected_month]
    flagged = month_df[month_df["is_anomaly"]].sort_values("anomaly_score", ascending=False)

    col1, col2, col3 = st.columns(3)
    col1.metric("Clinics reporting this month", len(month_df))
    col2.metric("Clinics flagged", len(flagged))
    col3.metric(
        "Flagged rate",
        f"{(len(flagged) / len(month_df) * 100):.1f}%" if len(month_df) else "n/a",
    )

    st.divider()

    if flagged.empty:
        st.success("No clinics flagged this month.")
    else:
        st.subheader("Flagged clinics")
        for _, row in flagged.iterrows():
            with st.container(border=True):
                header_col, forecast_col = st.columns([3, 1])
                header_col.markdown(f"### {row['facility_name']} ({row['facility_id']}) — {row['region']}")
                if pd.notna(row["forecast_next_quarter_suppression_pct"]):
                    forecast_col.metric(
                        "Forecast: suppression % next quarter",
                        f"{row['forecast_next_quarter_suppression_pct']:.1f}%",
                    )
                st.markdown(f"**Why flagged:** {reasons_to_sentence(row['anomaly_reasons'])}")
                st.caption(
                    f"Patients tested: {row['patients_tested']} · "
                    f"Suppression: {row['suppression_pct']:.1f}% · "
                    f"Drug stock: {row['drug_stock_level']:.1f} · "
                    f"Reporting delay: {row['reporting_delay_days']} days"
                )

    with st.expander("All clinics this month"):
        display_cols = {
            "facility_id": "Facility",
            "facility_name": "Name",
            "region": "Region",
            "is_anomaly": "Flagged",
            "suppression_pct": "Suppression %",
            "forecast_next_quarter_suppression_pct": "Forecast (next qtr)",
        }
        st.dataframe(
            month_df[list(display_cols)].rename(columns=display_cols),
            hide_index=True,
            use_container_width=True,
        )

    st.caption(
        f"Model versions in use — anomaly: v{month_df['model_version_anomaly'].iloc[0]}, "
        f"forecast: v{month_df['model_version_forecast'].iloc[0]}"
    )


if __name__ == "__main__":
    main()
