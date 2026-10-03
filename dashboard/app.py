"""NETWORK SECURITY MONITOR - Streamlit dashboard.

Reads aggregates from PostgreSQL (predictions, processed traffic, drift
reports, retraining events) and model status from the MLflow registry
via the FastAPI service.

Responsive: KPI grid collapses 4 -> 2 -> 1 columns, charts and tables
scale to viewport, sidebar starts collapsed on phones, and an optional
auto-refresh (st.fragment) keeps the monitor live.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text

st.set_page_config(
    page_title="Network Security Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---- Connections --------------------------------------------------------


@st.cache_resource
def get_engine():
    url = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://nad_user:nad_password@postgres:5432/network_ids",
    )
    return create_engine(url, pool_pre_ping=True)


@st.cache_resource
def get_api_base():
    return os.getenv("API_BASE_URL", "http://api:8000")


def query_df(sql: str) -> pd.DataFrame:
    try:
        return pd.read_sql_query(text(sql), get_engine())
    except Exception as exc:  # DB not migrated yet / first run
        st.sidebar.warning(f"Database unavailable: {exc.__class__.__name__}")
        return pd.DataFrame()


# ---- Data loads ---------------------------------------------------------


@st.cache_data(ttl=30)
def load_kpis(source: str = "all") -> dict:
    where = "" if source == "all" else f"WHERE source = '{source}'"
    df = query_df(
        f"""
        SELECT
            COUNT(*)                                            AS total_traffic,
            SUM(CASE WHEN is_anomaly THEN 1 ELSE 0 END)         AS anomalies
        FROM predictions
        {where}
        """
    )
    if df.empty or df.iloc[0]["total_traffic"] is None:
        return {"total_traffic": 0, "anomalies": 0}
    row = df.iloc[0]
    total = int(row["total_traffic"])
    anomalies = int(row["anomalies"] or 0)
    return {
        "total_traffic": total,
        "anomalies": anomalies,
        "normal_traffic": total - anomalies,
        "anomaly_rate": (anomalies / total) if total else 0.0,
    }


@st.cache_data(ttl=30)
def load_threat_types(source: str = "all") -> pd.DataFrame:
    where = "WHERE is_anomaly" if source == "all" else f"WHERE is_anomaly AND source = '{source}'"
    return query_df(
        f"""
        SELECT attack_type AS threat, COUNT(*) AS count
        FROM predictions
        {where}
        GROUP BY attack_type
        ORDER BY count DESC
        """
    )


@st.cache_data(ttl=60)
def load_retraining_events() -> pd.DataFrame:
    return query_df(
        """
        SELECT created_at, trigger, promoted, new_model_version, new_f1
        FROM retraining_events
        ORDER BY created_at DESC
        LIMIT 15
        """
    )


@st.cache_data(ttl=60)
def load_feature_drift_history() -> pd.DataFrame:
    return query_df(
        """
        SELECT created_at, feature_drift_share
        FROM drift_reports
        ORDER BY created_at ASC
        LIMIT 200
        """
    )


def fetch_json(path: str) -> dict | None:
    import requests

    try:
        response = requests.get(f"{get_api_base()}{path}", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


# ---- Responsive styling -------------------------------------------------

CSS = """
<style>
/* ---- Fluid type & layout (scales continuously between breakpoints) ---- */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: clamp(8px, 1vw, 16px);
}
.kpi-card {
    background: linear-gradient(135deg, #10192e 0%, #0b1220 100%);
    border: 1px solid #1f2d4d;
    border-radius: 12px;
    padding: clamp(10px, 1.2vw, 16px) clamp(8px, 1vw, 14px);
    text-align: center;
    min-width: 0;
}
.kpi-label {
    color: #8ea2c6;
    font-size: clamp(0.68rem, 0.6rem + 0.35vw, 0.85rem);
    letter-spacing: 0.06em;
}
.kpi-value {
    color: #f0f6ff;
    font-size: clamp(1.25rem, 1rem + 1.3vw, 1.85rem);
    font-weight: 700;
    line-height: 1.25;
}

/* Page shell: readable line length on ultrawide, centered */
.block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
    margin-inline: auto;
}

/* Wide tables scroll horizontally instead of breaking layout */
.stMarkdown table { display: block; max-width: 100%; overflow-x: auto; }

/* Never allow accidental horizontal page scroll on small screens */
.stApp { overflow-x: hidden; }

/* Long model-name chips wrap instead of overflowing */
code { word-break: break-word; }

/* Title scales fluidly */
h1 { font-size: clamp(1.5rem, 1.1rem + 1.6vw, 2.6rem) !important; }

/* ---- Tablet (portrait iPad, small laptops) ---- */
@media (max-width: 1024px) {
    .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}

/* ---- Large phones / portrait tablets ---- */
@media (max-width: 768px) {
    .kpi-grid { gap: 10px; }
    [data-testid="stMetricValue"] { font-size: 1.4rem; }
    .block-container { padding-top: 1.4rem; }
}

/* ---- Small phones ---- */
@media (max-width: 480px) {
    .kpi-grid { grid-template-columns: 1fr; }
    .kpi-card { padding: 10px 8px; }
    .block-container { padding-top: 1rem; padding-inline: 4vw; }
}

/* ---- Landscape phones (short screens) ---- */
@media (max-height: 480px) and (orientation: landscape) {
    h1 { font-size: 1.4rem !important; }
    .kpi-value { font-size: 1.15rem; }
    .block-container { padding-top: 0.6rem; padding-bottom: 1rem; }
}
</style>
"""


def kpi_grid(kpis: dict) -> None:
    """One markdown block = full CSS-grid control (responsive)."""
    cards = [
        ("Total Traffic", f"{kpis['total_traffic']:,}"),
        ("Normal Traffic", f"{kpis.get('normal_traffic', 0):,}"),
        ("Anomalies", f"{kpis['anomalies']:,}"),
        ("Anomaly Rate", f"{kpis.get('anomaly_rate', 0):.1%}"),
    ]
    cells = "".join(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div></div>'
        for label, value in cards
    )
    st.markdown(
        f'<div class="kpi-grid">{cells}</div>',
        unsafe_allow_html=True,
    )


# ---- Dashboard render (wrapped by st.fragment for auto-refresh) ---------


def render_dashboard() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("🛡️ Network Security Monitor")
    st.caption(
        f"Live view · refreshed {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )

    # --- Traffic source filter (real captured traffic by default) ---
    sources = query_df("SELECT DISTINCT source FROM predictions")
    options = ["all"] + sorted(sources["source"].tolist()) if not sources.empty else ["all"]
    default_source = "capture" if "capture" in options else "all"
    source = st.select_slider(
        "Traffic source",
        options=options,
        value=default_source,
        help="'capture' = real packets from your NIC via scripts/capture.py",
    )

    kpis = load_kpis(source)
    kpi_grid(kpis)
    st.divider()

    # --- Threat types + model status ---
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("Threat Types")
        threats = load_threat_types(source)
        if threats.empty:
            st.info("No predictions yet — trigger the training DAG, then the monitoring DAG.")
        else:
            pretty = threats.assign(threat=threats["threat"].str.replace("_", " ").str.title())
            fig = px.bar(
                pretty.sort_values("count"),
                x="count",
                y="threat",
                orientation="h",
                text="count",
                color="count",
                color_continuous_scale="Reds",
                height=300,
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                coloraxis_showscale=False,
                font=dict(size=12),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        st.subheader("Model Status")
        info = fetch_json("/model/info")
        drift = fetch_json("/drift/latest")
        if info:
            st.markdown(f"**Current Model:** `{info['model_name']}` v{info['model_version']}")
            f1 = info.get("training_f1")
            if f1 is not None:
                st.metric("F1 Score (test)", f"{f1:.1%}")
            if info.get("last_retrained"):
                st.caption(f"Last retrained: {str(info['last_retrained'])[:10]}")
        else:
            st.info("No production model yet — run the training DAG.")

        if drift:
            ok = not drift["overall_drift"]
            st.markdown(f"Drift Status: {'✅ Normal' if ok else '⚠️ Drift Detected'}")
            if drift.get("feature_drift_share") is not None:
                st.caption(
                    f"Feature drift share: {drift['feature_drift_share']:.0%}"
                    f" · checked {str(drift['created_at'])[:16]}"
                )
            if drift.get("live_f1"):
                st.caption(f"Live F1: {drift['live_f1']:.1%}")
        else:
            st.caption("Drift Status: no monitoring runs yet.")

    st.divider()

    # --- Drift & retraining history ---
    drift_left, history_right = st.columns(2, gap="large")

    with drift_left:
        st.subheader("Feature Drift Over Time")
        hist = load_feature_drift_history()
        if hist.empty:
            st.caption("No drift reports yet.")
        else:
            hist = hist.assign(created_at=pd.to_datetime(hist["created_at"]))
            fig = px.line(
                hist,
                x="created_at",
                y="feature_drift_share",
                markers=True,
                labels={"feature_drift_share": "Drifted feature share", "created_at": ""},
                height=280,
            )
            fig.add_hline(y=0.2, line_dash="dash", annotation_text="retrain threshold")
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), font=dict(size=12))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with history_right:
        st.subheader("Automatic Retraining Log")
        events = load_retraining_events()
        if events.empty:
            st.caption("No retraining events yet.")
        else:
            events = events.assign(
                created_at=pd.to_datetime(events["created_at"]).dt.strftime("%m-%d %H:%M"),
                promoted=events["promoted"].map(
                    lambda v: "✅" if pd.notna(v) and bool(v) else "—"
                ),
                new_f1=events["new_f1"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—"),
                new_model_version=events["new_model_version"].map(
                    lambda v: f"v{int(v)}" if pd.notna(v) else "—"
                ),
            )
            st.dataframe(
                events[["created_at", "trigger", "promoted", "new_model_version", "new_f1"]],
                use_container_width=True,
                hide_index=True,
            )

    st.divider()
    with st.expander("Orchestration (Airflow & MLflow)"):
        st.markdown(
            """
            | DAG | Schedule | Purpose |
            |---|---|---|
            | `network_ids_training` | daily 02:00 | ingest → validate → featurize → train → MLflow → registry |
            | `network_ids_monitoring` | hourly | drift + performance monitor → branch → auto-retrain |

            Airflow UI: [http://localhost:8080](http://localhost:8080) ·
            MLflow UI: [http://localhost:5000](http://localhost:5000)
            """
        )


# ---- Sidebar controls ---------------------------------------------------

with st.sidebar:
    st.header("⚙️ Monitor settings")
    auto_refresh = st.checkbox("Auto-refresh", value=False, help="Re-poll data sources live")
    interval = st.select_slider("Refresh interval", options=["10s", "30s", "60s"], value="30s")
    st.caption(
        "KPIs & threats refresh every 30 s of cache TTL; auto-refresh re-runs the page "
        "at the selected interval."
    )

if auto_refresh:
    # Re-run just the dashboard body on an interval (no extra deps)
    st.fragment(run_every=interval)(render_dashboard)()
else:
    render_dashboard()
