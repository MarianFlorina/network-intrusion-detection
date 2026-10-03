"""Airflow DAG: continuous monitoring & automatic retraining loop.

simulate live traffic -> monitor (drift + performance) -> persist report ->
branch:
  * drift detected  -> retrain -> evaluate -> promote if gate passes
  * no drift        -> no-op

Runs hourly. This is the closed MLOps loop that makes the system
self-healing: distribution shift triggers retraining without a human.

Trigger a drift scenario manually:
    airflow dags trigger network_ids_monitoring \
        -c '{"day": "drifted", "drift_profile": {"src_bytes_scale": 2.5, "attack_rate": 0.10}}'
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
from airflow import DAG
from airflow.providers.standard.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.trigger_rule import TriggerRule

logger = logging.getLogger(__name__)

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

DRIFT_SEED_PREFIX = "live-day"


def _run_conf(context) -> dict:
    dag_run = context.get("dag_run")
    return dict(getattr(dag_run, "conf", None) or {})


def simulate_live_traffic(**context):
    """Materialize the live monitoring window from REAL captured traffic.

    Reads source='capture' rows written by scripts/capture.py (your NIC).
    Only if the capture pool is empty does it fall back to a synthetic
    batch, clearly labelled source='live' (demo continuity).
    """
    from src import db
    from src.data_ingestion import generate_flows, save_raw
    from src.feature_engineering import add_engineered_features

    conf = _run_conf(context)
    day = conf.get("day") or datetime.now(timezone.utc).strftime("%Y%m%d%H")

    window = db.fetch_dataframe(
        "SELECT * FROM processed_records WHERE source='capture' "
        "ORDER BY created_at DESC LIMIT 50000"
    )
    if not window.empty:
        logger.info("Live window: %s real captured flows", len(window))
        return f"capture:{len(window)}"

    # Fallback only when nothing has been captured yet
    logger.warning("No captured traffic yet - falling back to synthetic window")
    drift_profile = conf.get("drift_profile")
    df = generate_flows(n=20000, seed=f"{DRIFT_SEED_PREFIX}-{day}", drift=drift_profile)
    path = save_raw(df, f"live-{day}")
    df = add_engineered_features(df)
    df["batch_id"] = f"live-{day}"
    df["source"] = "live"
    flow_cols = [c for c in df.columns if c not in {"bytes_ratio", "src_bytes_log", "duration_log"}]
    db.insert_dataframe(df[flow_cols], "processed_records")
    logger.info("Synthetic fallback batch %s (%s rows) saved %s", day, len(df), path)
    return str(path)


def monitor(**context):
    from src.drift_detection import run_drift_detection

    verdict = run_drift_detection()
    logger.info(
        "Drift verdict: overall=%s retrain=%s share=%.2f drifted=%s",
        verdict["overall_drift"],
        verdict["retrain_recommended"],
        verdict["feature_drift_share"],
        verdict["drifted_features"],
    )
    return verdict


def branch_on_drift(**context):
    verdict = context["ti"].xcom_pull(task_ids="monitor") or {}
    retrain = bool(verdict.get("retrain_recommended"))
    logger.info("Branch: %s", "retrain_pipeline" if retrain else "no_retrain_needed")
    return "retrain_pipeline" if retrain else "no_retrain_needed"


def retrain(**context):
    """Retrain on an ADAPTED corpus: synthetic anchors + pseudo-labelled real
    captured traffic when enough exists, otherwise the synthetic corpus alone.
    Goes through the normal promotion gate either way."""
    from src.adaptation import build_adapted_corpus
    from src.training import run_training_pipeline

    corpus, info = build_adapted_corpus()
    if corpus is not None:
        logger.info("Adapted corpus: %s", info)
        results = run_training_pipeline(
            corpus, trigger="drift",
            extra_notes=(
                f"; adaptation[pseudo={info['confident']}/{info['captured']} captured, "
                f"share={info.get('pseudo_share')}]"
            ),
        )
    else:
        logger.info("Adaptation unavailable (%s); retraining on synthetic corpus", info.get("reason"))
        results = run_training_pipeline(trigger="drift")
    logger.info(
        "Retrain complete: promoted=%s new_version=%s",
        results["promoted"],
        results["new_model_version"],
    )
    return results["promoted"]


def no_retrain(**context):
    logger.info("No significant drift - continuing with current production model")


def record_event(**context):
    """Always-run bookkeeping (fires whichever branch was taken)."""
    from src import db

    ti = context["ti"]
    verdict = ti.xcom_pull(task_ids="monitor") or {}
    db.insert_retraining_event(
        trigger="monitoring_cycle",
        old_model_version=None,
        new_model_version=None,
        new_run_id=None,
        new_f1=verdict.get("live_f1"),
        promoted=bool(ti.xcom_pull(task_ids="retrain_pipeline")),
        notes=(
            f"drift_share={verdict.get('feature_drift_share')}; "
            f"retrain_recommended={verdict.get('retrain_recommended')}"
        ),
    )


with DAG(
    dag_id="network_ids_monitoring",
    description="Hourly drift monitoring with automatic retraining on drift",
    default_args=default_args,
    schedule="0 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["network-ids", "monitoring", "drift", "mlflow"],
) as dag:
    live_task = PythonOperator(task_id="simulate_live_traffic", python_callable=simulate_live_traffic)
    monitor_task = PythonOperator(task_id="monitor", python_callable=monitor)
    branch_task = BranchPythonOperator(task_id="branch_on_drift", python_callable=branch_on_drift)

    retrain_task = PythonOperator(task_id="retrain_pipeline", python_callable=retrain)
    no_task = PythonOperator(task_id="no_retrain_needed", python_callable=no_retrain)
    record_task = PythonOperator(
        task_id="record_event",
        python_callable=record_event,
        trigger_rule=TriggerRule.ALL_DONE,  # runs whichever branch was taken
    )

    live_task >> monitor_task >> branch_task
    branch_task >> [retrain_task, no_task] >> record_task
