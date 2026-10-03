"""Airflow DAG: end-to-end training pipeline.

ingest -> validate -> preprocess -> feature engineering -> model training
(MLflow) -> candidate selection -> promotion gate -> model registry.

Scheduled daily; can also be triggered manually:
    airflow dags trigger network_ids_training
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

import pandas as pd
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

logger = logging.getLogger(__name__)

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def ingest(**context):
    """Generate/collect the raw batch and persist it as the run's dataset."""
    from src.data_ingestion import generate_flows, save_raw

    batch_id = f"train-{context['ds']}"
    df = generate_flows(n=40000, seed=batch_id)
    path = save_raw(df, batch_id)
    logger.info("Ingested %s rows -> %s", len(df), path)
    context["ti"].xcom_push(key="batch_id", value=batch_id)
    return str(path)


def validate(**context):
    from src.data_validation import validate

    df = pd.read_csv(context["ti"].xcom_pull(task_ids="ingest"))
    report = validate(df)
    logger.info("Validation: %s", json.dumps(report.summary()))
    if not report.passed:
        raise ValueError(f"Data validation failed: {report.failures}")
    return report.summary()


def preprocess_and_featurize(**context):
    """Clean + engineer features, persist to Postgres, keep parquet for DVC."""
    from src import db
    from src.config import settings
    from src.feature_engineering import add_engineered_features
    from src.data_ingestion import LABEL_COLUMN

    df = pd.read_csv(context["ti"].xcom_pull(task_ids="ingest"))
    df = add_engineered_features(df)
    df["batch_id"] = context["ti"].xcom_pull(task_ids="ingest")
    df["source"] = "synthetic"
    df.to_parquet(settings.DATA_DIR / "processed" / f"{context['ds']}.parquet", index=False)
    # Persist feature rows for the dashboard (raw flow columns only)
    flow_cols = [c for c in df.columns if c not in {"bytes_ratio", "src_bytes_log", "duration_log"}]
    inserted = db.insert_dataframe(df[flow_cols], "processed_records")
    logger.info("Persisted %s processed rows", inserted)
    assert LABEL_COLUMN in df.columns


def train(**context):
    from src.training import run_training_pipeline

    results = run_training_pipeline(trigger="scheduled")
    logger.info(
        "Training done: candidate=%s promoted=%s",
        results["candidate"]["model_name"],
        results["promoted"],
    )
    context["ti"].xcom_push(key="pipeline_run_id", value=results["pipeline_run_id"])
    return results["promoted"]


with DAG(
    dag_id="network_ids_training",
    description="Daily network intrusion model training with MLflow registry promotion",
    default_args=default_args,
    schedule="0 2 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["network-ids", "training", "mlflow"],
) as dag:
    ingest_task = PythonOperator(task_id="ingest", python_callable=ingest)
    validate_task = PythonOperator(task_id="validate", python_callable=validate)
    featurize_task = PythonOperator(task_id="preprocess_and_featurize", python_callable=preprocess_and_featurize)
    train_task = PythonOperator(task_id="train_and_register", python_callable=train)

    ingest_task >> validate_task >> featurize_task >> train_task
