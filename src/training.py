"""Model training, evaluation and MLflow experiment tracking.

Each model in the zoo gets its own MLflow run with parameters, metrics
and artifacts. The configured candidate model is evaluated against the
promotion gate and (conditionally) registered as the new Production
champion in the MLflow Model Registry.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src import db
from src.config import settings
from src.feature_engineering import (
    FeatureBuilder,
    add_engineered_features,
    make_feature_builder,
)
from src.models import LABELS, build_model, is_unsupervised

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = settings.EXPERIMENT_NAME
REGISTRY_MODEL_NAME = settings.REGISTRY_MODEL_NAME


def _get_or_create_experiment() -> str:
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return mlflow.create_experiment(EXPERIMENT_NAME)
    return experiment.experiment_id


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    label = settings.get("training.label_column")
    ratios = settings.get("data")
    X, y = df.drop(columns=[label]), df[label]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=ratios["validation_ratio"] + ratios["test_ratio"],
        stratify=y,
        random_state=settings.RANDOM_STATE,
    )
    rel_test = ratios["test_ratio"] / (ratios["validation_ratio"] + ratios["test_ratio"])
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_test, stratify=y_temp, random_state=settings.RANDOM_STATE,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _metrics(y_true, y_pred) -> dict:
    return {
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
    }


def _binary_metrics(y_true, y_pred) -> dict:
    """Anomaly-detection view: attack-vs-normal.

    Unsupervised detectors flag anomalies without attributing classes, so
    this is their primary quality metric (macro F1 measures attribution).
    """
    to_binary = lambda v: np.where(np.asarray(v) == "normal", "normal", "attack")
    return {
        "anomaly_f1": float(f1_score(to_binary(y_true), to_binary(y_pred), pos_label="attack", zero_division=0)),
        "anomaly_precision": float(precision_score(to_binary(y_true), to_binary(y_pred), pos_label="attack", zero_division=0)),
        "anomaly_recall": float(recall_score(to_binary(y_true), to_binary(y_pred), pos_label="attack", zero_division=0)),
    }


def train_model(model_name: str, X_train, y_train, X_val, y_val, feature_builder: FeatureBuilder, parent_run=None):
    """Train one zoo model inside its own MLflow run and log everything."""
    experiment_id = _get_or_create_experiment()
    start = time.time()

    Xtr = feature_builder.transform(X_train)
    ytr = y_train.to_numpy()

    with mlflow.start_run(experiment_id=experiment_id, run_name=f"{model_name}-{int(time.time())}", nested=True) as run:
        model = build_model(model_name)
        model.fit(Xtr, ytr)

        pred_val = model.predict(feature_builder.transform(X_val))
        val_metrics = _metrics(y_val, pred_val) | _binary_metrics(y_val, pred_val)
        fit_seconds = time.time() - start

        mlflow.log_params(
            {
                "model_name": model_name,
                "n_train_rows": len(X_train),
                "n_val_rows": len(X_val),
                "n_features": Xtr.shape[1],
                "supervised": not is_unsupervised(model_name),
                **(
                    {
                        f"hp_{k}": v
                        for k, v in model.get_params(deep=False).items()
                        if isinstance(v, (int, float, str, bool))
                    }
                    if hasattr(model, "get_params")
                    else {}
                ),
            }
        )
        mlflow.log_metrics(
            {f"val_{k}": v for k, v in val_metrics.items()} | {"fit_seconds": fit_seconds}
        )

        mlflow.log_dict({"val_metrics": val_metrics}, "metrics.json")

        # Persist model + feature builder as MLflow artifacts
        model_path = settings.MODELS_DIR / f"{model_name}.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model_files")
        mlflow.log_dict(
            {"feature_columns": feature_builder.feature_columns}, "feature_builder_meta.json"
        )

        return {
            "run_id": run.info.run_id,
            "model_name": model_name,
            "val_metrics": val_metrics,
            "model_path": str(model_path),
            "fit_seconds": fit_seconds,
        }


def evaluate_on_test(model_name: str, model_path: str, feature_builder: FeatureBuilder, X_test, y_test) -> dict:
    model = joblib.load(model_path)
    pred = model.predict(feature_builder.transform(X_test))
    return _metrics(y_test, pred) | _binary_metrics(y_test, pred)


def get_production_model_version() -> dict | None:
    """Return the current champion (alias) version info, or None.

    Uses the modern MLflow aliases API; 'champion' plays the role the old
    'Production' stage played.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)
    try:
        version = client.get_model_version_by_alias(
            REGISTRY_MODEL_NAME, settings.get("project.registry_alias", "champion")
        )
    except Exception:
        return None  # alias not set yet -> no champion
    run = client.get_run(version.run_id)
    return {
        "version": version.version,
        "run_id": version.run_id,
        "f1": run.data.metrics.get("test_f1"),
    }


def passes_promotion_gate(test_metrics: dict) -> tuple[bool, list[str]]:
    gate = settings.get("training.validation")
    reasons = []
    if test_metrics["f1"] < gate["min_f1"]:
        reasons.append(f"f1 {test_metrics['f1']:.4f} < {gate['min_f1']}")
    if test_metrics["precision"] < gate["min_precision"]:
        reasons.append(f"precision {test_metrics['precision']:.4f} < {gate['min_precision']}")
    if test_metrics["recall"] < gate["min_recall"]:
        reasons.append(f"recall {test_metrics['recall']:.4f} < {gate['min_recall']}")
    return not reasons, reasons


def beats_champion(test_metrics: dict) -> tuple[bool, str]:
    champ = get_production_model_version()
    if champ is None or champ.get("f1") is None:
        return True, "no champion yet"
    improvement = test_metrics["f1"] - champ["f1"]
    gate = settings.get("training.validation")
    if improvement >= gate["min_f1_improvement"]:
        return True, f"beats champion by {improvement:+.4f}"
    return False, f"improvement {improvement:+.4f} < {gate['min_f1_improvement']}"


def register_candidate(run_id: str, model_path: str, test_metrics: dict, notes: str = "") -> str:
    """Register the candidate model artifact and promote it to Production."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)

    # Log model + feature builder as artifacts OF THE CANDIDATE RUN, so the
    # registry version points at a self-contained `runs:/<run_id>/model` source.
    bundle_path = settings.MODELS_DIR / "model.joblib"
    joblib.dump(joblib.load(model_path), bundle_path)  # standardized name for serving
    with mlflow.start_run(run_id=run_id, nested=True):
        mlflow.log_artifact(str(bundle_path), artifact_path="model")
        mlflow.log_artifact(str(settings.MODELS_DIR / "feature_builder.joblib"), artifact_path="model")
        mlflow.log_dict(test_metrics, "model/test_metrics.json")

    try:  # create_registered_model is idempotent-safe for first promotion
        client.create_registered_model(REGISTRY_MODEL_NAME)
    except Exception:
        pass  # already exists

    result = client.create_model_version(
        name=REGISTRY_MODEL_NAME,
        source=f"runs:/{run_id}/model",
        run_id=run_id,
        description=notes,
    )
    # Modern aliases API: point 'champion' at the new version (the old
    # 'Production' stage equivalent; previous champion keeps its version
    # history and simply loses the alias).
    client.set_registered_model_alias(
        name=REGISTRY_MODEL_NAME,
        alias=settings.get("project.registry_alias", "champion"),
        version=result.version,
    )
    logger.info(
        "Registered model version %s with alias '%s'",
        result.version,
        settings.get("project.registry_alias", "champion"),
    )
    return result.version


def run_training_pipeline(
    df: pd.DataFrame | None = None, trigger: str = "scheduled", extra_notes: str = ""
) -> dict:
    """Full training flow: split -> compare zoo -> evaluate -> maybe register."""
    from src.data_ingestion import LABEL_COLUMN

    if df is None:
        df = db.fetch_dataframe(f"SELECT * FROM processed_records WHERE source='synthetic'")
        df = df.drop(columns=["id", "batch_id", "source", "created_at"], errors="ignore")

    # Log a dataset summary to MLflow as the pipeline-level run
    experiment_id = _get_or_create_experiment()
    summary = {"rows": len(df), "label_counts": df[LABEL_COLUMN].value_counts().to_dict()}

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    fb = make_feature_builder(X_train)
    fb_path = settings.MODELS_DIR / "feature_builder.joblib"
    joblib.dump(fb, fb_path)

    # Drift reference = training distribution (read by drift_detection)
    ref_dir = settings.DATA_DIR / "processed"
    ref_dir.mkdir(parents=True, exist_ok=True)
    X_train.assign(**{settings.get("training.label_column"): y_train}).to_parquet(
        ref_dir / "reference.parquet", index=False
    )

    results = []
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"pipeline-{trigger}-{int(time.time())}") as pipeline_run:
        mlflow.log_dict(summary, "dataset_summary.json")
        mlflow.log_params({"trigger": trigger, "rows": len(df)})

        for name in settings.get("training.models"):
            try:
                res = train_model(name, X_train, y_train, X_val, y_val, fb, parent_run=pipeline_run.info.run_id)
                results.append(res)
                logger.info("%s val F1=%.4f", name, res["val_metrics"]["f1"])
            except Exception:
                logger.exception("model %s failed", name)

        if not results:
            raise RuntimeError("All models failed to train")

        # Test-set evaluation for every model; pick candidate
        candidate_name = settings.get("training.candidate_model")
        for res in results:
            test_metrics = evaluate_on_test(res["model_name"], res["model_path"], fb, X_test, y_test)
            res["test_metrics"] = test_metrics
            # Log test metrics onto the MODEL'S OWN run so champion/challenger
            # comparison reads test_f1 from the registry version's run.
            with mlflow.start_run(run_id=res["run_id"], nested=True):
                mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            with mlflow.start_run(experiment_id=experiment_id, run_name=f"eval-{res['model_name']}", nested=True) as eval_run:
                mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
                mlflow.set_tag("mlflow.parentRunId", pipeline_run.info.run_id)
                mlflow.set_tag("evaluated_model", res["model_name"])

        candidate = next(r for r in results if r["model_name"] == candidate_name)
        mlflow.log_metrics({f"candidate_test_{k}": v for k, v in candidate["test_metrics"].items()})

        # ---- Promotion gate ----
        champion_before = get_production_model_version()
        ok, reasons = passes_promotion_gate(candidate["test_metrics"])
        better, compare_note = beats_champion(candidate["test_metrics"])
        promote = ok and better
        mlflow.log_metrics({"gate_passed": int(ok), "beats_champion": int(better), "promoted": int(promote)})

        new_version = None
        if promote:
            # Persist test-set prediction sample as artifact, then register
            sample_idx = X_test.sample(min(200, len(X_test)), random_state=settings.RANDOM_STATE).index
            model = joblib.load(candidate["model_path"])
            preds = model.predict(fb.transform(X_test.loc[sample_idx]))
            sample_df = X_test.loc[sample_idx].assign(
                prediction=preds, true_label=y_test.loc[sample_idx]
            )
            sample_df.to_parquet(settings.DATA_DIR / "processed" / "latest.parquet", index=False)
            new_version = register_candidate(
                candidate["run_id"], candidate["model_path"], candidate["test_metrics"],
                notes=f"trigger={trigger}; {compare_note}{extra_notes}",
            )
        else:
            logger.info("Candidate NOT promoted: %s / %s", reasons, compare_note)

        pipeline_id = db.insert_retraining_event(
            trigger=trigger,
            old_model_version=(champion_before or {}).get("version"),
            old_f1=(champion_before or {}).get("f1"),
            new_model_version=new_version,
            new_run_id=candidate["run_id"],
            new_f1=candidate["test_metrics"]["f1"],
            promoted=promote,
            notes=f"gate={ok}; {compare_note}; {reasons}{extra_notes}",
        )

        return {
            "pipeline_run_id": pipeline_run.info.run_id,
            "candidate": candidate,
            "all_results": results,
            "promoted": promote,
            "new_model_version": new_version,
        }
