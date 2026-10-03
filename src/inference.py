"""Batch + online inference using the Production model from MLflow Registry.

The production artifact bundle contains model.joblib + feature_builder.joblib
(see training.register_candidate). Downloads them once per process and caches
in memory, so Airflow batch scoring and the FastAPI app share identical
preprocessing at serving time.
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

from src.config import settings

logger = logging.getLogger(__name__)


class ProductionModel:
    """Loads and caches the champion model + feature builder.

    Champion = the registry version carrying the configured alias
    (default 'champion'; the modern replacement for the deprecated
    'Production' stage).
    """

    def __init__(self, alias: str | None = None):
        self.alias = alias or settings.get("project.registry_alias", "champion")
        self.model = None
        self.feature_builder = None
        self.version = None
        self.run_id = None
        self._load()

    def _load(self) -> None:
        client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)
        try:
            version = client.get_model_version_by_alias(
                settings.REGISTRY_MODEL_NAME, self.alias
            )
        except Exception:
            raise RuntimeError(
                f"No version with alias '{self.alias}' for "
                f"'{settings.REGISTRY_MODEL_NAME}'. Run the training pipeline first."
            )
        self.version = version.version
        self.run_id = version.run_id

        with tempfile.TemporaryDirectory() as tmp:
            local = client.download_artifacts(version.run_id, "model", tmp)
            local = Path(local)
            import joblib

            self.model = joblib.load(local / "model.joblib" if (local / "model.joblib").exists() else local)
            fb_path = local / "feature_builder.joblib"
            if fb_path.exists():
                import sys

                sys.path.insert(0, str(settings.PROJECT_ROOT))
                self.feature_builder = joblib.load(fb_path)
        logger.info("Loaded %s v%s (run %s)", settings.REGISTRY_MODEL_NAME, self.version, self.run_id)

    def predict_df(self, df):
        """Score a DataFrame of raw flows; returns (labels, scores)."""
        import numpy as np

        if self.feature_builder is not None:
            X = self.feature_builder.transform(df)
        else:
            from src.feature_engineering import add_engineered_features

            X = add_engineered_features(df).drop(columns=["attack_type"], errors="ignore").values
        labels = self.model.predict(X)

        scores = np.zeros(len(labels), dtype=float)
        if hasattr(self.model, "decision_function"):
            try:
                raw = self.model.decision_function(X)
                if raw.ndim == 2:  # classifiers: use max class probability instead
                    raw = None
                else:
                    scores = (raw - raw.min()) / (np.ptp(raw) + 1e-9)
            except Exception:
                raw = None
        if hasattr(self.model, "predict_proba"):
            try:
                proba = self.model.predict_proba(X)
                scores = 1.0 - proba.max(axis=1)
            except Exception:
                pass
        confidence = 1.0 - scores
        return labels, scores, confidence


@lru_cache(maxsize=1)
def get_production_model(alias: str | None = None) -> ProductionModel:
    return ProductionModel(alias)


def score_batch(df, batch_id: str = "inference", source: str = "synthetic", persist: bool = True) -> dict:
    """Score a batch and (optionally) write predictions to the database."""
    import pandas as pd

    from src import db

    model = get_production_model()
    labels, scores, confidence = model.predict_df(df)

    true_labels = df["attack_type"] if "attack_type" in df.columns else None
    records = pd.DataFrame(
        {
            "batch_id": batch_id,
            "source": source,
            "model_name": settings.REGISTRY_MODEL_NAME,
            "model_version": str(model.version),
            "run_id": model.run_id,
            "attack_type": labels,
            "anomaly_score": scores,
            "is_anomaly": labels != "normal",
            "true_label": true_labels if true_labels is not None else None,
            "confidence": confidence,
            # pandas.to_sql bypasses SQLAlchemy column defaults; set created_at explicitly
            "created_at": datetime.now(timezone.utc),
        }
    )
    n = db.insert_dataframe(records, "predictions") if persist else 0

    anomaly_rate = float((labels != "normal").mean())
    return {
        "scored_rows": len(records),
        "persisted_rows": n,
        "anomalies": int((labels != "normal").sum()),
        "anomaly_rate": anomaly_rate,
        "attack_breakdown": pd.Series(labels).value_counts().to_dict(),
        "model_version": str(model.version),
    }

if __name__ == "__main__":
    from src.data_ingestion import generate_flows

    flows = generate_flows(n=5000, seed="demo")
    result = score_batch(flows, batch_id="cli-demo", source="synthetic", persist=False)
    import pprint

    pprint.pprint(result)
