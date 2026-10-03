"""FastAPI serving layer for the production anomaly-detection model.

Endpoints:
  POST /predict         - score a batch of network flows
  POST /predict-record  - score a single flow record
  GET  /health          - liveness + model metadata
  GET  /model/info      - registry version, run id, training metrics
  GET  /drift/latest    - latest drift verdict from the monitoring DAG
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import settings

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Network Anomaly Detection API",
    version="1.0.0",
    description="Serving the champion IDS model (MLflow registry) for network flow classification.",
)

# ---- Schemas -----------------------------------------------------------


class FlowRecord(BaseModel):
    duration: float = Field(0.0, ge=0)
    protocol: str = "tcp"
    service: str = "http"
    flag: str = "SF"
    src_bytes: float = Field(0.0, ge=0)
    dst_bytes: float = Field(0.0, ge=0)
    count: float = Field(0.0, ge=0)
    srv_count: float = Field(0.0, ge=0)
    same_srv_rate: float = Field(0.0, ge=0, le=1)
    diff_srv_rate: float = Field(0.0, ge=0, le=1)
    src_port_entropy: float = Field(0.0, ge=0)
    packet_rate: float = Field(0.0, ge=0)
    connection_duration_std: float = Field(0.0, ge=0)
    failed_logins: float = Field(0.0, ge=0)
    bytes_per_packet: float = Field(0.0, ge=0)


class PredictRequest(BaseModel):
    records: list[FlowRecord]
    batch_id: str = "api"
    persist: bool = True


class PredictResponse(BaseModel):
    model_version: str
    n_records: int
    predictions: list[dict[str, Any]]
    anomaly_rate: float
    persist_rows: int


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    run_id: str
    training_f1: float | None
    training_precision: float | None
    training_recall: float | None
    last_retrained: str | None


# ---- Model loading -----------------------------------------------------


class _LazyModel:
    """Defers registry download until first request (keeps /health fast)."""

    def __init__(self):
        self._model = None
        self._error: str | None = None

    def get(self):
        if self._model is None and self._error is None:
            try:
                from src.inference import ProductionModel

                self._model = ProductionModel()
            except Exception as exc:  # pragma: no cover
                logger.exception("Failed to load production model")
                self._error = str(exc)
        return self._model

    def reset(self):
        self._model = None
        self._error = None


_model = _LazyModel()


def _flow_frame(records: list[FlowRecord]) -> pd.DataFrame:
    return pd.DataFrame([r.model_dump() for r in records])


# ---- Endpoints ---------------------------------------------------------


@app.get("/health")
def health():
    model = _model.get()
    return {
        "status": "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
        "error": _model._error,
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/model/info", response_model=ModelInfo)
def model_info():
    from src.training import get_production_model_version

    champion = get_production_model_version()
    if champion is None:
        raise HTTPException(404, "No champion model registered yet (alias unset)")
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)
    run = client.get_run(champion["run_id"])
    return ModelInfo(
        model_name=settings.REGISTRY_MODEL_NAME,
        model_version=str(champion["version"]),
        run_id=champion["run_id"],
        training_f1=run.data.metrics.get("test_f1"),
        training_precision=run.data.metrics.get("test_precision"),
        training_recall=run.data.metrics.get("test_recall"),
        last_retrained=run.info.end_time and datetime.fromtimestamp(run.info.end_time / 1000).isoformat(),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = _model.get()
    if model is None:
        raise HTTPException(503, f"Model unavailable: {_model._error}")
    if not req.records:
        raise HTTPException(400, "records must not be empty")
    if len(req.records) > 10000:
        raise HTTPException(400, "records must not exceed 10000")
    df = _flow_frame(req.records)
    labels, scores, confidence = model.predict_df(df)
    predictions = [
        {
            "prediction": label,
            "is_anomaly": label != "normal",
            "anomaly_score": round(float(score), 6),
            "confidence": round(float(conf), 6),
        }
        for label, score, conf in zip(labels, scores, confidence)
    ]
    persist_rows = 0
    if req.persist:
        from src import db

        frame = pd.DataFrame(
            {
                "batch_id": req.batch_id,
                "source": "api",
                "model_name": settings.REGISTRY_MODEL_NAME,
                "model_version": str(model.version),
                "run_id": model.run_id,
                "attack_type": labels,
                "anomaly_score": scores,
                "is_anomaly": labels != "normal",
                "true_label": None,
                "confidence": confidence,
                # pandas.to_sql bypasses SQLAlchemy column defaults; set created_at explicitly
                "created_at": datetime.now(timezone.utc),
            }
        )
        persist_rows = db.insert_dataframe(frame, "predictions")
    return PredictResponse(
        model_version=str(model.version),
        n_records=len(predictions),
        predictions=predictions,
        anomaly_rate=float(sum(p["is_anomaly"] for p in predictions) / len(predictions)),
        persist_rows=persist_rows,
    )


@app.post("/predict-record")
def predict_record(record: FlowRecord):
    response = predict(PredictRequest(records=[record], batch_id="api-record", persist=False))
    return response.predictions[0]


@app.get("/drift/latest")
def drift_latest():
    from src import db

    row = db.fetch_dataframe("SELECT * FROM drift_reports ORDER BY created_at DESC LIMIT 1")
    if row.empty:
        raise HTTPException(404, "No drift reports yet - run the monitoring DAG")
    rec = row.iloc[0]
    return {
        "overall_drift": bool(rec["overall_drift"]),
        "retrain_recommended": bool(rec["retrain_recommended"]),
        "feature_drift_share": float(rec["feature_drift_share"] or 0.0),
        "drifted_features": (rec["drifted_features"] or "").split(",") if rec["drifted_features"] else [],
        "live_f1": rec["live_f1"],
        "model_version": rec["model_version"],
        "created_at": rec["created_at"],
    }
