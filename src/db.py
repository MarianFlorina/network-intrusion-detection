"""Database layer: schema creation and typed session helpers.

Used by the pipeline (persist processed rows/predictions), the API
(logging predictions) and the dashboard (reading aggregates).
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import settings


class Base(DeclarativeBase):
    pass


class ProcessedRecord(Base):
    """Feature-engineered rows that flow into training / inference."""

    __tablename__ = "processed_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_id = Column(String(64), index=True)
    source = Column(String(32), default="synthetic")  # synthetic | live | drift
    duration = Column(Float)
    protocol = Column(String(16))
    service = Column(String(32))
    flag = Column(String(16))
    src_bytes = Column(Float)
    dst_bytes = Column(Float)
    count = Column(Float)
    srv_count = Column(Float)
    same_srv_rate = Column(Float)
    diff_srv_rate = Column(Float)
    src_port_entropy = Column(Float)
    packet_rate = Column(Float)
    connection_duration_std = Column(Float)
    failed_logins = Column(Float)
    bytes_per_packet = Column(Float)
    attack_type = Column(String(32), index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class PredictionRecord(Base):
    """Scores emitted by the production model (API + batch inference)."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_id = Column(String(64), index=True)
    source = Column(String(32), default="synthetic")
    model_name = Column(String(128))
    model_version = Column(String(32))
    run_id = Column(String(64), index=True)
    attack_type = Column(String(32), index=True)  # predicted label
    anomaly_score = Column(Float)
    is_anomaly = Column(Boolean, index=True)
    true_label = Column(String(32), nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


class DriftReport(Base):
    """One row per monitoring run (drift_detection.py output)."""

    __tablename__ = "drift_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64))                    # producing MLflow run (if any)
    model_version = Column(String(32))
    window_rows = Column(Integer)
    drifted_features = Column(Text)                # comma-separated
    feature_drift_share = Column(Float)
    live_f1 = Column(Float, nullable=True)
    live_precision = Column(Float, nullable=True)
    live_recall = Column(Float, nullable=True)
    performance_drift = Column(Boolean, default=False)
    overall_drift = Column(Boolean, default=False)
    retrain_recommended = Column(Boolean, default=False)
    report_json = Column(Text)                     # full PSI/KS details
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


class RetrainingEvent(Base):
    """Audit trail of the closed loop: drift -> retrain -> promote."""

    __tablename__ = "retraining_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trigger = Column(String(32))                   # drift | scheduled | manual
    old_model_version = Column(String(32), nullable=True)
    new_model_version = Column(String(32), nullable=True)
    new_run_id = Column(String(64), nullable=True)
    new_f1 = Column(Float, nullable=True)
    old_f1 = Column(Float, nullable=True)
    promoted = Column(Boolean, default=False)
    notes = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


_engine = None
_SessionFactory = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True, future=True)
        Base.metadata.create_all(_engine)
    return _engine


def get_session() -> Session:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)
    return _SessionFactory()


def insert_dataframe(df, table: str) -> int:
    """Append a DataFrame to a table; returns inserted row count."""
    if df.empty:
        return 0
    engine = get_engine()
    with engine.begin() as conn:
        df.to_sql(table, conn, if_exists="append", index=False, chunksize=5000)
    return len(df)


def fetch_dataframe(query: str):
    """Run a read query and return a DataFrame."""
    import pandas as pd

    engine = get_engine()
    return pd.read_sql_query(query, engine)


def insert_retraining_event(
    *,
    trigger: str,
    old_model_version: str | None,
    new_model_version: str | None,
    new_run_id: str | None,
    new_f1: float | None,
    old_f1: float | None = None,
    promoted: bool,
    notes: str = "",
) -> int:
    with get_session() as session:
        event = RetrainingEvent(
            trigger=trigger,
            old_model_version=old_model_version,
            new_model_version=new_model_version,
            new_run_id=new_run_id,
            new_f1=new_f1,
            old_f1=old_f1,
            promoted=promoted,
            notes=notes,
        )
        session.add(event)
        session.commit()
        return event.id


def insert_drift_report(report: dict) -> int:
    """Persist a drift_detection result dict."""
    with get_session() as session:
        row = DriftReport(
            run_id=report.get("run_id"),
            model_version=report.get("model_version"),
            window_rows=report.get("window_rows"),
            drifted_features=",".join(report.get("drifted_features", [])),
            feature_drift_share=report.get("feature_drift_share"),
            live_f1=report.get("live_f1"),
            live_precision=report.get("live_precision"),
            live_recall=report.get("live_recall"),
            performance_drift=report.get("performance_drift", False),
            overall_drift=report.get("overall_drift", False),
            retrain_recommended=report.get("retrain_recommended", False),
            report_json=report.get("report_json"),
        )
        session.add(row)
        session.commit()
        return row.id
