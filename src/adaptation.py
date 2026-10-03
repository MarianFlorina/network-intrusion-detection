"""Model adaptation to REAL captured traffic (pseudo-labeling / self-training).

Captured flows carry no ground-truth labels, so this module:
  1. Pulls capture-source rows from processed_records
  2. Scores them with the current Production model and keeps only
     high-confidence predictions as PSEUDO-LABELS (classic self-training:
     the model teaches the model, gated by confidence)
  3. Blends them with a fresh synthetic ANCHOR batch so all five attack
     classes stay represented in training (real traffic is ~99% benign;
     without anchors the adapted model would drift toward all-normal)
  4. Runs the normal gated pipeline -> registry only promotes a genuine
     champion improvement, and the audit trail records the adaptation

Drift then compares live capture against a reference that INCLUDES real
traffic, so alert storms stop once the model has adapted.
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from src import db
from src.config import settings
from src.data_ingestion import generate_flows

logger = logging.getLogger(__name__)

FLOW_COLUMNS = [
    "duration", "src_bytes", "dst_bytes", "count", "srv_count",
    "same_srv_rate", "diff_srv_rate", "src_port_entropy", "packet_rate",
    "connection_duration_std", "failed_logins", "bytes_per_packet",
    "protocol", "service", "flag", "attack_type",
]


def adaptation_status() -> dict:
    """How much captured traffic is available vs needed for adaptation."""
    min_rows = int(settings.get("adaptation.min_rows", 3000))
    df = db.fetch_dataframe(
        "SELECT COUNT(*) AS n FROM processed_records WHERE source='capture'"
    )
    n = int(df.iloc[0]["n"]) if not df.empty else 0
    return {
        "capture_rows": n,
        "min_rows": min_rows,
        "ready": n >= min_rows,
        "needed": max(0, min_rows - n),
    }


def persist_capture_flows(df: pd.DataFrame, batch_id: str) -> int:
    """Store captured flow rows so adaptation (and drift) can consume them."""
    from datetime import datetime, timezone

    cols = [c for c in FLOW_COLUMNS if c in df.columns and c != "attack_type"]
    frame = df[cols].copy()
    frame["batch_id"] = batch_id
    frame["source"] = "capture"
    # pandas.to_sql bypasses SQLAlchemy column defaults; set created_at explicitly
    frame["created_at"] = datetime.now(timezone.utc)
    return db.insert_dataframe(frame, "processed_records")


def fetch_capture_corpus() -> pd.DataFrame:
    """All captured flows (unlabelled; attack_type is NULL for these rows)."""
    df = db.fetch_dataframe(
        "SELECT * FROM processed_records WHERE source='capture'"
    )
    drop = [c for c in ("id", "batch_id", "source", "created_at") if c in df.columns]
    return df.drop(columns=drop) if not df.empty else df


def pseudo_label(capture_df: pd.DataFrame, min_confidence: float) -> tuple[pd.DataFrame, dict]:
    """Self-training step: production model labels its own high-confidence flows."""
    from src.inference import get_production_model

    model = get_production_model()
    labels, scores, confidence = model.predict_df(capture_df)

    df = capture_df.copy()
    df["attack_type"] = labels
    df["_confidence"] = confidence

    info = {
        "captured": len(df),
        "confident": int((df["_confidence"] >= min_confidence).sum()),
        "label_breakdown": pd.Series(labels).value_counts().to_dict(),
    }
    kept = (
        df[df["_confidence"] >= min_confidence]
        .drop(columns=["_confidence"])
        .reset_index(drop=True)
    )
    return kept, info


def build_adapted_corpus() -> tuple[pd.DataFrame | None, dict]:
    """Synthetic anchors + pseudo-labelled real flows = adapted training set."""
    cfg = {
        "min_rows": int(settings.get("adaptation.min_rows", 3000)),
        "min_confidence": float(settings.get("adaptation.min_confidence", 0.85)),
        "anchor_rows": int(settings.get("adaptation.anchor_rows", 20000)),
    }
    capture = fetch_capture_corpus()
    if len(capture) < cfg["min_rows"]:
        return None, {
            "reason": f"not enough captured rows ({len(capture)} < {cfg['min_rows']})",
            **cfg,
        }

    labelled, info = pseudo_label(capture, cfg["min_confidence"])
    if labelled.empty:
        return None, {"reason": "no flows met the confidence threshold", **cfg, **info}

    anchor = generate_flows(n=cfg["anchor_rows"], seed=f"anchor-{int(time.time())}")
    corpus = (
        pd.concat([anchor[FLOW_COLUMNS], labelled[FLOW_COLUMNS]], ignore_index=True)
        .sample(frac=1.0, random_state=settings.RANDOM_STATE)
        .reset_index(drop=True)
    )
    info.update(
        {
            "anchor_rows": len(anchor),
            "corpus_rows": len(corpus),
            "pseudo_share": round(len(labelled) / len(corpus), 3),
        }
    )
    return corpus, info


def run_capture_adaptation(trigger: str = "adaptation") -> dict:
    """Full adaptation cycle; returns a dict safe for DAG/CLI display."""
    corpus, info = build_adapted_corpus()
    if corpus is None:
        logger.info("Adaptation skipped: %s", info.get("reason"))
        return {"adapted": False, **info}

    from src.training import run_training_pipeline

    note = (
        f"; adaptation[pseudo={info['confident']}/{info['captured']} captured, "
        f"share={info.get('pseudo_share')}]"
    )
    results = run_training_pipeline(corpus, trigger=trigger, extra_notes=note)
    return {
        "adapted": True,
        "adaptation_info": info,
        "promoted": results["promoted"],
        "new_model_version": results["new_model_version"],
        "candidate_f1": results["candidate"]["test_metrics"]["f1"],
        "pipeline_run_id": results["pipeline_run_id"],
    }
