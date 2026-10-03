"""Drift detection on live traffic + production model monitoring.

Compares the current live window against the training reference
distribution using:
  * Population Stability Index (PSI) per numeric feature
  * Kolmogorov-Smirnov test for numeric features
  * Cramer's V / fraction-shift check for categorical features
  * Live F1 vs training F1 (performance drift) when labels are available

The verdict (`retrain_recommended`) is what Airflow branches on.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from scipy import stats

from src.config import settings
from src.db import fetch_dataframe
from src.feature_engineering import add_engineered_features

logger = logging.getLogger(__name__)

NUMERIC_FEATURES: list[str] = settings.get("training.numeric_features")
CATEGORICAL_FEATURES: list[str] = settings.get("training.categorical_features")
LABEL_COLUMN: str = settings.get("training.label_column")

PSI_BINS = 10


def population_stability_index(reference: np.ndarray, current: np.ndarray) -> float:
    """PSI over quantile-binned reference distribution."""
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) < 50 or len(current) < 50:
        return 0.0
    quantiles = np.linspace(0, 1, PSI_BINS + 1)[1:-1]
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) == 0:
        return 0.0
    ref_counts = np.histogram(reference, bins=[-np.inf, *edges, np.inf])[0] / len(reference)
    cur_counts = np.histogram(current, bins=[-np.inf, *edges, np.inf])[0] / len(current)
    # Smooth to avoid log(0)
    eps = 1e-4
    ref_counts = np.clip(ref_counts, eps, None)
    cur_counts = np.clip(cur_counts, eps, None)
    return float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    ref = reference.fillna("missing").value_counts(normalize=True)
    cur = current.fillna("missing").value_counts(normalize=True)
    categories = set(ref.index) | set(cur.index)
    eps = 1e-4
    psi = 0.0
    for cat in categories:
        p = max(float(ref.get(cat, 0.0)), eps)
        q = max(float(cur.get(cat, 0.0)), eps)
        psi += (q - p) * np.log(q / p)
    return float(psi)


def ks_test(reference: np.ndarray, current: np.ndarray) -> float:
    """Returns KS p-value; small values indicate distribution shift."""
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) < 50 or len(current) < 50:
        return 1.0
    return float(stats.ks_2samp(reference, current).pvalue)


def analyze_feature_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Per-feature drift analysis across numeric and categorical columns."""
    results = {}
    for feature in NUMERIC_FEATURES + ["bytes_ratio", "src_bytes_log", "duration_log"]:
        if feature not in reference.columns or feature not in current.columns:
            continue
        ref = reference[feature].to_numpy(dtype=float)
        cur = current[feature].to_numpy(dtype=float)
        psi = population_stability_index(ref, cur)
        p_value = ks_test(ref, cur)
        psi_warn = float(settings.get("drift.psi_warn"))
        psi_drift = float(settings.get("drift.psi_drift"))
        status = "ok"
        if psi >= psi_drift or p_value < float(settings.get("drift.ks_alpha")) and psi >= psi_warn:
            status = "drift"
        elif psi >= psi_warn:
            status = "warn"
        results[feature] = {
            "psi": round(psi, 4),
            "ks_p_value": round(p_value, 6),
            "status": status,
        }
    for feature in CATEGORICAL_FEATURES:
        if feature not in reference.columns or feature not in current.columns:
            continue
        psi = categorical_psi(reference[feature], current[feature])
        results[feature] = {
            "psi": round(psi, 4),
            "ks_p_value": None,
            "status": "drift" if psi >= float(settings.get("drift.psi_drift")) else ("warn" if psi >= float(settings.get("drift.psi_warn")) else "ok"),
        }
    return results


def evaluate_performance_drift(live_window: pd.DataFrame, training_f1: float | None) -> dict:
    """Compare live F1 (when ground truth exists in the window) to training F1."""
    config = settings.get("drift.perf")
    out = {"live_f1": None, "live_precision": None, "live_recall": None, "performance_drift": False}
    if LABEL_COLUMN not in live_window.columns or training_f1 is None:
        return out
    scored = live_window.dropna(subset=["prediction", LABEL_COLUMN])
    if len(scored) < int(config["min_sample_size"]):
        return out
    from src.training import _metrics

    metrics = _metrics(scored[LABEL_COLUMN], scored["prediction"])
    out.update(
        live_f1=metrics["f1"],
        live_precision=metrics["precision"],
        live_recall=metrics["recall"],
        performance_drift=metrics["f1"] < float(config["min_f1"]),
    )
    return out


def run_drift_detection(
    reference_path=None, live_window: pd.DataFrame | None = None, model_version: str | None = None
) -> dict:
    """Full monitoring pass; returns a verdict dict consumed by Airflow.

    Reference = feature-engineered training rows (data/processed/reference.parquet).
    Live window = rows from `processed_records`/`predictions` since the last check,
    or an explicit DataFrame passed by the DAG / demo script.
    """
    from pathlib import Path

    ref_path = Path(reference_path) if reference_path else settings.DATA_DIR / "processed" / "reference.parquet"
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Reference dataset missing: {ref_path}. Run the training pipeline first."
        )
    reference = add_engineered_features(pd.read_parquet(ref_path))

    if live_window is None:
        live_window = fetch_dataframe(
            "SELECT * FROM processed_records ORDER BY created_at DESC LIMIT 50000"
        )
    if live_window.empty:
        return {
            "overall_drift": False,
            "retrain_recommended": False,
            "reason": "no live data yet",
            "window_rows": 0,
            "drifted_features": [],
            "feature_drift_share": 0.0,
            "performance_drift": False,
        }

    live = add_engineered_features(live_window)

    # Attach production-model predictions so performance drift can be measured
    if "prediction" not in live.columns:
        try:
            from src.inference import get_production_model

            model = get_production_model()
            labels, _, _ = model.predict_df(live)
            live["prediction"] = labels
        except Exception:
            logger.info("No production model available; skipping performance drift")

    min_rows = int(settings.get("drift.min_recent_rows"))
    if len(live) < min_rows:
        return {
            "overall_drift": False,
            "retrain_recommended": False,
            "reason": f"insufficient live rows ({len(live)} < {min_rows})",
            "window_rows": len(live),
            "drifted_features": [],
            "feature_drift_share": 0.0,
            "performance_drift": False,
        }

    feature_results = analyze_feature_drift(reference, live)
    drifted = [f for f, r in feature_results.items() if r["status"] == "drift"]
    warned = [f for f, r in feature_results.items() if r["status"] == "warn"]
    share = len(drifted) / max(len(feature_results), 1)

    overall_drift = share >= float(settings.get("drift.min_feature_drift_share"))

    # ---- Performance drift (needs production model F1 from registry) ----
    training_f1 = None
    try:
        from src.training import get_production_model_version

        champion = get_production_model_version()
        training_f1 = (champion or {}).get("f1")
        model_version = model_version or (champion or {}).get("version")
    except Exception:
        logger.warning("Could not resolve production model for performance drift", exc_info=True)

    perf = evaluate_performance_drift(live, training_f1)
    retrain = overall_drift or perf["performance_drift"]

    verdict = {
        "overall_drift": overall_drift,
        "retrain_recommended": retrain,
        "reason": (
            f"feature_drift_share={share:.2f}, perf_drift={perf['performance_drift']}"
        ),
        "window_rows": len(live),
        "model_version": model_version,
        "drifted_features": drifted,
        "warned_features": warned,
        "feature_drift_share": round(share, 4),
        "performance_drift": perf["performance_drift"],
        "live_f1": perf["live_f1"],
        "live_precision": perf["live_precision"],
        "live_recall": perf["live_recall"],
        "training_f1": training_f1,
        "feature_results": feature_results,
    }

    # Persist full report + summary row
    from src import db

    verdict["report_json"] = json.dumps(
        {k: v for k, v in verdict.items() if k != "report_json"}, default=str
    )
    db.insert_drift_report(verdict)
    return verdict


if __name__ == "__main__":
    import pprint

    pprint.pprint(run_drift_detection())
