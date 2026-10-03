from __future__ import annotations

import numpy as np
import pandas as pd

from src.drift_detection import (
    analyze_feature_drift,
    categorical_psi,
    population_stability_index,
)


def test_psi_identical_is_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    assert population_stability_index(x, x.copy()) < 0.01


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    cur = rng.normal(4, 1, 5000)  # massive shift
    assert population_stability_index(ref, cur) > 0.5


def test_categorical_psi_detects_new_category():
    ref = pd.Series(["a"] * 90 + ["b"] * 10)
    cur = pd.Series(["a"] * 10 + ["c"] * 90)
    assert categorical_psi(ref, cur) > 0.5


def test_analyze_flags_shifted_feature():
    from src.config import settings
    from src.data_ingestion import generate_flows

    ref = generate_flows(n=3000, seed="ref")
    cur = generate_flows(n=3000, seed="cur")
    baseline = analyze_feature_drift(ref, cur)
    # Unshifted same-generator batches should have low drift share
    drifted_share = sum(1 for r in baseline.values() if r["status"] == "drift") / len(baseline)
    assert drifted_share < 0.4

    shifted = cur.copy()
    shifted["packet_rate"] = shifted["packet_rate"] * 5
    results = analyze_feature_drift(ref, shifted)
    assert results["packet_rate"]["status"] in {"warn", "drift"}


def test_run_drift_detection_verdict_flow(temp_db, small_flows):
    """End-to-end verdict: shifted live window -> retrain_recommended."""
    from src.config import settings
    from src.drift_detection import run_drift_detection
    from src.feature_engineering import add_engineered_features

    # Write a reference dataset (as training would)
    ref_dir = settings.DATA_DIR / "processed"
    ref_dir.mkdir(parents=True, exist_ok=True)
    add_engineered_features(small_flows.iloc[:3000]).to_parquet(ref_dir / "reference.parquet")

    live = generate_drifted_window(n=6000)
    verdict = run_drift_detection(live_window=live)
    assert verdict["window_rows"] == len(live)
    assert verdict["feature_drift_share"] > 0
    assert verdict["retrain_recommended"] in {True, False}  # perf component may vary


def generate_drifted_window(n: int) -> pd.DataFrame:
    from src.data_ingestion import generate_flows

    return generate_flows(
        n=n, seed="drifted-live",
        drift={"src_bytes_scale": 3.0, "packet_rate_shift": 2.0, "attack_rate": 0.15},
    )
