"""Local end-to-end demo (no Docker required).

1. Generates a synthetic training batch
2. Runs the full training pipeline with MLflow tracking (local file store)
3. Scores a fresh "live" batch through the registered Production model
4. Runs drift detection on a drifted batch and prints the verdict

Usage:
    python -m scripts.run_local
Requires only: pip install -r requirements.txt
(Mlflow tracking falls back to ./mlruns when no server is running.)
"""

from __future__ import annotations

import os


def main() -> None:
    os.environ.setdefault("MLFLOW_TRACKING_URI", "file:./mlruns")
    os.environ.pop("POSTGRES_HOST", None)  # force SQLite fallback

    import pandas as pd

    from src.config import settings
    from src.data_ingestion import generate_flows, save_raw
    from src.db import fetch_dataframe
    from src.drift_detection import run_drift_detection
    from src.inference import score_batch
    from src.training import run_training_pipeline

    print("=" * 70)
    print("STEP 1 - Generate training batch")
    print("=" * 70)
    df = generate_flows(n=30000, seed="local-demo")
    save_raw(df, "local-demo")
    print(df["attack_type"].value_counts())

    print("\n" + "=" * 70)
    print("STEP 2 - Train model zoo & register champion (MLflow)")
    print("=" * 70)
    results = run_training_pipeline(df, trigger="local-demo")
    for r in results["all_results"]:
        m = r["test_metrics"]
        print(
            f"  {r['model_name']:<18} macroF1={m['f1']:.4f}  anomalyF1={m['anomaly_f1']:.4f}"
            f"  P={m['precision']:.4f}  R={m['recall']:.4f}"
        )
    print(f"  -> promoted: {results['promoted']} (version {results['new_model_version']})")

    print("\n" + "=" * 70)
    print("STEP 3 - Score a fresh live batch with the Production model")
    print("=" * 70)
    live = generate_flows(n=8000, seed="live-fresh")
    save_raw(live, "live-fresh")
    score = score_batch(live, batch_id="live-fresh", source="live", persist=True)
    print(f"  scored={score['scored_rows']:,}  anomalies={score['anomalies']:,} "
          f"({score['anomaly_rate']:.2%})  model=v{score['model_version']}")
    print(f"  breakdown: {score['attack_breakdown']}")

    print("\n" + "=" * 70)
    print("STEP 4 - Drift detection on SHIFTED traffic (simulated attack wave)")
    print("=" * 70)
    drifted = generate_flows(
        n=8000, seed="live-drift",
        drift={"src_bytes_scale": 2.8, "packet_rate_shift": 1.8, "attack_rate": 0.12},
    )
    save_raw(drifted, "live-drift")
    score_batch(drifted, batch_id="live-drift", source="live", persist=True)
    verdict = run_drift_detection(live_window=drifted)
    print(f"  overall_drift={verdict['overall_drift']}  retrain_recommended={verdict['retrain_recommended']}")
    print(f"  drifted features: {verdict['drifted_features']}")
    print(f"  feature drift share: {verdict['feature_drift_share']}")

    print("\nDone. Explore experiments:  mlflow ui --backend-store-uri file:./mlruns")
    print("Dashboard:                  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
