"""CLI: run the full training pipeline (same entry point the DAG uses).

Usage: python -m scripts.train --input data/raw/train.csv
Requires MLflow reachable at MLFLOW_TRACKING_URI (default http://localhost:5000).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.training import run_training_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train + register the IDS model")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--trigger", type=str, default="manual")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    results = run_training_pipeline(df, trigger=args.trigger)

    summary = {
        "candidate": results["candidate"]["model_name"],
        "candidate_test_f1": results["candidate"]["test_metrics"]["f1"],
        "promoted": results["promoted"],
        "new_model_version": results["new_model_version"],
        "pipeline_run_id": results["pipeline_run_id"],
        "all_models": {
            r["model_name"]: r["test_metrics"]["f1"] for r in results["all_results"]
        },
    }
    out = Path("data/metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
