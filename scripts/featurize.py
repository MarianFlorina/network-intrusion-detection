"""CLI: feature engineering pass (persisted for DVC versioning).

Usage: python -m scripts.featurize --input data/raw/train.csv --out data/processed/train.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.feature_engineering import add_engineered_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Engineer features")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--out", type=str, default="data/processed/train.parquet")
    args = parser.parse_args()

    df = add_engineered_features(pd.read_csv(args.input))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {len(df):,} rows -> {out}")


if __name__ == "__main__":
    main()
