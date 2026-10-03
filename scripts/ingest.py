"""CLI: generate a synthetic network-flow batch.

Usage: python -m scripts.ingest --n 40000 --seed dvc-base --out data/raw/train.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_ingestion import generate_flows, save_raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic network flows")
    parser.add_argument("--n", type=int, default=40000)
    parser.add_argument("--seed", type=str, default="dvc-base")
    parser.add_argument("--attack-rate", type=float, default=None)
    parser.add_argument("--out", type=str, default="data/raw/train.csv")
    args = parser.parse_args()

    df = generate_flows(n=args.n, seed=args.seed, attack_rate=args.attack_rate)
    path = save_raw(df, Path(args.out).stem)
    print(f"wrote {len(df):,} rows -> {path}")


if __name__ == "__main__":
    main()
