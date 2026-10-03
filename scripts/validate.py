"""CLI: run the data validation gate.

Usage: python -m scripts.validate --input data/raw/train.csv
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.data_validation import validate


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a flow dataset")
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    report = validate(df)
    print(json.dumps(report.summary(), indent=2))
    if report.failures:
        raise SystemExit(1)
    print("validation PASSED")


if __name__ == "__main__":
    main()
