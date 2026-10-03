"""CLI: ingest NSL-KDD into the project's flow schema (real labelled bootstrap).

1. Download the dataset (any mirror), e.g.:
   https://www.unb.ca/cic/datasets/nsl-kdd.html  (KDDTrain+.txt / KDDTest+.txt)

2. Run:
   python -m scripts.ingest_nslkdd --train KDDTrain+.txt --test KDDTest+.txt

Outputs data/raw/nslkdd_train.csv (+ _test.csv) in the 16-column project
schema, ready for the standard pipeline:

   python -m scripts.validate --input data/raw/nslkdd_train.csv
   python -m scripts.train --input data/raw/nslkdd_train.csv
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Map NSL-KDD onto the project flow schema")
    parser.add_argument("--train", required=True, help="path to KDDTrain+.txt")
    parser.add_argument("--test", default=None, help="path to KDDTest+.txt (optional)")
    parser.add_argument("--out", default="data/raw/nslkdd_train.csv")
    parser.add_argument("--max-rows", type=int, default=None, help="stratified downsample")
    parser.add_argument(
        "--min-class-count", type=int, default=30,
        help="drop classes rarer than this (keeps stratified splitting valid)",
    )
    args = parser.parse_args()

    from src.nslkdd_ingest import ingest

    stats = ingest(
        args.train,
        args.out,
        test_path=args.test,
        max_rows=args.max_rows,
        min_class_count=args.min_class_count,
    )
    print(json.dumps(stats, indent=2, default=str))
    print(f"\nWrote {stats['rows_out']:,} labelled real-traffic rows -> {stats['train_path']}")
    print("Next: python -m scripts.train --input", stats["train_path"])


if __name__ == "__main__":
    main()
