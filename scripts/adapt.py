"""CLI: adapt the production model to YOUR real network traffic.

Captured flows are pseudo-labelled by the current champion (confidence-
filtered self-training), blended with synthetic attack anchors, and run
through the normal promotion gate. Registry promotion happens only if the
adapted model genuinely beats the current champion.

Usage:
  python -m scripts.adapt --status     # captured rows vs required
  python -m scripts.adapt --run        # run adaptation now (gated)
"""

from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt model to real captured traffic")
    parser.add_argument("--status", action="store_true", help="show adaptation readiness")
    parser.add_argument("--run", action="store_true", help="run adaptation (gated pipeline)")
    args = parser.parse_args()

    os.environ.setdefault("MLFLOW_TRACKING_URI", "file:./mlruns")

    from src.adaptation import adaptation_status, run_capture_adaptation

    if args.status:
        print(json.dumps(adaptation_status(), indent=2))
        return

    if args.run:
        print("Running adaptation (pseudo-labelling + gated retrain)...")
        result = run_capture_adaptation()
        print(json.dumps(result, indent=2, default=str))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
