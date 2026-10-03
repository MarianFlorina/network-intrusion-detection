"""Detached watcher: run `scripts.adapt --run` when the capture pool is ready.

Polls processed_records every `--interval` minutes; when
adaptation_status().ready is True, executes the adaptation cycle, writes a
result report to data/adaptation_result.json and exits.

Usage (detached):
  .venv/Scripts/python -m scripts.adapt_watch --interval 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path(".freebuff/adapt_watch.log")


def _log(message: str) -> None:
    """Write progress to file AND stdout (file survives detached execution)."""
    line = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait for adaptation readiness, then adapt")
    parser.add_argument("--interval", type=float, default=5.0, help="poll minutes")
    parser.add_argument("--timeout-hours", type=float, default=12.0, help="give up after this long")
    args = parser.parse_args()

    os.environ.setdefault("MLFLOW_TRACKING_URI", "file:./mlruns")
    logging.getLogger("mlflow").setLevel(logging.ERROR)

    from src.adaptation import adaptation_status, run_capture_adaptation

    _log(f"watcher started (pid={os.getpid()}, poll={args.interval}min)")
    deadline = time.time() + args.timeout_hours * 3600
    while time.time() < deadline:
        status = adaptation_status()
        _log(f"pool {status['capture_rows']}/{status['min_rows']} ready={status['ready']}")
        if status["ready"]:
            _log("THRESHOLD REACHED - running adaptation cycle...")
            result = run_capture_adaptation()
            report = {"finished_at": datetime.now(timezone.utc).isoformat(), **result}
            with open("data/adaptation_result.json", "w") as f:
                json.dump(report, f, indent=2, default=str)
            _log(f"adaptation finished: promoted={result.get('promoted')} "
                 f"version={result.get('new_model_version')}")
            return
        time.sleep(args.interval * 60)

    _log("timeout: pool never reached threshold")


if __name__ == "__main__":
    main()
