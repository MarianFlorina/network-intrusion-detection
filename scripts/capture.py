"""CLI: capture your laptop's real traffic and score it with the production model.

Examples:
  python -m scripts.capture --check              # environment diagnostics
  python -m scripts.capture --list-interfaces    # pick an interface (index or name)
  python -m scripts.capture --window 30 --iface 4
  python -m scripts.capture --window 30 --iface "Wi-Fi"
  python -m scripts.capture --continuous --interval 60 --iface "Ethernet"

Windows: install Npcap (https://npcap.com, WinPcap-compatible mode) and run
the terminal as Administrator.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

# Hide scapy's third-party deprecation noise (cryptography DH warning etc.)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from cryptography.utils import CryptographyDeprecationWarning  # type: ignore

    warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)
except Exception:
    pass


from src.traffic_capture import (
    capture_forever,
    capture_window,
    check_environment,
    _friendly_interfaces,
)


def _quiet_scapy() -> None:
    """Best-effort suppression of scapy import-time warnings."""
    import logging

    logging.getLogger("scapy").setLevel(logging.ERROR)


def _resolve_iface(value: str | None) -> str | None:
    """Map a CLI --iface value (index, friendly name, or raw NPF name)."""
    if value is None:
        return None
    if value.isdigit():
        ifaces = _friendly_interfaces()
        try:
            return ifaces[int(value)]["name"]
        except (IndexError, ValueError):
            print(f"No interface with index {value}. Run --list-interfaces.")
            sys.exit(1)
    # friendly name (e.g. "Wi-Fi") or raw device name
    ifaces = _friendly_interfaces()
    for it in ifaces:
        if it["friendly"].lower() == value.lower() or it["name"] == value:
            return it["name"]
    print(f"No interface named {value!r}. Run --list-interfaces.")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live network traffic capture -> IDS scoring")
    parser.add_argument("--check", action="store_true", help="run environment diagnostics and exit")
    parser.add_argument("--list-interfaces", action="store_true", help="list capture interfaces")
    parser.add_argument(
        "--iface", type=str, default=None,
        help="interface index (from --list-interfaces), friendly name (e.g. Wi-Fi), or raw NPF name",
    )
    parser.add_argument("--window", type=float, default=30.0, help="seconds per capture window")
    parser.add_argument("--continuous", action="store_true", help="capture until Ctrl+C")
    parser.add_argument("--interval", type=float, default=60.0, help="window length in continuous mode")
    parser.add_argument("--no-persist", action="store_true", help="do not write predictions to the DB")
    args = parser.parse_args()

    os.environ.setdefault("MLFLOW_TRACKING_URI", "file:./mlruns")
    _quiet_scapy()

    if args.check:
        report = check_environment()
        print(json.dumps(report, indent=2, default=str))
        if report["ok"]:
            print("\nEnvironment OK for live capture.")
        else:
            print("\nFix the errors above, then re-run --check.")
            sys.exit(1)
        return

    if args.list_interfaces:
        for i, it in enumerate(_friendly_interfaces()):
            print(f"[{i}] {it['friendly']:<12} {it['description'][:55]:<55} {it['name'][:38]}")
        return

    env = check_environment()
    if not env["ok"]:
        print("Environment not ready. Run: python -m scripts.capture --check")
        for err in env["errors"]:
            print(f"  ERROR: {err}")
        sys.exit(1)

    iface = _resolve_iface(args.iface)

    if args.continuous:
        capture_forever(iface, interval=args.interval, persist=not args.no_persist)
    else:
        df = capture_window(iface, args.window)
        if df.empty:
            print("No flows captured (try a longer window or generate some traffic).")
            return
        from src.inference import score_batch
        from src.adaptation import adaptation_status, persist_capture_flows

        import time as _t

        batch_id = f"capture-{int(_t.time())}"
        # Persist flows for adaptation + drift reference (in addition to predictions)
        stored = persist_capture_flows(df, batch_id)
        result = score_batch(
            df, batch_id=batch_id, source="capture", persist=not args.no_persist,
        )
        status = adaptation_status()
        print(
            f"flows={result['scored_rows']:,} anomalies={result['anomalies']:,} "
            f"({result['anomaly_rate']:.2%}) model=v{result['model_version']}"
        )
        print(f"breakdown: {result['attack_breakdown']}")
        print(
            f"stored {stored:,} flows for adaptation "
            f"({status['capture_rows']:,}/{status['min_rows']:,} ready={status['ready']})"
        )
        if status["ready"]:
            print("-> run: .venv\\Scripts\\python -m scripts.adapt --run")


if __name__ == "__main__":
    main()
