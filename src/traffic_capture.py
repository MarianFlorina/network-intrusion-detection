"""Real traffic capture: scapy sniffer -> flow aggregator -> 16-feature schema.

Turns your laptop's actual packets into the same feature schema the model
was trained on, then streams them through the Production model into the
dashboard/database — the top box of the architecture diagram, made real.

Windows prerequisites:
  * Npcap driver   -> https://npcap.com  (install with "WinPcap API-compatible mode")
  * Administrator  -> packet sniffing requires elevated privileges

Usage:
  python -m scripts.capture --check              # environment diagnostics
  python -m scripts.capture --list-interfaces    # pick an interface name
  python -m scripts.capture --window 30          # capture 30s, score, persist
  python -m scripts.capture --continuous --interval 60
"""

from __future__ import annotations

import logging
import math
import os
import platform
import signal
import threading
import time
from dataclasses import dataclass, field

from src.config import settings

logger = logging.getLogger(__name__)

NUMERIC_FEATURES: list[str] = settings.get("training.numeric_features")
CATEGORICAL_FEATURES: list[str] = settings.get("training.categorical_features")

SERVICE_BY_PORT = {
    80: "http", 8080: "http", 8000: "http",
    443: "https", 8443: "https",
    53: "dns",
    25: "smtp", 587: "smtp", 465: "smtp",
    22: "ssh",
    21: "ftp", 20: "ftp",
}
WELL_KNOWN_MAX = 1024

# Map scapy/OS names -> friendly labels
IFACE_HINTS = {
    "wi-fi": "Wi-Fi", "wlan": "Wi-Fi", "wireless": "Wi-Fi", "802.11": "Wi-Fi",
    "wi-fi direct": "Virtual", "microsoft wi-fi direct": "Virtual",
    "ethernet": "Ethernet", "realtek pcie gbe": "Ethernet", "realtek": "Ethernet",
    "intel\u00ae ethernet": "Ethernet", "gigabit": "Ethernet", "killer": "Ethernet",
    "loopback": "Loopback", "bluetooth": "Bluetooth", "wan miniport": "Skip",
    "virtual": "Virtual", "vpn": "VPN", "tap": "VPN", "qos": "Skip",
    "npcap": "Skip", "microsoft km-test": "Skip", "lltdio": "Skip",
    "ndis": "Skip", "wfp": "Skip",
}


def _npf_device_count() -> int:
    """Count NPF (Npcap) devices visible through scapy."""
    try:
        from scapy.all import conf

        ifaces = getattr(conf, "ifaces", None)
        if not ifaces:
            return 0
        return sum(
            1 for data in list(ifaces.values())
            if "\\Device\\NPF_" in str(getattr(data, "name", "") or data)
            or "\\Device\\NPF_" in str(getattr(data, "network_name", "") or "")
        )
    except Exception:
        return 0


def _micro_sniff_ok() -> bool:
    """Definitive Npcap test: try a 1-second capture on the default iface."""
    try:
        from scapy.all import sniff

        sniff(count=1, timeout=1, store=False, filter="ip")
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Environment checks
# --------------------------------------------------------------------------- #

def check_environment() -> dict:
    """Diagnose everything needed for live capture. Never raises."""
    result = {
        "os": platform.system(),
        "scapy_installed": False,
        "npcap_available": False,
        "admin": False,
        "interfaces": [],
        "errors": [],
        "warnings": [],
        "ok": False,
    }

    try:
        import scapy  # noqa: F401

        result["scapy_installed"] = True
    except Exception as exc:
        result["errors"].append(
            f"scapy not importable: {exc}. Install with: pip install scapy"
        )
        return result

    if result["os"] != "Windows":
        result["warnings"].append(
            "Non-Windows host: libpcap is used instead of Npcap; root may be required."
        )
        result["npcap_available"] = True  # assume libpcap present; sniff() will tell us
    else:
        # Npcap detection for scapy >= 2.5: the presence of NPF (Npcap)
        # devices in conf.ifaces is the reliable signal. On failure, fall
        # back to a 1-second functional micro-sniff, which is definitive.
        npf_devices = _npf_device_count()
        if npf_devices > 0:
            result["npcap_available"] = True
        else:
            result["npcap_available"] = _micro_sniff_ok()

        if not result["npcap_available"]:
            result["errors"].append(
                "Npcap driver not detected (no NPF devices found). Install from "
                "https://npcap.com and enable 'Install Npcap in WinPcap API-compatible "
                "Mode' during setup, then reboot."
            )

    # Admin check (platform-appropriate)
    if result["os"] == "Windows":
        try:
            import ctypes

            result["admin"] = bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            result["admin"] = False
        if not result["admin"]:
            result["errors"].append(
                "Not running as Administrator. Packet capture needs elevation: "
                "restart the terminal with 'Run as administrator'."
            )
    else:
        result["admin"] = os.geteuid() == 0
        if not result["admin"]:
            result["warnings"].append("Not root: capture may fail or see only your own traffic.")

    try:
        result["interfaces"] = _friendly_interfaces()
    except Exception as exc:
        result["errors"].append(f"Could not enumerate interfaces: {exc}")

    result["ok"] = not result["errors"]
    return result


def _friendly_interfaces() -> list[dict]:
    """Return [{name, description, friendly}] for display/selection.

    Version-tolerant: scapy >= 2.5 exposes conf.ifaces everywhere;
    older/other platforms fall back to get_if_list / get_windows_if_list.
    """
    out: list[dict] = []

    # Strategy 1: conf.ifaces (works on all platforms, scapy >= 2.5)
    try:
        from scapy.all import conf

        for it in list(getattr(conf, "ifaces", {}).values()):
            name = str(getattr(it, "network_name", None) or getattr(it, "name", "") or it)
            desc = str(getattr(it, "description", "") or "")
            classified = _classify_iface(name, desc)
            if classified:
                out.append(classified)
        if out:
            return out
    except Exception:
        pass

    # Strategy 2 (Windows legacy): get_windows_if_list
    if platform.system() == "Windows":
        try:
            from scapy.arch.windows import get_windows_if_list

            for it in get_windows_if_list():
                classified = _classify_iface(
                    str(it.get("name", "")), str(it.get("description", ""))
                )
                if classified:
                    out.append(classified)
            return out
        except Exception:
            pass

    # Strategy 3: plain POSIX-style list
    from scapy.all import get_if_list

    for name in get_if_list():
        classified = _classify_iface(name, "")
        if classified:
            out.append(classified)
    return out


def _classify_iface(name: str, desc: str) -> dict | None:
    """Classify an interface; None = virtual/miniport noise, hide from user."""
    lowered = (name + " " + desc).lower()
    friendly = "Unknown"
    for key, label in IFACE_HINTS.items():
        if key in lowered:
            friendly = label
            break
    if friendly == "Skip":
        return None
    return {"name": name, "description": desc, "friendly": friendly}


# --------------------------------------------------------------------------- #
# Flow tracking
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class _FlowKey:
    """Directional canonical 5-tuple (src side = initiator).

    Frozen so it is hashable and can key the flows dict.
    """

    src: str
    dst: str
    sport: int
    dport: int
    proto: str

    def reverse(self) -> "_FlowKey":
        return _FlowKey(self.dst, self.src, self.dport, self.sport, self.proto)


@dataclass
class _FlowStats:
    packets: int = 0
    src_bytes: float = 0.0     # bytes initiator -> responder
    dst_bytes: float = 0.0     # bytes responder -> initiator
    start: float = 0.0
    last: float = 0.0
    syn: int = 0
    fin: int = 0
    rst: int = 0
    forward_pkts: int = 0      # initiator -> responder
    inter_arrivals: list = field(default_factory=list)


class FlowTracker:
    """Aggregates packets into bidirectional flows with timeout expiry.

    The initiator of each flow is the first endpoint seen, so src/dst byte
    direction stays consistent with KDD-style semantics.
    """

    def __init__(self, flow_timeout: float = 5.0, idle_expiry: float = 60.0):
        self.flow_timeout = flow_timeout       # seconds of inactivity before a flow "closes"
        self.idle_expiry = idle_expiry         # force-expire flows idle this long
        self.flows: dict[_FlowKey, _FlowStats] = {}
        self.closed: list[_FlowStats] = []

    # -- packet ingestion ---------------------------------------------------

    def add_packet(self, pkt) -> None:
        """Feed one scapy packet (IP + optional TCP/UDP layers required)."""
        from scapy.all import IP, TCP, UDP, ICMP

        try:
            if IP not in pkt:
                return
        except TypeError:
            return  # not a scapy packet; sniff() never sends us these
        ip = pkt[IP]
        now = float(pkt.time) if pkt.time else time.time()

        if TCP in pkt:
            key = _FlowKey(ip.src, ip.dst, int(pkt[TCP].sport), int(pkt[TCP].dport), "tcp")
        elif UDP in pkt:
            key = _FlowKey(ip.src, ip.dst, int(pkt[UDP].sport), int(pkt[UDP].dport), "udp")
        elif ICMP in pkt:
            key = _FlowKey(ip.src, ip.dst, 0, 0, "icmp")
        else:
            return

        # Canonical initiator: first endpoint we saw for this conversation
        canonical = next(
            (k for k in (key, key.reverse()) if k in self.flows), key
        )
        stats = self.flows.setdefault(
            canonical, _FlowStats(start=now, last=now)
        )
        is_forward = (key.src == canonical.src and key.sport == canonical.sport)
        if is_forward:
            stats.src_bytes += len(pkt)
            stats.forward_pkts += 1
        else:
            stats.dst_bytes += len(pkt)

        if stats.packets:
            stats.inter_arrivals.append(now - stats.last)
        stats.last = now
        stats.packets += 1

        if TCP in pkt:
            flags = int(pkt[TCP].flags)
            stats.syn += bool(flags & 0x02)
            stats.fin += bool(flags & 0x01)
            stats.rst += bool(flags & 0x04)

    # -- flow expiry ---------------------------------------------------------

    def expire(self, now: float | None = None) -> list[tuple[_FlowKey, _FlowStats]]:
        """Move timed-out flows to the closed list; return (key, stats) pairs."""
        now = now or time.time()
        newly_closed = []
        for key, stats in list(self.flows.items()):
            idle = now - stats.last
            tcp_done = stats.rst > 0 or (stats.syn >= 1 and stats.fin >= 1)
            if idle >= self.flow_timeout or tcp_done:
                newly_closed.append((key, stats))
                self.closed.append(stats)
                del self.flows[key]
            elif idle >= self.idle_expiry:
                del self.flows[key]  # discard ancient partial flows
        return newly_closed

    # -- feature extraction ---------------------------------------------------

    @staticmethod
    def _flow_flag(stats: _FlowStats) -> str:
        """KDD-style connection flag from TCP control-bit counts."""
        if stats.rst > 0:
            return "REJ" if stats.packets <= 2 else "RSTO"
        if stats.syn >= 1 and stats.fin >= 1:
            return "SF"       # normal establishment + teardown
        if stats.syn >= 1:
            return "S0"       # attempted, never established (scan signature)
        return "SF"

    @staticmethod
    def to_feature_rows(closed: list[tuple[_FlowKey, _FlowStats]]) -> list[dict]:
        """Convert closed (key, stats) flows into full training-schema rows."""
        rows = []
        for key, s in closed:
            duration = max(s.last - s.start, 1e-3)
            total_bytes = s.src_bytes + s.dst_bytes
            count = max(s.packets, 1)
            packet_rate = count / duration
            bytes_per_packet = total_bytes / count
            # Port-spread proxy: many SYNs from one host = port scan -> high.
            src_port_entropy = math.log1p(s.syn)
            if len(s.inter_arrivals) > 1:
                mean_ia = sum(s.inter_arrivals) / len(s.inter_arrivals)
                var_ia = sum((i - mean_ia) ** 2 for i in s.inter_arrivals) / len(s.inter_arrivals)
                conn_std = var_ia ** 0.5
            else:
                conn_std = 0.0
            failed_logins = float(min(s.rst, 12))
            rows.append(
                {
                    # numeric features (training schema)
                    "duration": duration,
                    "src_bytes": s.src_bytes,
                    "dst_bytes": s.dst_bytes,
                    "count": float(count),
                    "srv_count": float(count),
                    "same_srv_rate": 1.0,   # one service per flow in this aggregation
                    "diff_srv_rate": 0.0,
                    "src_port_entropy": src_port_entropy,
                    "packet_rate": packet_rate,
                    "connection_duration_std": conn_std,
                    "failed_logins": failed_logins,
                    "bytes_per_packet": bytes_per_packet,
                    # categorical features from the flow key + flags
                    "protocol": key.proto,
                    "service": SERVICE_BY_PORT.get(key.dport, "other"),
                    "flag": FlowTracker._flow_flag(s),
                }
            )
        return rows


# --------------------------------------------------------------------------- #
# Capture loop
# --------------------------------------------------------------------------- #

_STOP = threading.Event()


def request_stop(signum=None, frame=None) -> None:
    """Signal handler: request the capture loop to stop within ~1 second."""
    _STOP.set()


def install_stop_handlers() -> None:
    """Route Ctrl+C (and Ctrl+Break on Windows) into the stop event."""
    for sig in (signal.SIGINT, getattr(signal, "SIGBREAK", None)):
        if sig is None:
            continue
        try:
            signal.signal(sig, request_stop)
        except (ValueError, OSError):
            pass  # not on main thread / unsupported platform


def capture_window(
    interface: str | None,
    seconds: float,
    flow_timeout: float = 5.0,
    bpf_filter: str = "ip",
) -> "pd.DataFrame":  # noqa: F821
    """Sniff one window of real traffic and return flow-schema rows.

    Blocks for `seconds` (in 1-second chunks so Ctrl+C reacts fast), then
    expires open flows and converts them to the training schema. This is
    what the CLI and the Airflow live task call.
    """
    from scapy.all import sniff

    tracker = FlowTracker(flow_timeout=flow_timeout)

    def _feed(pkt) -> None:
        tracker.add_packet(pkt)

    logger.info("Capturing %.0fs on %s (filter=%r)...", seconds, interface or "default", bpf_filter)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and not _STOP.is_set():
        sniff(
            iface=interface,
            filter=bpf_filter,
            prn=_feed,
            timeout=min(1.0, deadline - time.monotonic()),
            store=False,
            stop_filter=lambda _pkt: _STOP.is_set(),
        )

    closed = tracker.expire(now=time.time() + flow_timeout)
    rows = FlowTracker.to_feature_rows(closed)
    logger.info("Closed %d flows -> %d feature rows", len(closed), len(rows))

    import pandas as pd

    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    return pd.DataFrame(rows, columns=cols)


def capture_forever(
    interface: str | None,
    interval: float = 60.0,
    flow_timeout: float = 5.0,
    bpf_filter: str = "ip",
    persist: bool = True,
):
    """Continuous capture: every `interval` seconds, score + persist a window.

    Mirrors the monitoring DAG's simulate_live_traffic task, but with real
    packets. Runs until Ctrl+C. Every window is also persisted as
    source='capture' so adaptation can pseudo-label it later.
    """
    from src.adaptation import adaptation_status, persist_capture_flows
    from src.inference import score_batch

    install_stop_handlers()
    window_no = 0
    while not _STOP.is_set():
        window_no += 1
        try:
            df = capture_window(interface, interval, flow_timeout, bpf_filter)
            if df.empty:
                print(f"[window {window_no}] no flows captured")
            else:
                batch_id = f"capture-{int(time.time())}"
                stored = persist_capture_flows(df, batch_id) if persist else 0
                result = score_batch(
                    df, batch_id=batch_id, source="capture", persist=persist
                )
                status = adaptation_status()
                print(
                    f"[window {window_no}] flows={result['scored_rows']:,} "
                    f"anomalies={result['anomalies']:,} ({result['anomaly_rate']:.2%}) "
                    f"model=v{result['model_version']} | adaptation pool "
                    f"{status['capture_rows']:,}/{status['min_rows']:,}"
                    + (" READY -> python -m scripts.adapt --run" if status["ready"] else "")
                )
            if _STOP.is_set():
                print("capture stopped by Ctrl+C")
                break
        except KeyboardInterrupt:
            print("capture stopped")
            break
        except Exception as exc:
            if _STOP.is_set():
                print("capture stopped by Ctrl+C")
                break
            logger.exception("capture window failed")
            print(f"[window {window_no}] error: {exc}")
            time.sleep(2)
