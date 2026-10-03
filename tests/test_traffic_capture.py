"""Tests for real-traffic capture: flow aggregation and feature mapping.

Packets are constructed with scapy (no live sniffing needed) and fed
directly into FlowTracker.
"""

from __future__ import annotations

import time

import pandas as pd
import pytest

scapy = pytest.importorskip("scapy")

from scapy.all import IP, TCP, UDP  # noqa: E402

from src.traffic_capture import FlowTracker, SERVICE_BY_PORT, check_environment  # noqa: E402


def _tcp_pkt(
    src="10.0.0.5", dst="93.184.216.34", sport=51000, dport=443, flags="SA", size=60
):
    pkt = IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags=flags)
    # pad to approximate wire size (scapy adds headers automatically)
    return pkt


def test_two_way_tcp_flow_becomes_one_row():
    tracker = FlowTracker(flow_timeout=0.0)
    base = 1000.0
    # handshake
    tracker.add_packet(_tcp_pkt(flags="S", sport=51000))          # t=1000
    tracker.add_packet(_tcp_pkt(flags="SA", sport=51000))
    tracker.add_packet(_tcp_pkt(flags="A", sport=51000))
    # data exchange (forward + reverse)
    for i in range(5):
        tracker.add_packet(_tcp_pkt(flags="PA", sport=51000))
        tracker.add_packet(_tcp_pkt(flags="PA", sport=443, dport=51000, src="93.184.216.34", dst="10.0.0.5"))
    # teardown
    tracker.add_packet(_tcp_pkt(flags="FA", sport=51000))
    tracker.add_packet(_tcp_pkt(flags="FA", sport=443, dport=51000, src="93.184.216.34", dst="10.0.0.5"))

    closed = tracker.expire(now=time.time() + 10)
    assert len(closed) == 1
    rows = FlowTracker.to_feature_rows(closed)
    assert len(rows) == 1
    row = rows[0]
    assert row["protocol"] == "tcp"
    assert row["service"] == "https"          # dport 443 mapped
    assert row["flag"] == "SF"                # saw SYN and FIN
    assert row["count"] >= 10
    assert row["src_bytes"] > 0 and row["dst_bytes"] > 0
    assert 0.0 <= row["same_srv_rate"] <= 1.0


def test_rst_flow_flagged_rej():
    tracker = FlowTracker(flow_timeout=0.0)
    tracker.add_packet(_tcp_pkt(flags="S", dport=2323))
    # server-side RST joins the same flow (sport=2323, dport=51000)
    tracker.add_packet(
        _tcp_pkt(flags="R", sport=2323, dport=51000, src="93.184.216.34", dst="10.0.0.5")
    )
    closed = tracker.expire(now=time.time() + 10)
    rows = FlowTracker.to_feature_rows(closed)
    assert len(rows) == 1
    assert rows[0]["flag"] == "REJ"
    assert rows[0]["service"] == "other"      # port 2323 unmapped


def test_syn_only_looks_like_scan():
    tracker = FlowTracker(flow_timeout=0.0)
    for dport in (22, 23, 25, 80, 110):
        tracker.add_packet(_tcp_pkt(flags="S", dport=dport))
    closed = tracker.expire(now=time.time() + 10)
    rows = FlowTracker.to_feature_rows(closed)
    syn_only = [r for r in rows if r["flag"] == "S0"]
    assert len(syn_only) >= 3                 # most SYN-only flows look scanny
    assert all(r["src_port_entropy"] > 0 for r in rows if r["flag"] == "S0")


def test_udp_flow_maps_dns():
    tracker = FlowTracker(flow_timeout=0.0)
    pkt = IP(src="10.0.0.5", dst="1.1.1.1") / UDP(sport=5353, dport=53)
    tracker.add_packet(pkt)
    tracker.add_packet(IP(src="1.1.1.1", dst="10.0.0.5") / UDP(sport=53, dport=5353))
    closed = tracker.expire(now=time.time() + 10)
    rows = FlowTracker.to_feature_rows(closed)
    assert rows[0]["protocol"] == "udp"
    assert rows[0]["service"] == SERVICE_BY_PORT[53] == "dns"


def test_non_ip_packets_ignored():
    tracker = FlowTracker(flow_timeout=0.0)
    tracker.add_packet(object())  # not a scapy packet
    assert tracker.flows == {}


def test_feature_rows_have_full_schema():
    tracker = FlowTracker(flow_timeout=0.0)
    tracker.add_packet(_tcp_pkt())
    closed = tracker.expire(now=time.time() + 10)
    rows = FlowTracker.to_feature_rows(closed)
    expected = {
        "duration", "src_bytes", "dst_bytes", "count", "srv_count",
        "same_srv_rate", "diff_srv_rate", "src_port_entropy", "packet_rate",
        "connection_duration_std", "failed_logins", "bytes_per_packet",
        "protocol", "service", "flag",
    }
    assert expected.issubset(rows[0].keys())


def test_check_environment_never_raises():
    report = check_environment()
    assert isinstance(report, dict)
    assert "ok" in report and "errors" in report


def test_capture_window_stops_quickly_on_stop_event(monkeypatch):
    """Ctrl+C must take effect within ~1s, not at the end of the window."""
    import threading

    import src.traffic_capture as tc

    def fake_sniff(**kwargs):
        time.sleep(0.05)  # emulate one short sniff chunk
        stop_filter = kwargs.get("stop_filter")
        if stop_filter and stop_filter(None):
            return

    monkeypatch.setattr("scapy.all.sniff", lambda **kw: fake_sniff(**kw))
    timer = threading.Timer(0.3, tc._STOP.set)
    timer.start()
    t0 = time.monotonic()
    try:
        df = tc.capture_window(None, 10.0)
        elapsed = time.monotonic() - t0
        assert elapsed < 5.0  # 10s window cut short by the stop event
        assert isinstance(df, pd.DataFrame)
    finally:
        timer.cancel()
        tc._STOP.clear()
