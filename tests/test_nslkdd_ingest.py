"""Tests for NSL-KDD -> project schema mapping (constructed sample, no download)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.nslkdd_ingest import (
    LABEL_MAP,
    NSL_BASE_COLUMNS,
    load_nslkdd_csv,
    map_nslkdd,
)


def _nsl_row(**overrides) -> dict:
    """A plausible NSL-KDD row: normal http flow, then override per test."""
    row = {c: 0 for c in NSL_BASE_COLUMNS}
    row.update(
        {
            "duration": 12,
            "protocol_type": "tcp",
            "service": "http",
            "flag": "SF",
            "src_bytes": 4216,
            "dst_bytes": 10864,
            "count": 5,
            "srv_count": 5,
            "same_srv_rate": 1.0,
            "diff_srv_rate": 0.0,
            "num_failed_logins": 0,
            "label": "normal",
        }
    )
    row.update(overrides)
    return row


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=NSL_BASE_COLUMNS + ["label"])


def test_label_map_covers_all_nslkdd_attacks():
    """The official 22 train attacks + representative test-only attacks map."""
    train_attacks = {
        "back", "land", "neptune", "pod", "smurf", "teardrop",  # DoS
        "satan", "ipsweep", "nmap", "portsweep",                # Probe
        "guess_passwd", "ftp_write", "imap", "phf", "multihop",
        "warezclient", "warezmaster", "spy",                    # R2L
        "buffer_overflow", "loadmodule", "rootkit", "perl",     # U2R
        "normal",
    }
    missing = train_attacks - set(LABEL_MAP)
    assert not missing, f"unmapped labels: {missing}"


def test_map_direct_columns():
    df = map_nslkdd(_frame([_nsl_row()]), min_class_count=1)[0]
    row = df.iloc[0]
    assert row["duration"] == 12.0
    assert row["src_bytes"] == 4216.0
    assert row["dst_bytes"] == 10864.0
    assert row["count"] == 5.0
    assert row["failed_logins"] == 0.0
    assert row["protocol"] == "tcp"


def test_map_derived_proxies():
    df = map_nslkdd(_frame([_nsl_row()]), min_class_count=1)[0]
    row = df.iloc[0]
    assert row["packet_rate"] == pytest.approx(5.0 / 12.0, rel=1e-6)
    assert row["bytes_per_packet"] == pytest.approx((4216 + 10864) / 5, rel=1e-6)
    assert row["src_port_entropy"] > 0  # neutral constant, matches captured flows
    assert row["connection_duration_std"] == 0.0


def test_service_and_flag_folding():
    rows = [
        _nsl_row(service="domain_u", label="normal"),
        _nsl_row(service="http_443", label="normal"),
        _nsl_row(service="some_weird", label="normal"),
        _nsl_row(flag="RSTR", label="normal"),
        _nsl_row(flag="S3", label="normal"),
    ]
    df = map_nslkdd(_frame(rows), min_class_count=1)[0]
    # some_weird -> other; flag-override rows keep base service http
    assert sorted(df["service"]) == ["dns", "http", "http", "https", "other"]
    assert sorted(df["flag"]) == ["RSTO", "S0", "SF", "SF", "SF"]


def test_label_grouping_to_five_classes():
    rows = [
        _nsl_row(label="neptune"),     # dos
        _nsl_row(label="smurf"),       # dos
        _nsl_row(label="satan"),       # probe
        _nsl_row(label="portsweep"),   # probe
        _nsl_row(label="guess_passwd"),# r2l
        _nsl_row(label="warezclient"), # r2l
        _nsl_row(label="buffer_overflow"),  # u2r
        _nsl_row(label="normal"),
        _nsl_row(label="normal"),
        _nsl_row(label="normal"),
    ]
    df = map_nslkdd(_frame(rows), min_class_count=1)[0]
    assert set(df["attack_type"]) == {"ddos", "port_scan", "brute_force", "botnet", "normal"}


def test_unknown_label_dropped_and_reported():
    rows = [_nsl_row(label="normal"), _nsl_row(label="zero_day_xyz")]
    out, stats = map_nslkdd(_frame(rows), min_class_count=1)
    assert len(out) == 1
    assert stats["unknown_labels_dropped"] == 1


def test_rare_class_dropped_with_min_count():
    rows = [_nsl_row(label="normal")] * 50 + [_nsl_row(label="perl")]  # 1 u2r row
    out, stats = map_nslkdd(_frame(rows), min_class_count=30)
    assert "botnet" not in set(out["attack_type"])
    assert stats["rare_classes_dropped"] == {"botnet": 1}


def test_out_of_range_values_clipped_and_reported():
    rows = [_nsl_row(duration=999999, src_bytes=99_000_000, label="normal")] * 10
    out, stats = map_nslkdd(_frame(rows), min_class_count=1)
    assert (out["duration"] <= 600.0).all()
    assert (out["src_bytes"] <= 5e6).all()
    # packet_rate derives from duration (999999s -> tiny rate), so only
    # duration, src_bytes and bytes_per_packet breach the gate here
    assert set(stats["clipped"]) == {"duration", "src_bytes", "bytes_per_packet"}


def test_load_headerless_42_and_43_columns(tmp_path):
    base = _frame([_nsl_row()])
    p42 = tmp_path / "t42.csv"
    base.to_csv(p42, index=False, header=False)
    loaded = load_nslkdd_csv(p42)
    assert list(loaded.columns) == NSL_BASE_COLUMNS + ["label"]

    p43 = tmp_path / "t43.csv"
    base.assign(difficulty=21).to_csv(p43, index=False, header=False)
    loaded43 = load_nslkdd_csv(p43)
    assert "difficulty" in loaded43.columns
