"""NSL-KDD ingestion: map the labelled public dataset onto OUR flow schema.

NSL-KDD (Canadian Institute for Cybersecurity) is the classic labelled
network-intrusion benchmark: fixed 42-column CSV rows (41 features + label,
optionally + difficulty grade). This module turns it into the exact 16-column
schema the pipeline trains on, so the bootstrap corpus is REAL labelled data
instead of synthetic flows.

Column mapping (NSL-KDD -> ours):
  direct          : duration, src_bytes, dst_bytes, count, srv_count,
                    same_srv_rate, diff_srv_rate, num_failed_logins->failed_logins,
                    protocol_type->protocol (tcp/udp/icmp match exactly)
  folded domains  : service (70 -> http/https/dns/smtp/ssh/ftp/other),
                    flag (11 -> SF/S0/REJ/RSTO)
  derived proxy   : packet_rate = count / duration        (connection-rate proxy;
                    NSL-KDD has no per-packet data)
                    bytes_per_packet = (src_bytes+dst_bytes)/count
  not available   : src_port_entropy      -> constant log1p(1) (no training signal;
                    model relies on the other 11 numeric features)
                    connection_duration_std -> constant 0.0

Label mapping (NSL-KDD attack names -> our 5 classes):
  normal                      -> normal
  dos   (neptune, smurf, ...) -> ddos
  probe (satan, nmap, ...)    -> port_scan
  r2l   (guess_passwd, ...)   -> brute_force   (remote auth attacks)
  u2r   (buffer_overflow, ...)-> botnet        (host-compromise class)

Values outside the validation gate's ranges are clipped (printed as stats)
so the bootstrap passes the same gate as every other dataset.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --- NSL-KDD schema ---------------------------------------------------------- #

NSL_BASE_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]

# --- Label mapping ----------------------------------------------------------- #

DOS = {
    "back", "land", "neptune", "pod", "smurf", "teardrop", "mailbomb",
    "apache2", "processtable", "udpstorm", "worm",
}
PROBE = {"satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"}
R2L = {
    "guess_passwd", "ftp_write", "imap", "phf", "multihop", "warezclient",
    "warezmaster", "spy", "xlock", "xsnoop", "snmpguess", "snmpgetattack",
    "httptunnel", "sendmail", "named",
}
U2R = {
    "buffer_overflow", "loadmodule", "rootkit", "perl", "sqlattack", "xterm", "ps",
}

LABEL_MAP: dict[str, str] = {"normal": "normal"}
for _attack in DOS:
    LABEL_MAP[_attack] = "ddos"
for _attack in PROBE:
    LABEL_MAP[_attack] = "port_scan"
for _attack in R2L:
    LABEL_MAP[_attack] = "brute_force"
for _attack in U2R:
    LABEL_MAP[_attack] = "botnet"

# --- Categorical folding ------------------------------------------------------ #

SERVICE_MAP = {
    "http": "http", "http_2784": "http", "http_8001": "http",
    "http_443": "https",
    "domain": "dns", "domain_u": "dns",
    "smtp": "smtp", "pop_2": "smtp", "pop_3": "smtp", "imap4": "smtp",
    "ssh": "ssh",
    "ftp": "ftp", "ftp_data": "ftp",
    "telnet": "ssh", "klogin": "ssh", "kshell": "ssh", "login": "ssh",
    "shell": "ssh", "netbios_ssn": "ssh",  # remote-login family -> ssh
}
# everything else -> "other" (validation gate folds identically)

FLAG_MAP = {
    "SF": "SF", "S0": "S0", "REJ": "REJ", "RSTO": "RSTO",
    "RSTR": "RSTO", "RSTOS0": "RSTO",
    "S1": "S0", "S2": "S0", "S3": "S0", "SH": "S0",
    "OTH": "SF",
}

# Validation-gate range rules (must mirror src/data_validation.py)
_CLIP_RULES = {
    "duration": (0.0, 600.0),
    "src_bytes": (0.0, 5e6),
    "dst_bytes": (0.0, 5e6),
    "count": (0.0, 5000.0),
    "srv_count": (0.0, 5000.0),
    "packet_rate": (0.0, 50000.0),
    "bytes_per_packet": (0.0, 10000.0),
}

NEUTRAL_PORT_ENTROPY = math.log1p(1)  # matches a typical captured normal flow


def load_nslkdd_csv(path) -> pd.DataFrame:
    """Load a KDDTrain+/KDDTest+ style file (headerless; 42 or 43 columns)."""
    df = pd.read_csv(path, header=None)
    n = df.shape[1]
    if n == 43:  # includes difficulty grade
        df.columns = NSL_BASE_COLUMNS + ["label", "difficulty"]
    elif n == 42:
        df.columns = NSL_BASE_COLUMNS + ["label"]
    else:
        raise ValueError(
            f"Expected 42 or 43 columns (NSL-KDD), got {n} in {path}"
        )
    return df


def map_nslkdd(df: pd.DataFrame, min_class_count: int = 30) -> tuple[pd.DataFrame, dict]:
    """Map an NSL-KDD dataframe onto the project schema.

    Returns (mapped_df, stats). Rows with unknown labels or ultra-rare
    classes (< min_class_count, which would break stratified splitting)
    are dropped and reported.
    """
    stats: dict = {"rows_in": len(df)}

    out = pd.DataFrame(index=df.index)
    out["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0)
    out["src_bytes"] = pd.to_numeric(df["src_bytes"], errors="coerce").fillna(0.0)
    out["dst_bytes"] = pd.to_numeric(df["dst_bytes"], errors="coerce").fillna(0.0)
    out["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(1.0).clip(lower=1.0)
    out["srv_count"] = pd.to_numeric(df["srv_count"], errors="coerce").fillna(1.0).clip(lower=1.0)
    out["same_srv_rate"] = pd.to_numeric(df["same_srv_rate"], errors="coerce").fillna(0.0)
    out["diff_srv_rate"] = pd.to_numeric(df["diff_srv_rate"], errors="coerce").fillna(0.0)
    out["failed_logins"] = pd.to_numeric(df["num_failed_logins"], errors="coerce").fillna(0.0)

    # Derived proxies (documented at module head)
    out["packet_rate"] = out["count"] / out["duration"].clip(lower=1e-3)
    out["bytes_per_packet"] = (out["src_bytes"] + out["dst_bytes"]) / out["count"]
    out["src_port_entropy"] = NEUTRAL_PORT_ENTROPY
    out["connection_duration_std"] = 0.0

    out["protocol"] = (
        df["protocol_type"].astype(str).str.lower().where(
            lambda s: s.isin(["tcp", "udp", "icmp"]), "tcp"
        )
    )
    out["service"] = df["service"].astype(str).str.lower().map(SERVICE_MAP).fillna("other")
    out["flag"] = df["flag"].astype(str).str.upper().map(FLAG_MAP).fillna("SF")

    # Labels: map, drop unknowns
    raw_labels = df["label"].astype(str).str.strip().str.lower()
    mapped = raw_labels.map(LABEL_MAP)
    stats["unknown_labels_dropped"] = int(mapped.isna().sum())
    out["attack_type"] = mapped

    out = out[mapped.notna()].copy()

    # Drop ultra-rare classes (would break stratified train/test split)
    counts = out["attack_type"].value_counts()
    rare = counts[counts < min_class_count].index.tolist()
    if rare:
        stats["rare_classes_dropped"] = {c: int(counts[c]) for c in rare}
        out = out[~out["attack_type"].isin(rare)]

    # Clip to validation-gate ranges; report how much was clipped
    stats["clipped"] = {}
    for col, (lo, hi) in _CLIP_RULES.items():
        n_clipped = int(((out[col] < lo) | (out[col] > hi)).sum())
        if n_clipped:
            stats["clipped"][col] = n_clipped
            out[col] = out[col].clip(lo, hi)

    out = out.reset_index(drop=True)
    stats["rows_out"] = len(out)
    stats["label_counts"] = out["attack_type"].value_counts().to_dict()
    return out, stats


def ingest(
    train_path,
    out_path,
    test_path=None,
    test_out=None,
    max_rows: int | None = None,
    min_class_count: int = 30,
) -> dict:
    """Full ingest: load -> map -> optional downsample -> write CSV(s)."""
    df = load_nslkdd_csv(train_path)
    mapped, stats = map_nslkdd(df, min_class_count=min_class_count)

    if max_rows and len(mapped) > max_rows:
        # stratified-ish downsample preserving class ratios
        frac = max_rows / len(mapped)
        mapped = (
            mapped.groupby("attack_type", group_keys=False)
            .apply(lambda g: g.sample(frac=frac, random_state=42))
            .reset_index(drop=True)
        )
        stats["downsampled_to"] = len(mapped)

    from pathlib import Path

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mapped.to_csv(out_path, index=False)

    if test_path:
        test_df = load_nslkdd_csv(test_path)
        test_mapped, test_stats = map_nslkdd(test_df, min_class_count=min_class_count)
        test_out = Path(test_out or out_path.with_name(out_path.stem + "_test.csv"))
        test_mapped.to_csv(test_out, index=False)
        stats["test_rows_out"] = test_stats["rows_out"]
        stats["test_path"] = str(test_out)

    stats["train_path"] = str(out_path)
    return stats
