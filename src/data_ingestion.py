"""Synthetic network-flow generator + raw CSV ingestion.

Produces traffic shaped like Netflow/KDD-Cup style flows with realistic
attack patterns (DDoS, port scan, brute force, botnet, benign traffic).
In a real deployment, replace `generate_flows` with a loader that reads
your network logs — the schema is documented in README.md.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import settings

NUMERIC_FEATURES: list[str] = settings.get("training.numeric_features")
CATEGORICAL_FEATURES: list[str] = settings.get("training.categorical_features")
LABEL_COLUMN: str = settings.get("training.label_column")

ATTACK_TYPES = ["normal", "ddos", "port_scan", "brute_force", "botnet"]

BASE_RATE = 0.955  # share of benign flows in a healthy network


def _rng(seed_text: str) -> np.random.Generator:
    seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def _flows(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """Draw one benign traffic cluster."""
    return {
        "duration": rng.lognormal(0.4, 1.0, n).clip(0, 60),
        "src_bytes": rng.lognormal(6.0, 1.0, n).clip(40, 150000),
        "dst_bytes": rng.lognormal(7.0, 1.1, n).clip(40, 300000),
        "count": rng.poisson(6, n).astype(float).clip(0, 200),
        "srv_count": rng.poisson(4, n).astype(float).clip(0, 150),
        "same_srv_rate": rng.beta(6, 2, n),
        "diff_srv_rate": rng.beta(1.5, 6, n),
        "src_port_entropy": rng.normal(3.2, 0.5, n).clip(0, 6),
        "packet_rate": rng.lognormal(2.2, 0.6, n).clip(1, 3000),
        "connection_duration_std": rng.lognormal(-0.5, 0.8, n).clip(0, 100),
        "failed_logins": rng.choice([0, 0, 0, 1], n).astype(float),
        "bytes_per_packet": rng.normal(500, 180, n).clip(20, 1500),
        "protocol": rng.choice(["tcp", "udp", "icmp"], n, p=[0.75, 0.22, 0.03]),
        "service": rng.choice(
            ["http", "https", "dns", "smtp", "ssh", "ftp"], n, p=[0.42, 0.33, 0.12, 0.06, 0.05, 0.02]
        ),
        "flag": rng.choice(["SF", "S0", "REJ", "RSTO"], n, p=[0.88, 0.06, 0.04, 0.02]),
    }


def _attack(kind: str, rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    base = _flows(rng, n)
    if kind == "ddos":
        # volumetric: tiny durations, huge packet rates, thousands of connections
        base["duration"] = rng.lognormal(-1.5, 0.4, n).clip(0, 2)
        base["src_bytes"] = rng.lognormal(9.0, 0.5, n).clip(1000, 1e6)
        base["dst_bytes"] = rng.lognormal(3.0, 0.6, n).clip(20, 5000)
        base["count"] = rng.poisson(400, n).astype(float).clip(50, 2000)
        base["srv_count"] = rng.poisson(350, n).astype(float).clip(50, 1500)
        base["same_srv_rate"] = rng.beta(30, 2, n)
        base["packet_rate"] = rng.lognormal(6.2, 0.5, n).clip(500, 20000)
        base["bytes_per_packet"] = rng.normal(1400, 60, n).clip(900, 1600)
        base["protocol"] = rng.choice(["tcp", "udp"], n, p=[0.7, 0.3])
        base["service"] = rng.choice(["http", "https"], n, p=[0.8, 0.2])
        base["flag"] = rng.choice(["S0", "SF"], n, p=[0.6, 0.4])
    elif kind == "port_scan":
        base["duration"] = rng.lognormal(-2.0, 0.3, n).clip(0, 0.5)
        base["src_bytes"] = rng.lognormal(2.5, 0.4, n).clip(20, 300)
        base["dst_bytes"] = rng.lognormal(1.5, 0.4, n).clip(10, 200)
        base["count"] = rng.poisson(120, n).astype(float).clip(30, 600)
        base["srv_count"] = rng.poisson(15, n).astype(float).clip(1, 80)
        base["same_srv_rate"] = rng.beta(1.2, 8, n)
        base["diff_srv_rate"] = rng.beta(12, 2, n)
        base["src_port_entropy"] = rng.normal(5.2, 0.4, n).clip(3.5, 6)
        base["packet_rate"] = rng.lognormal(3.0, 0.5, n).clip(20, 2000)
        base["service"] = rng.choice(
            ["http", "ssh", "ftp", "smtp", "dns"], n, p=[0.3, 0.25, 0.2, 0.15, 0.1]
        )
        base["flag"] = rng.choice(["S0", "REJ", "RSTO"], n, p=[0.7, 0.2, 0.1])
    elif kind == "brute_force":
        base["duration"] = rng.lognormal(1.2, 0.6, n).clip(0.1, 30)
        base["failed_logins"] = rng.choice([3.0, 5.0, 8.0, 12.0], n, p=[0.3, 0.3, 0.25, 0.15])
        base["srv_count"] = rng.poisson(30, n).astype(float).clip(5, 120)
        base["same_srv_rate"] = rng.beta(20, 2, n)
        base["bytes_per_packet"] = rng.normal(180, 60, n).clip(40, 500)
        base["packet_rate"] = rng.lognormal(3.6, 0.5, n).clip(30, 3000)
        base["service"] = np.full(n, "ssh")
        base["flag"] = rng.choice(["REJ", "S0", "SF"], n, p=[0.5, 0.3, 0.2])
    elif kind == "botnet":
        # beaconing: periodic small flows, odd entropy, long-lived C2 sessions
        base["duration"] = rng.lognormal(2.2, 0.8, n).clip(1, 120)
        base["src_bytes"] = rng.lognormal(4.0, 0.6, n).clip(60, 20000)
        base["dst_bytes"] = rng.lognormal(3.6, 0.6, n).clip(60, 20000)
        base["src_port_entropy"] = rng.normal(4.6, 0.5, n).clip(2, 6)
        base["packet_rate"] = rng.lognormal(1.6, 0.5, n).clip(1, 300)
        base["connection_duration_std"] = rng.lognormal(2.0, 0.7, n).clip(1, 200)
        base["same_srv_rate"] = rng.beta(14, 3, n)
        base["protocol"] = rng.choice(["tcp", "udp"], n, p=[0.85, 0.15])
        base["service"] = rng.choice(["dns", "http", "https"], n, p=[0.4, 0.35, 0.25])
        base["flag"] = rng.choice(["SF", "RSTO"], n, p=[0.75, 0.25])
    return base


def generate_flows(
    n: int = 20000, seed: str = "base", attack_rate: float | None = None, drift: dict | None = None
) -> pd.DataFrame:
    """Generate a labelled flow dataset.

    `seed` controls reproducible variety between batches.
    `attack_rate` overrides the benign/attack mix (e.g. simulate an attack wave).
    `drift` shifts benign traffic statistics, e.g. {"attack_rate": 0.10,
    "src_bytes_scale": 2.5, "packet_rate_shift": 1.5, "service_weights": {...}}.
    """
    rng = _rng(seed)
    rate = BASE_RATE if attack_rate is None else 1.0 - attack_rate
    if drift and "attack_rate" in drift:
        rate = 1.0 - float(drift["attack_rate"])
    n_attack = int(round(n * (1.0 - rate)))
    n_normal = n - n_attack

    parts: list[pd.DataFrame] = []
    if n_normal:
        parts.append(pd.DataFrame(_flows(rng, n_normal)).assign(attack_type="normal"))
    if n_attack:
        kinds = rng.choice(ATTACK_TYPES[1:], n_attack)
        for kind in ATTACK_TYPES[1:]:
            k = int((kinds == kind).sum())
            if k:
                parts.append(pd.DataFrame(_attack(kind, rng, k)).assign(attack_type=kind))

    df = pd.concat(parts, ignore_index=True)
    df = df.sample(frac=1.0, random_state=rng.integers(0, 2**31)).reset_index(drop=True)

    # ---- benign-drift shifts (applied AFTER labelling, to benign rows only) ----
    if drift:
        benign = df["attack_type"] == "normal"
        if "src_bytes_scale" in drift:
            df.loc[benign, "src_bytes"] = (df.loc[benign, "src_bytes"] * drift["src_bytes_scale"]).clip(40, 2e6)
        if "dst_bytes_scale" in drift:
            df.loc[benign, "dst_bytes"] = (df.loc[benign, "dst_bytes"] * drift["dst_bytes_scale"]).clip(40, 2e6)
        if "packet_rate_shift" in drift:
            df.loc[benign, "packet_rate"] = (df.loc[benign, "packet_rate"] * drift["packet_rate_shift"]).clip(1, 20000)
        if "duration_scale" in drift:
            df.loc[benign, "duration"] = (df.loc[benign, "duration"] * drift["duration_scale"]).clip(0, 300)
        if "service_weights" in drift and benign.any():
            weights = drift["service_weights"]
            services = list(weights)
            probs = [weights[s] for s in services]
            df.loc[benign, "service"] = rng.choice(services, int(benign.sum()), p=probs)

    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [LABEL_COLUMN]]


def load_raw_csv(path: str | Path) -> pd.DataFrame:
    """Load an external CSV (e.g. exported network logs) with the expected schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw CSV not found: {path}. Expected columns: "
            f"{NUMERIC_FEATURES + CATEGORICAL_FEATURES + [LABEL_COLUMN]}"
        )
    return pd.read_csv(path)


def save_raw(df: pd.DataFrame, batch_id: str) -> Path:
    out = settings.DATA_DIR / "raw" / f"{batch_id}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out
