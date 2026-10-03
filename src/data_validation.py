"""Data validation gate: schema, ranges, duplicates, label integrity, balance.

Runs as the first task of the training DAG. Raises on critical failures so
Airflow marks the run failed and the loop stops before bad data reaches
training. Summary statistics are returned for logging to MLflow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import settings

logger = logging.getLogger(__name__)

NUMERIC_FEATURES: list[str] = settings.get("training.numeric_features")
CATEGORICAL_FEATURES: list[str] = settings.get("training.categorical_features")
LABEL_COLUMN: str = settings.get("training.label_column")

ALLOWED_LABELS = {"normal", "ddos", "port_scan", "brute_force", "botnet"}

RANGE_RULES: dict[str, tuple[float, float]] = {
    "duration": (0.0, 600.0),
    "src_bytes": (0.0, 5e6),
    "dst_bytes": (0.0, 5e6),
    "count": (0.0, 5000.0),
    "srv_count": (0.0, 5000.0),
    "same_srv_rate": (0.0, 1.0),
    "diff_srv_rate": (0.0, 1.0),
    "src_port_entropy": (0.0, 10.0),
    "packet_rate": (0.0, 50000.0),
    "connection_duration_std": (0.0, 1000.0),
    "failed_logins": (0.0, 1000.0),
    "bytes_per_packet": (0.0, 10000.0),
}

VALID_PROTOCOLS = {"tcp", "udp", "icmp"}
VALID_SERVICES = {"http", "https", "dns", "smtp", "ssh", "ftp"}
VALID_FLAGS = {"SF", "S0", "REJ", "RSTO"}


@dataclass
class ValidationReport:
    passed: bool = True
    n_rows: int = 0
    n_duplicate_rows: int = 0
    n_null_cells: int = 0
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "validation_passed": int(self.passed),
            "rows": self.n_rows,
            "duplicate_rows": self.n_duplicate_rows,
            "null_cells": self.n_null_cells,
            "n_failures": len(self.failures),
            "n_warnings": len(self.warnings),
        }


def validate(df: pd.DataFrame, fail_on_error: bool = True) -> ValidationReport:
    report = ValidationReport()
    report.n_rows = len(df)

    # --- 1. Schema ---
    expected = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [LABEL_COLUMN]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        report.failures.append(f"missing_columns: {missing}")
    if missing and fail_on_error:
        report.passed = False
        raise ValueError(f"Schema validation failed, missing columns: {missing}")

    # --- 2. Nulls ---
    report.n_null_cells = int(df[expected].isna().sum().sum())
    if report.n_null_cells:
        null_cols = df[expected].columns[df[expected].isna().any()].tolist()
        report.warnings.append(f"null cells in {null_cols}; filled by preprocessing")
        df.dropna(subset=[LABEL_COLUMN], inplace=True)
        for c in NUMERIC_FEATURES:
            df[c] = df[c].fillna(df[c].median())
        for c in CATEGORICAL_FEATURES:
            df[c] = df[c].fillna("unknown")

    # --- 3. Duplicates ---
    report.n_duplicate_rows = int(df.duplicated().sum())
    if report.n_duplicate_rows:
        report.warnings.append(f"{report.n_duplicate_rows} duplicate rows dropped")
        df.drop_duplicates(inplace=True)
        report.n_rows = len(df)

    # --- 4. Numeric ranges ---
    for col, (lo, hi) in RANGE_RULES.items():
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        out_of_range = int(((values < lo) | (values > hi)).sum())
        if out_of_range:
            share = out_of_range / max(len(values), 1)
            if share > 0.05:
                report.failures.append(f"{col}: {share:.1%} of values outside [{lo}, {hi}]")
                report.passed = False
            else:
                report.warnings.append(f"{col}: {out_of_range} values clipped to range")
                df[col] = df[col].clip(lo, hi)

    # --- 5. Categorical domains ---
    domain_checks = {
        "protocol": VALID_PROTOCOLS,
        "service": VALID_SERVICES,
        "flag": VALID_FLAGS,
    }
    for col, allowed in domain_checks.items():
        if col not in df.columns:
            continue
        unknown = set(df[col].dropna().astype(str)) - allowed
        if unknown:
            report.warnings.append(f"{col}: unknown values {sorted(unknown)} -> 'other'")
            df[col] = df[col].where(df[col].isin(allowed), "other")

    # --- 6. Label integrity ---
    bad_labels = set(df[LABEL_COLUMN].astype(str)) - ALLOWED_LABELS
    if bad_labels:
        report.failures.append(f"unknown labels: {sorted(bad_labels)}")
        report.passed = False
        if fail_on_error:
            raise ValueError(f"Label validation failed: unknown labels {sorted(bad_labels)}")

    # --- 7. Class balance ---
    counts = df[LABEL_COLUMN].value_counts()
    if "normal" not in counts or counts.get("normal", 0) < 0.10 * max(len(df), 1):
        report.warnings.append("benign traffic below 10% — verify attack_rate")
    minority = counts.drop("normal", errors="ignore")
    if len(minority) == 0:
        report.failures.append("no attack rows present")
        report.passed = False
    elif (minority / max(len(df), 1) < 0.002).any():
        rare = minority[minority / max(len(df), 1) < 0.002]
        report.warnings.append(f"very rare classes (stratification may fail): {rare.index.tolist()}")

    if not report.passed and fail_on_error:
        raise ValueError("Data validation failed: " + "; ".join(report.failures))

    report.stats = {
        "label_counts": counts.to_dict(),
        "attack_rate": float(1.0 - counts.get("normal", 0) / max(len(df), 1)),
    }
    return report
