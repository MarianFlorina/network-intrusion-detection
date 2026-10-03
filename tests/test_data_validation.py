from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_validation import validate


def test_validation_passes_on_clean_data(small_flows):
    report = validate(small_flows.copy(), fail_on_error=True)
    assert report.passed
    assert report.n_rows > 0
    assert report.stats["attack_rate"] > 0


def test_validation_catches_unknown_labels(small_flows):
    df = small_flows.copy()
    df.loc[df.index[:5], "attack_type"] = "zero_day"
    with pytest.raises(ValueError, match="unknown labels"):
        validate(df, fail_on_error=True)


def test_validation_clips_out_of_range_values(small_flows):
    df = small_flows.copy()
    df.loc[df.index[:3], "same_srv_rate"] = 1.5  # 3 rows = <5% -> clipped, warning
    report = validate(df, fail_on_error=False)
    assert (df["same_srv_rate"] <= 1.0).all()
    assert any("clipped" in w for w in report.warnings)


def test_validation_fails_when_range_breach_exceeds_threshold(small_flows):
    df = small_flows.copy()
    df["packet_rate"] = df["packet_rate"] * 1e6  # far out of range for all rows
    with pytest.raises(ValueError, match="outside"):
        validate(df, fail_on_error=True)


def test_validation_missing_column_raises(small_flows):
    df = small_flows.copy().drop(columns=["duration"])
    with pytest.raises(ValueError, match="missing columns"):
        validate(df, fail_on_error=True)


def test_validation_handles_nulls_and_duplicates():
    from src.data_ingestion import generate_flows

    df = generate_flows(n=800, seed="nulls")
    df.loc[df.index[:10], "src_bytes"] = np.nan
    df = pd.concat([df, df.head(20)], ignore_index=True)
    report = validate(df, fail_on_error=False)
    assert report.n_null_cells >= 10
    assert report.n_duplicate_rows == 20
    assert report.passed
