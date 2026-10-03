from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_engineering import (
    add_engineered_features,
    make_feature_builder,
)


def test_engineered_features_idempotent(small_flows):
    once = add_engineered_features(small_flows)
    twice = add_engineered_features(once)
    assert "bytes_ratio" in once.columns
    pd.testing.assert_frame_equal(once, twice)


def test_bytes_ratio_math(small_flows):
    df = add_engineered_features(small_flows)
    expected = df["src_bytes"] / (df["dst_bytes"] + 1e-6)
    assert np.allclose(df["bytes_ratio"], expected)


def test_feature_builder_transform_shape(small_flows):
    df = add_engineered_features(small_flows)
    split = int(len(df) * 0.8)
    fb = make_feature_builder(df.iloc[:split])
    X = fb.transform(df.iloc[split:])
    assert X.shape[0] == len(df) - split
    assert X.shape[1] > 12  # numeric + one-hot expansion
    assert np.isfinite(X).all()


def test_feature_builder_survives_roundtrip(small_flows, tmp_path):
    from src.feature_engineering import save_feature_builder, load_feature_builder

    df = add_engineered_features(small_flows)
    fb = make_feature_builder(df.iloc[:2000])
    path = tmp_path / "fb.joblib"
    save_feature_builder(fb, path)
    fb2 = load_feature_builder(path)
    a = fb.transform(df.iloc[2000:2100])
    b = fb2.transform(df.iloc[2000:2100])
    assert np.allclose(a, b)
