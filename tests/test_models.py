from __future__ import annotations

import numpy as np
import pandas as pd

from src.models import build_model, is_unsupervised


def _fit_predict(name: str, fb_transform, df: pd.DataFrame, train_idx, test_idx):
    from src.feature_engineering import make_feature_builder

    fb = make_feature_builder(df.iloc[train_idx])
    X_train = fb.transform(df.iloc[train_idx])
    y_train = df["attack_type"].iloc[train_idx].to_numpy()
    model = build_model(name)
    model.fit(X_train, y_train)
    preds = model.predict(fb.transform(df.iloc[test_idx]))
    return preds, df["attack_type"].iloc[test_idx].to_numpy()


def test_random_forest_learns(small_flows):
    df = small_flows
    split = int(len(df) * 0.7)
    preds, y = _fit_predict("random_forest", None, df, range(split), range(split, len(df)))
    acc = float(np.mean(preds == y))
    assert acc > 0.80


def test_xgboost_learns(small_flows):
    df = small_flows
    split = int(len(df) * 0.7)
    preds, y = _fit_predict("xgboost", None, df, range(split), range(split, len(df)))
    acc = float(np.mean(preds == y))
    assert acc > 0.80


def test_unsupervised_wrappers_emit_class_labels(small_flows):
    df = small_flows
    split = int(len(df) * 0.7)
    train_idx, test_idx = list(range(split)), list(range(split, len(df)))
    for name in ["isolation_forest", "one_class_svm", "autoencoder"]:
        preds, y = _fit_predict(name, None, df, train_idx, test_idx)
        assert set(preds).issubset(set(y) | {"normal"})
        assert is_unsupervised(name)


def test_build_model_unknown_raises():
    import pytest

    with pytest.raises(ValueError):
        build_model("transformer")


def test_autoencoder_flags_anomalies(small_flows):
    from src.feature_engineering import add_engineered_features, make_feature_builder

    df = add_engineered_features(small_flows)
    split = int(len(df) * 0.7)
    fb = make_feature_builder(df.iloc[:split])
    model = build_model("autoencoder")  # wrapped -> emits class labels
    X_train = fb.transform(df.iloc[:split])
    y_train = df["attack_type"].iloc[:split].to_numpy()
    model.fit(X_train, y_train)
    preds = model.predict(fb.transform(df.iloc[split:]))
    assert set(np.unique(preds)).issubset(set(y_train) | {"normal"})
    # The inner detector must still flag some anomalies
    raw = model.detector.predict(fb.transform(df.iloc[split:]))
    assert set(np.unique(raw)).issubset({-1, 1})
    assert (raw == -1).sum() > 0
