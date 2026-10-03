"""Tests for capture adaptation: persistence, status, pseudo-label filtering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.adaptation import (
    adaptation_status,
    build_adapted_corpus,
    persist_capture_flows,
    pseudo_label,
)
from src.config import _Settings, settings


def _capture_frame(n: int = 500) -> pd.DataFrame:
    """Unlabelled capture rows (all 15 feature columns, no attack_type)."""
    from src.data_ingestion import generate_flows

    df = generate_flows(n=n, seed="adapt-test").drop(columns=["attack_type"])
    return df


def test_persist_and_status(temp_db):
    stored = persist_capture_flows(_capture_frame(300), "cap-test-1")
    assert stored == 300
    status = adaptation_status()
    assert status["capture_rows"] >= 300
    assert "ready" in status and "min_rows" in status


def test_pseudo_label_filters_by_confidence(temp_db, small_flows, monkeypatch):
    """Stub the production model: confident rows kept, uncertain dropped."""
    from src.inference import ProductionModel

    class StubModel:
        version = "9"
        run_id = "stub"

        def predict_df(self, df):
            n = len(df)
            labels = np.array(["normal"] * n, dtype=object)
            labels[: n // 2] = "ddos"
            confidence = np.concatenate(
                [np.full(n // 2, 0.97), np.full(n - n // 2, 0.40)]
            )
            return labels, 1.0 - confidence, confidence

    monkeypatch.setattr(
        "src.inference.get_production_model", lambda stage="Production": StubModel()
    )
    kept, info = pseudo_label(_capture_frame(200), min_confidence=0.85)
    assert info["captured"] == 200
    assert info["confident"] == 100
    assert len(kept) == 100
    assert set(kept["attack_type"]).issubset({"normal", "ddos"})


def test_build_adapted_corpus_insufficient(temp_db, monkeypatch):
    monkeypatch.setitem(_Settings.raw, "adaptation", {"min_rows": 10 ** 9})
    corpus, info = build_adapted_corpus()
    assert corpus is None
    assert "reason" in info


def test_build_adapted_corpus_blends_anchors(temp_db, small_flows, monkeypatch):
    """Enough capture + confident stub -> corpus with anchors + pseudo rows."""
    persist_capture_flows(_capture_frame(600), "cap-test-2")

    class StubModel:
        version = "9"
        run_id = "stub"

        def predict_df(self, df):
            n = len(df)
            rng = np.random.default_rng(0)
            labels = rng.choice(["normal", "ddos", "port_scan"], n, p=[0.9, 0.05, 0.05])
            confidence = np.full(n, 0.95)
            return labels, 1.0 - confidence, confidence

    monkeypatch.setattr(
        "src.inference.get_production_model", lambda stage="Production": StubModel()
    )
    monkeypatch.setitem(
        _Settings.raw,
        "adaptation",
        {"min_rows": 300, "min_confidence": 0.85, "anchor_rows": 1000},
    )
    corpus, info = build_adapted_corpus()
    assert corpus is not None
    assert info["corpus_rows"] == info["anchor_rows"] + info["confident"]
    assert info["pseudo_share"] > 0
    # corpus must carry the full labelled schema
    assert "attack_type" in corpus.columns
    assert set(corpus["attack_type"].unique()).issubset(
        {"normal", "ddos", "port_scan", "brute_force", "botnet"}
    )
