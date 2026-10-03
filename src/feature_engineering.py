"""Preprocessing + feature engineering.

Fits a reusable sklearn Pipeline on the training split, persists it with
the model (the same object must be applied to live traffic), and adds
engineered ratio features. Exported via MLflow's `pipelines` API so the
serving image can apply byte-identical transformations.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from src.config import settings

logger = logging.getLogger(__name__)

NUMERIC_FEATURES: list[str] = settings.get("training.numeric_features")
CATEGORICAL_FEATURES: list[str] = settings.get("training.categorical_features")
LABEL_COLUMN: str = settings.get("training.label_column")

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ratio/temporal features computed on raw flows (idempotent).

    The three new features are derived purely from base columns, so the
    same function applied at serving time reproduces training features.
    """
    df = df.copy()
    eps = 1e-6
    if "bytes_per_packet" not in df.columns:
        total = df["src_bytes"] + df["dst_bytes"]
        denom = (df["count"] * df["packet_rate"]).replace(0, np.nan)
        df["bytes_per_packet"] = (total / denom.replace(0, np.nan)).clip(0, 5000).fillna(0)
    if "bytes_ratio" not in df.columns:
        df["bytes_ratio"] = df["src_bytes"] / (df["dst_bytes"] + eps)
    if "src_bytes_log" not in df.columns:
        df["src_bytes_log"] = np.log1p(df["src_bytes"].clip(lower=0))
    if "duration_log" not in df.columns:
        df["duration_log"] = np.log1p(df["duration"].clip(lower=0))
    return df


class FeatureBuilder:
    """Wraps the fitted ColumnTransformer; saved next to each model run."""

    def __init__(self, transformer: Pipeline, feature_columns: list[str]):
        self.transformer = transformer
        self.feature_columns = feature_columns
        self.numeric_features = NUMERIC_FEATURES
        self.categorical_features = CATEGORICAL_FEATURES

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        # Pass the full frame; the ColumnTransformer picks the columns it needs.
        # add_engineered_features is idempotent so raw or featurized input both work.
        df = add_engineered_features(df)
        return self.transformer.transform(df)


def build_preprocessor() -> ColumnTransformer:
    # bytes_per_packet is already in NUMERIC_FEATURES; add the three derived ones
    numeric = NUMERIC_FEATURES + ["bytes_ratio", "src_bytes_log", "duration_log"]
    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def make_feature_builder(X_train: pd.DataFrame) -> FeatureBuilder:
    """Fit the transformer on training rows and wrap it in a FeatureBuilder."""
    X = add_engineered_features(X_train)
    transformer = build_preprocessor().fit(X)
    return FeatureBuilder(transformer, FEATURE_COLUMNS)


def save_feature_builder(fb: FeatureBuilder, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fb, path)
    return path


def load_feature_builder(path: Path) -> FeatureBuilder:
    return joblib.load(path)
