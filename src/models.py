"""Model zoo: the five candidate architectures compared on every run.

Supervised classifiers predict the attack class directly. Unsupervised
detectors (Isolation Forest, One-Class SVM, Autoencoder) are wrapped so
they emit class labels too: anomalies are clustered and each cluster is
mapped to the most frequent true attack class among its training members,
which is how a pure anomaly detector is deployed when labelled classes
are unavailable.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.svm import OneClassSVM

from src.config import settings

logger = logging.getLogger(__name__)

LABELS = ["normal", "ddos", "port_scan", "brute_force", "botnet"]
ANOMALY_LABELS = LABELS[1:]

RANDOM_STATE = settings.RANDOM_STATE


class AnomalyToLabelWrapper:
    """Maps detector output (-1 anomaly / +1 normal) to attack classes.

    K-means is fitted on the anomalous training points; each centroid is
    labelled with the majority true class of its members. At prediction
    time anomalies are assigned their nearest centroid's label.
    """

    def __init__(self, detector, n_clusters: int = 4):
        self.detector = detector
        self.n_clusters = n_clusters
        self.kmeans = None
        self.centroid_labels: list[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        # Classic anomaly-detection setup: learn the shape of NORMAL traffic
        # only, so every deviation (known or novel attack) scores as anomalous.
        if y is not None:
            benign = X[np.asarray(y) == "normal"]
            fit_X = benign if len(benign) >= 100 else X
        else:
            fit_X = X
        # Quadratic-time detectors get a fixed-size benign subsample
        if isinstance(self.detector, OneClassSVM) and len(fit_X) > 6000:
            idx = np.random.default_rng(RANDOM_STATE).choice(
                len(fit_X), size=6000, replace=False
            )
            fit_X = fit_X[idx]
        self.detector.fit(fit_X)
        anomaly_mask = self.detector.predict(X) == -1
        anomalies = X[anomaly_mask]
        if len(anomalies) == 0:  # degenerate detector: fabricate one cluster
            anomalies = X[:1]
            anomaly_mask = np.zeros(len(X), dtype=bool)
            anomaly_mask[0] = True
        n_clusters = max(1, min(self.n_clusters, len(anomalies)))
        if y is not None:
            # One cluster per attack class present among anomalies gives the
            # mapper a chance to recover the true class boundaries.
            n_clusters = max(1, min(n_clusters, len(set(y[anomaly_mask].tolist()))))
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=RANDOM_STATE)
        self.kmeans.fit(anomalies)

        if y is not None and len(anomalies) > 0:
            cluster_ids = self.kmeans.predict(anomalies)
            y_anomalies = y[anomaly_mask]
            # Centroids map to ATTACK classes only: training anomalies include
            # false positives labelled 'normal', and letting a centroid inherit
            # that label would silence real anomalies at inference time.
            attack_pool = [l for l in np.unique(y_anomalies) if l != "normal"]
            fallback = str(attack_pool[0]) if attack_pool else ANOMALY_LABELS[0]
            self.centroid_labels = []
            for c in range(n_clusters):
                members = y_anomalies[cluster_ids == c]
                attack_members = members[members != "normal"]
                if len(attack_members) > 0:
                    values, counts = np.unique(attack_members, return_counts=True)
                    self.centroid_labels.append(str(values[np.argmax(counts)]))
                else:
                    self.centroid_labels.append(fallback)
        else:
            self.centroid_labels = [
                ANOMALY_LABELS[i % len(ANOMALY_LABELS)] for i in range(n_clusters)
            ]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = self.detector.predict(X)
        out = np.full(len(X), "normal", dtype=object)
        anomaly_idx = np.where(raw == -1)[0]
        if len(anomaly_idx) and self.kmeans is not None:
            clusters = self.kmeans.predict(X[anomaly_idx])
            out[anomaly_idx] = np.array(self.centroid_labels, dtype=object)[clusters]
        return out

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Higher = more anomalous."""
        if hasattr(self.detector, "decision_function"):
            return -self.detector.decision_function(X)
        return np.zeros(len(X))


class LabelEncodedClassifier:
    """Adapts estimators that require integer class ids (e.g. XGBoost).

    Encodes string labels on fit, decodes predictions back to strings so
    every zoo model shares the same string-label contract.
    """

    def __init__(self, estimator):
        self.estimator = estimator
        self.le_ = None

    def fit(self, X, y):
        from sklearn.preprocessing import LabelEncoder

        self.le_ = LabelEncoder().fit(y)
        self.estimator.fit(X, self.le_.transform(y))
        return self

    def predict(self, X):
        return self.le_.inverse_transform(self.estimator.predict(X).astype(int))

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def get_params(self, deep=False):
        return self.estimator.get_params(deep=deep)


class AutoencoderDetector:
    """MLP reconstruction-error detector.

    Trained to reconstruct its input (benign-dominant traffic); flows with
    reconstruction error above the 99th-percentile threshold are anomalies.
    """

    def __init__(self, encoding_dim: int = 8, epochs: int = 30, random_state: int = RANDOM_STATE):
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.random_state = random_state
        self.mlp: MLPRegressor | None = None
        self.threshold: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        n_features = X.shape[1]
        hidden = max(n_features, 16)
        self.mlp = MLPRegressor(
            hidden_layer_sizes=(hidden, self.encoding_dim, hidden),
            activation="relu",
            max_iter=self.epochs,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=self.random_state,
        )
        self.mlp.fit(X, X)
        errors = self.reconstruction_error(X)
        self.threshold = float(np.quantile(errors, 0.99))
        return self

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        recon = self.mlp.predict(X)
        return np.mean((X - recon) ** 2, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.reconstruction_error(X) > self.threshold, -1, 1)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Sklearn convention: higher = more NORMAL (negated by the wrapper)."""
        return self.threshold - self.reconstruction_error(X)


def build_model(name: str):
    """Instantiate a model from the zoo by name."""
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return LabelEncodedClassifier(
            XGBClassifier(
                n_estimators=400,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.85,
                colsample_bytree=0.85,
                tree_method="hist",
                n_jobs=-1,
                random_state=RANDOM_STATE,
                eval_metric="mlogloss",
            )
        )
    if name == "isolation_forest":
        return AnomalyToLabelWrapper(
            IsolationForest(
                n_estimators=200, contamination=0.05, random_state=RANDOM_STATE, n_jobs=-1
            )
        )
    if name == "one_class_svm":
        # Vanilla RBF OCSVM on a benign subsample (kernel approximation
        # degraded the boundary in benchmarks); wrapper handles subsampling.
        return AnomalyToLabelWrapper(OneClassSVM(nu=0.05, gamma="auto"))
    if name == "autoencoder":
        return AnomalyToLabelWrapper(AutoencoderDetector())
    raise ValueError(f"Unknown model: {name}")


def is_unsupervised(name: str) -> bool:
    return name in {"isolation_forest", "one_class_svm", "autoencoder"}
