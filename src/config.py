"""Central configuration: YAML settings + environment variables.

All pipeline components import `settings` from this module so there is a
single source of truth. Connection strings are resolved from the
environment so the same code runs locally (SQLite fallback) and inside
Docker (PostgreSQL / MLflow services).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Project root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_YAML = _load_yaml(PROJECT_ROOT / "config" / "config.yaml")


class _Settings:
    """Dot-accessible merged view of config.yaml and environment."""

    # ---- YAML sections (dot access, e.g. settings.get("drift.psi_warn")) ----
    raw: dict[str, Any] = _YAML

    @staticmethod
    def get(dotted_key: str, default: Any = None) -> Any:
        node: Any = _Settings.raw
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    # ---- Paths ----
    PROJECT_ROOT = PROJECT_ROOT
    MODELS_DIR = PROJECT_ROOT / "models"
    DATA_DIR = PROJECT_ROOT / "data"

    # ---- Experiment tracking ----
    EXPERIMENT_NAME: str = _YAML["project"]["experiment_name"]
    REGISTRY_MODEL_NAME: str = _YAML["project"]["registry_model_name"]
    RANDOM_STATE: int = _YAML["project"]["random_state"]

    # ---- MLflow connection ----
    @property
    def MLFLOW_TRACKING_URI(self) -> str:
        return os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    # ---- Database connection ----
    @property
    def DATABASE_URL(self) -> str:
        """PostgreSQL in Docker, SQLite fallback for local testing."""
        url = os.getenv("DATABASE_URL")
        if url:
            return url
        host = os.getenv("POSTGRES_HOST")
        if host:
            user = os.getenv("POSTGRES_USER", "nad_user")
            pwd = os.getenv("POSTGRES_PASSWORD", "nad_password")
            db = os.getenv("POSTGRES_DB", "network_ids")
            port = os.getenv("POSTGRES_PORT", "5432")
            return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"
        return f"sqlite:///{PROJECT_ROOT / 'data' / 'nad_local.db'}"

    @property
    def IS_POSTGRES(self) -> bool:
        return self.DATABASE_URL.startswith("postgresql")


settings = _Settings()

# Ensure standard directories exist
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
for key in ("raw_dir", "processed_dir", "drift_dir"):
    (settings.DATA_DIR / settings.get(f"data.{key}").split("/")[-1]).mkdir(
        parents=True, exist_ok=True
    )
