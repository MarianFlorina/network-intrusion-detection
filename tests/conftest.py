from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure project root is importable and tests use the SQLite fallback DB
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.pop("POSTGRES_HOST", None)
os.environ.pop("DATABASE_URL", None)
# Use a local file-based MLflow store in tests (no server running)
os.environ.setdefault("MLFLOW_TRACKING_URI", "file:./mlruns-test")


@pytest.fixture(scope="session")
def small_flows() -> pd.DataFrame:
    from src.data_ingestion import generate_flows

    return generate_flows(n=4000, seed="tests")


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    """Point the db layer at a throwaway SQLite file."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    from src import db

    db._engine = None
    db._SessionFactory = None
    yield db
    db._engine = None
    db._SessionFactory = None
