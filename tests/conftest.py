"""Shared test fixtures: fake AI4I-like data and a throwaway config.

Tests never download the real dataset and never touch Azure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


def make_raw_data(n_rows: int = 400, seed: int = 0) -> pd.DataFrame:
    """Small dataset with the same columns as the raw AI4I CSV.

    Failures are linked to high torque, so a model can learn something.
    """
    rng = np.random.default_rng(seed)
    torque = rng.normal(40, 10, n_rows).round(1)
    failure = (torque + rng.normal(0, 5, n_rows) > 52).astype(int)
    return pd.DataFrame(
        {
            "UDI": np.arange(1, n_rows + 1),
            "Product ID": [f"X{i:05d}" for i in range(n_rows)],
            "Type": rng.choice(["L", "M", "H"], n_rows),
            "Air temperature [K]": rng.normal(300, 2, n_rows).round(1),
            "Process temperature [K]": rng.normal(310, 1.5, n_rows).round(1),
            "Rotational speed [rpm]": rng.integers(1200, 2800, n_rows),
            "Torque [Nm]": torque,
            "Tool wear [min]": rng.integers(0, 250, n_rows),
            "Machine failure": failure,
            "TWF": failure,  # failure-reason columns: must be dropped by prep
            "HDF": 0,
            "PWF": 0,
            "OSF": 0,
            "RNF": 0,
        }
    )


@pytest.fixture
def raw_data() -> pd.DataFrame:
    return make_raw_data()


@pytest.fixture
def config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Config file pointing at fake data and a temporary MLflow store.

    Also moves into tmp_path, so model files (./mlruns) land there.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    raw_file = tmp_path / "raw.csv"
    make_raw_data().to_csv(raw_file, index=False)

    config = yaml.safe_load(Path(__file__).parents[1].joinpath("config.yml").read_text())
    config["data"]["source"] = str(raw_file)
    config["data"]["source_url"] = "http://invalid.example/never-downloaded.zip"
    config["model"]["params"]["n_estimators"] = 20  # keep tests fast
    config["mlflow"]["tracking_uri"] = f"sqlite:///{tmp_path / 'mlflow.db'}"
    config["mlflow"]["experiment"] = "test"

    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config))
    return path
