"""Tests for src/export.py: only approved models leave, and the export works on its own."""

from pathlib import Path

import mlflow
import pandas as pd
import pytest

from src import evaluate, export, prep, register, train
from src.common import read_json


def test_export_only_after_approval(config_path: Path, raw_data):
    config = str(config_path)
    prep.main(["--config", config])
    train.main(["--config", config])
    evaluate.main(["--config", config])
    register.main(["--config", config])  # version 1, status candidate

    # Not approved yet: nothing to export, and version 1 is refused.
    with pytest.raises(LookupError, match="No approved version"):
        export.main(["--config", config])
    with pytest.raises(PermissionError, match="only approved"):
        export.main(["--config", config, "--version", "1"])

    register.main(["--config", config, "--promote", "1"])
    export.main(["--config", config])

    folder = Path("outputs/export/tool-wear-v1")
    assert (folder / "MLmodel").exists()
    assert (folder / "requirements.txt").exists()
    info = read_json(folder / "export.json")
    assert info["version"] == "1"
    assert info["status"] == "champion"
    assert "test_average_precision" in info["test_metrics"]

    # Anyone can load the folder with plain MLflow, without our registry.
    model = mlflow.pyfunc.load_model(str(folder))
    features = raw_data.head(3).rename(columns=prep.to_snake_case)
    features = features[model.metadata.get_input_schema().input_names()]
    assert len(model.predict(pd.DataFrame(features))) == 3
