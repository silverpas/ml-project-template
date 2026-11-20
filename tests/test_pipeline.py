"""End-to-end smoke test: run every step on fake data with a temporary MLflow store."""

from pathlib import Path

import pandas as pd

from src import evaluate, prep, register, score, train
from src.common import read_json


def run_train_and_evaluate(config: str) -> dict:
    train.main(["--config", config])
    evaluate.main(["--config", config])
    return read_json("outputs/evaluation.json")


def test_full_pipeline(config_path: Path):
    config = str(config_path)

    prep.main(["--config", config])
    assert Path("data/processed/train.parquet").exists()

    # First model: no champion yet, so it becomes a candidate.
    first = run_train_and_evaluate(config)
    assert first["is_candidate"] is True
    assert first["champion_version"] is None
    assert "First model" in Path("outputs/evaluation.md").read_text()
    register.main(["--config", config])
    assert read_json("outputs/registration.json")["version"] == "1"

    # The registered model's inputs contain no leaking or ID columns.
    model_inputs = set(score.expected_columns(first["model_uri"]))
    assert not model_inputs & {"twf", "hdf", "pwf", "osf", "rnf", "udi", "product_id"}
    assert "machine_failure" not in model_inputs

    # Human approval: promote version 1 to champion.
    register.main(["--config", config, "--promote", "1"])

    # Same data + same seed = same model: not better, so not registered.
    second = run_train_and_evaluate(config)
    assert second["champion_version"] == "1"
    assert second["is_candidate"] is False

    # Score the raw file with the champion.
    raw_file = str(Path(config_path).parent / "raw.csv")
    score.main(["--config", config, "--input", raw_file, "--output", "data/predictions/pred.csv"])
    predictions = pd.read_csv("data/predictions/pred.csv")
    assert len(predictions) == 400
    assert predictions["failure_probability"].between(0, 1).all()
    assert set(predictions["predicted_failure"]) <= {0, 1}
