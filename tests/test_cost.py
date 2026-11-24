"""Tests for src/cost.py: the estimate, and where it is logged."""

from pathlib import Path

import mlflow
import pytest

from src import cost, prep, train
from src.common import load_config, read_json

COST = {
    "vm_size": "Standard_DS3_v2",
    "price_per_hour": 0.30,
    "currency": "USD",
    "extra_minutes": 20,
}


def test_estimate_counts_the_extra_minutes():
    # 40 job minutes + 20 extra = 1 hour at 0.30 per hour.
    assert cost.estimate_cost(40, COST, on_azure=True) == pytest.approx(0.30)


def test_a_laptop_costs_nothing():
    assert cost.estimate_cost(40, COST, on_azure=False) == 0.0


def test_end_logs_duration_and_cost_on_the_training_run(config_path: Path):
    config = str(config_path)
    cost.main(["start", "--config", config])
    prep.main(["--config", config])
    train.main(["--config", config])
    cost.main(["end", "--config", config])

    result = read_json("outputs/cost.json")
    assert result["where"] == "local"
    assert result["estimated_cost"] == 0.0
    assert result["job_minutes"] >= 0

    mlflow.set_tracking_uri(load_config(config)["mlflow"]["tracking_uri"])
    run = mlflow.get_run(read_json("outputs/train_run.json")["run_id"])
    assert "job_minutes" in run.data.metrics
    assert run.data.tags["cost_vm_size"] == "Standard_DS3_v2"


def test_inside_azure_ml_the_jobs_run_gets_the_cost(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MLFLOW_RUN_ID", "abc123")
    assert cost.in_azure_ml_job()
    assert cost.run_to_log_to("does-not-exist.json") == "abc123"


def test_rules_config_has_a_complete_cost_section():
    cost_cfg = load_config(Path(__file__).parents[1] / "config.yml")["cost"]
    assert set(cost_cfg) == set(COST)
    assert cost_cfg["price_per_hour"] > 0
