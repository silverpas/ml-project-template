"""Tests for model status tags: candidate, champion (only one), archived."""

from pathlib import Path

import mlflow
import pytest

from src.registry import (
    ARCHIVED,
    CANDIDATE,
    CHAMPION,
    champion_uri,
    find_version,
    promote,
    set_status,
)

MODEL = "test-model"


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> mlflow.MlflowClient:
    """An empty local registry with one model that has three versions."""
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    client = mlflow.MlflowClient()
    client.create_registered_model(MODEL)
    for _ in range(3):
        client.create_model_version(MODEL, source=str(tmp_path))  # no real model needed
    return client


def status(client: mlflow.MlflowClient, version: str) -> str | None:
    return client.get_model_version(MODEL, version).tags.get("status")


def test_no_champion_yet(registry):
    assert find_version(MODEL, CHAMPION) is None
    with pytest.raises(LookupError, match="--promote"):
        champion_uri(MODEL)


def test_promote_makes_champion_and_archives_the_previous_one(registry):
    promote(MODEL, "1")
    assert champion_uri(MODEL) == f"models:/{MODEL}/1"

    set_status(MODEL, "2", CANDIDATE)
    promote(MODEL, "2")
    assert champion_uri(MODEL) == f"models:/{MODEL}/2"
    assert status(registry, "1") == ARCHIVED
    assert find_version(MODEL, CANDIDATE) is None  # the candidate became champion


def test_promoting_the_champion_again_changes_nothing(registry):
    promote(MODEL, "3")
    promote(MODEL, "3")
    assert status(registry, "3") == CHAMPION
