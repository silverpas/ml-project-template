"""Small helpers shared by all pipeline scripts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import mlflow
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Read the YAML settings file."""
    with open(path) as f:
        return yaml.safe_load(f)


def setup_mlflow(config: dict[str, Any]) -> None:
    """Point MLflow at the tracking store and select the experiment.

    If MLFLOW_TRACKING_URI is set (as it is inside an Azure ML job),
    that wins over the local setting in the config file.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(config["mlflow"]["experiment"])


def git_commit() -> str:
    """Return the current git commit hash, or 'unknown' outside a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def file_sha256(path: str | Path) -> str:
    """Fingerprint of a file: same content always gives the same hash."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(data: dict[str, Any], path: str | Path) -> None:
    """Write a dict as pretty JSON, creating the folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON file into a dict."""
    return json.loads(Path(path).read_text())
