"""Model status in the MLflow registry: which version is the candidate, which is approved.

Each registered version gets a tag `status`:
    candidate   better than the champion, waiting for approval
    champion    approved, used for scoring (only one version at a time)
    archived    was champion before, replaced by a newer approved version

We use tags, not MLflow aliases, because tags work the same in local MLflow
and in the Azure ML registry (Azure ML does not document alias support).
"""

from __future__ import annotations

import mlflow

STATUS_TAG = "status"
CANDIDATE = "candidate"
CHAMPION = "champion"
ARCHIVED = "archived"


def find_version(model_name: str, status: str) -> str | None:
    """The newest version with this status, or None if there is none."""
    versions = mlflow.MlflowClient().search_model_versions(f"name='{model_name}'")
    matching = [int(v.version) for v in versions if v.tags.get(STATUS_TAG) == status]
    return str(max(matching)) if matching else None


def set_status(model_name: str, version: str, status: str) -> None:
    """Tag one version with a status."""
    mlflow.MlflowClient().set_model_version_tag(model_name, version, STATUS_TAG, status)


def promote(model_name: str, version: str) -> None:
    """Approval: make `version` the champion; the previous champion becomes archived."""
    previous = find_version(model_name, CHAMPION)
    if previous is not None and previous != version:
        set_status(model_name, previous, ARCHIVED)
    set_status(model_name, version, CHAMPION)


def champion_uri(model_name: str) -> str:
    """MLflow URI of the approved model, e.g. models:/tool-wear/3."""
    version = find_version(model_name, CHAMPION)
    if version is None:
        raise LookupError(
            f"No approved version of '{model_name}' yet. "
            "Approve one with: python -m src.register --promote <version>"
        )
    return f"models:/{model_name}/{version}"
