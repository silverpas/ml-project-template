"""Guards for the project rules on features and metrics (see AGENTS.md, "Rules").

These tests read the REAL config.yml, so a risky settings change fails CI.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.common import load_config
from src.evaluate import METRIC_NAMES, check_primary_metric, compute_metrics

ROOT = Path(__file__).parents[1]
CONFIG = load_config(ROOT / "config.yml")

# Known only AFTER a failure happened: using them as inputs leaks the answer.
LEAKING_COLUMNS = {"twf", "hdf", "pwf", "osf", "rnf"}
# Row identifiers: no signal, and the model would just memorize them.
ID_COLUMNS = {"udi", "product_id"}


def test_config_drops_leaking_and_id_columns():
    dropped = set(CONFIG["data"]["drop_columns"])
    assert LEAKING_COLUMNS <= dropped, f"not dropped: {LEAKING_COLUMNS - dropped}"
    assert ID_COLUMNS <= dropped, f"not dropped: {ID_COLUMNS - dropped}"


def test_target_is_not_dropped():
    assert CONFIG["data"]["target"] not in CONFIG["data"]["drop_columns"]


def test_primary_metric_is_supported_and_not_accuracy():
    metric = CONFIG["evaluation"]["primary_metric"]
    check_primary_metric(metric)  # raises if unknown
    assert metric != "accuracy"


def test_metric_names_match_compute_metrics():
    metrics = compute_metrics(pd.Series([0, 1, 0, 1]), np.array([0.2, 0.7, 0.4, 0.9]), 0.5)
    assert list(metrics) == METRIC_NAMES


def test_unknown_primary_metric_fails_clearly():
    with pytest.raises(ValueError, match="pr_auc"):
        check_primary_metric("pr_auc")


def test_predictions_table_follows_the_gold_naming_rule():
    """Gold tables written by ML start with `ml_` (lowercase, digits, underscores)."""
    assert re.fullmatch(r"ml_[a-z0-9_]+", CONFIG["scoring"]["predictions_table"])


def test_batch_score_job_writes_to_the_configured_table():
    job = yaml.safe_load((ROOT / "jobs" / "batch-score.yml").read_text())
    table_address = job["inputs"]["predictions_table"]
    assert table_address.endswith(f"/Tables/{CONFIG['scoring']['predictions_table']}")
