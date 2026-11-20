"""Tests for evaluation logic: metrics and the 'is the new model better?' rule."""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import compute_metrics, is_better, summary_markdown


def test_first_model_is_always_a_candidate():
    assert is_better(new_score=0.1, current_score=None, min_improvement=0.0)


def test_better_model_is_a_candidate():
    assert is_better(new_score=0.80, current_score=0.75, min_improvement=0.0)


def test_equal_or_worse_model_is_not_a_candidate():
    assert not is_better(new_score=0.75, current_score=0.75, min_improvement=0.0)
    assert not is_better(new_score=0.70, current_score=0.75, min_improvement=0.0)


def test_min_improvement_is_respected():
    assert not is_better(new_score=0.76, current_score=0.75, min_improvement=0.02)
    assert is_better(new_score=0.78, current_score=0.75, min_improvement=0.02)


def test_compute_metrics_perfect_model():
    y_true = pd.Series([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_metrics(y_true, y_proba, threshold=0.5)
    assert metrics == pytest.approx(
        {"average_precision": 1.0, "roc_auc": 1.0, "f1": 1.0, "precision": 1.0, "recall": 1.0}
    )


def test_compute_metrics_threshold_changes_predictions():
    y_true = pd.Series([0, 1])
    y_proba = np.array([0.3, 0.4])
    assert compute_metrics(y_true, y_proba, threshold=0.5)["recall"] == 0.0
    assert compute_metrics(y_true, y_proba, threshold=0.35)["recall"] == 1.0


def make_evaluation(is_candidate: bool, champion: bool) -> dict:
    metrics = {"average_precision": 0.8, "roc_auc": 0.9}
    return {
        "primary_metric": "average_precision",
        "new_metrics": metrics,
        "champion_version": "1" if champion else None,
        "champion_metrics": {"average_precision": 0.7, "roc_auc": 0.85} if champion else None,
        "is_candidate": is_candidate,
    }


def test_summary_shows_both_models_and_decision():
    text = summary_markdown(make_evaluation(is_candidate=True, champion=True))
    assert "Champion (v1)" in text
    assert "| **average_precision** (decides) | 0.800 | 0.700 |" in text
    assert "waiting for approval" in text


def test_summary_first_model_and_rejected_model():
    first = summary_markdown(make_evaluation(is_candidate=True, champion=False))
    assert "Champion (none yet)" in first and "First model" in first

    rejected = summary_markdown(make_evaluation(is_candidate=False, champion=True))
    assert "not registered" in rejected
