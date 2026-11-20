"""Step 3: compare the new model with the current approved ("champion") model.

Both models are scored on the SAME test set, so the comparison is fair.
The new model becomes a candidate only if it is better on the primary metric.

Usage:
    python -m src.evaluate --data data/processed

Writes --out (default outputs/evaluation.json) with both models' metrics and
the decision `is_candidate`, which register.py reads.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.common import load_config, read_json, setup_mlflow, write_json
from src.registry import CHAMPION, find_version

# The metrics compute_metrics returns; evaluation.primary_metric must be one of them.
METRIC_NAMES = ["average_precision", "roc_auc", "f1", "precision", "recall"]


def check_primary_metric(metric: str) -> None:
    """Fail early with a clear message if the config names a metric we do not compute."""
    if metric not in METRIC_NAMES:
        raise ValueError(
            f"evaluation.primary_metric '{metric}' is not supported. Choose one of: {METRIC_NAMES}"
        )


def compute_metrics(y_true: pd.Series, y_proba: Any, threshold: float) -> dict[str, float]:
    """Classification metrics from true labels and predicted failure probabilities."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def is_better(new_score: float, current_score: float | None, min_improvement: float) -> bool:
    """True if there is no current model yet, or the new one wins by at least min_improvement."""
    if current_score is None:
        return True
    return new_score > current_score + min_improvement


def load_champion(model_name: str) -> tuple[Any, str] | tuple[None, None]:
    """Load the current champion model and its version, or (None, None) if there is none."""
    try:
        version = find_version(model_name, CHAMPION)
    except MlflowException:
        return None, None  # model not registered yet
    if version is None:
        return None, None  # registered, but no version approved yet
    return mlflow.sklearn.load_model(f"models:/{model_name}/{version}"), version


def summary_markdown(evaluation: dict[str, Any]) -> str:
    """A short Markdown report of the comparison, e.g. for the GitHub run page."""
    metric = evaluation["primary_metric"]
    new, champion = evaluation["new_metrics"], evaluation["champion_metrics"]
    champion_label = (
        f"Champion (v{evaluation['champion_version']})" if champion else "Champion (none yet)"
    )

    lines = [
        "## Model evaluation",
        "",
        f"| Metric | New model | {champion_label} |",
        "|---|---|---|",
    ]
    for name, value in new.items():
        label = f"**{name}** (decides)" if name == metric else name
        current = f"{champion[name]:.3f}" if champion else "–"
        lines.append(f"| {label} | {value:.3f} | {current} |")

    if evaluation["is_candidate"]:
        decision = "✅ **Better than the champion: registered as candidate, waiting for approval.**"
        if not champion:
            decision = "✅ **First model: registered as candidate, waiting for approval.**"
    else:
        decision = "⏹️ **Not better than the champion: not registered, nothing to approve.**"
    lines += ["", decision, ""]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yml", help="Settings file")
    parser.add_argument("--data", default="data/processed", help="Output folder of prep")
    parser.add_argument("--run-info", default="outputs/train_run.json", help="Written by train")
    parser.add_argument("--out", default="outputs/evaluation.json", help="Decision file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    eval_cfg = config["evaluation"]
    metric = eval_cfg["primary_metric"]
    check_primary_metric(metric)
    run_info = read_json(args.run_info)

    test_df = pd.read_parquet(Path(args.data) / "test.parquet")
    X_test = test_df.drop(columns=[config["data"]["target"]])
    y_test = test_df[config["data"]["target"]]

    setup_mlflow(config)
    new_model = mlflow.sklearn.load_model(run_info["model_uri"])
    new_metrics = compute_metrics(
        y_test, new_model.predict_proba(X_test)[:, 1], eval_cfg["threshold"]
    )

    champion, champion_version = load_champion(config["model"]["name"])
    champion_metrics = None
    if champion is not None:
        champion_metrics = compute_metrics(
            y_test, champion.predict_proba(X_test)[:, 1], eval_cfg["threshold"]
        )

    candidate = is_better(
        new_metrics[metric],
        champion_metrics[metric] if champion_metrics else None,
        eval_cfg["min_improvement"],
    )

    # Store the test results on the training run, so everything is in one place.
    with mlflow.start_run(run_id=run_info["run_id"]):
        mlflow.log_metrics({f"test_{k}": v for k, v in new_metrics.items()})
        mlflow.set_tag("is_candidate", str(candidate))
        mlflow.set_tag("compared_with_champion_version", str(champion_version))

    evaluation = {
        **run_info,
        "primary_metric": metric,
        "new_metrics": new_metrics,
        "champion_version": champion_version,
        "champion_metrics": champion_metrics,
        "is_candidate": candidate,
    }
    write_json(evaluation, args.out)
    # Same result as a readable table (shown on the GitHub run page).
    Path(args.out).with_suffix(".md").write_text(summary_markdown(evaluation))
    champion_text = (
        f"champion v{champion_version} = {champion_metrics[metric]:.3f}"
        if champion_metrics
        else "no champion yet"
    )
    print(
        f"New model test {metric} = {new_metrics[metric]:.3f} ({champion_text}) "
        f"-> candidate: {candidate}"
    )


if __name__ == "__main__":
    main()
