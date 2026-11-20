"""Step 2: train a model and log everything about the run to MLflow.

Usage:
    python -m src.train --data data/processed

Logs to MLflow: parameters, a training metric, the data version, the git commit
and the model itself (with its input signature).
Writes --run-info (default outputs/train_run.json) so the next step knows which run to evaluate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.common import git_commit, load_config, read_json, setup_mlflow, write_json
from src.evaluate import compute_metrics

# MLflow saves sklearn models with "skops", a safe format that only loads
# object types you explicitly trust (a pickle file could run any code on load).
# These are the LightGBM types inside our model. Update if you change the model.
TRUSTED_MODEL_TYPES = [
    "collections.OrderedDict",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier",
]


def split_features_target(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the input columns (X) from the column to predict (y)."""
    return df.drop(columns=[target]), df[target]


def build_model(categorical_columns: list[str], params: dict[str, Any]) -> Pipeline:
    """Preprocessing + LightGBM in one sklearn Pipeline.

    Keeping the preprocessing inside the model means scoring code can pass the
    same raw columns as training, with no risk of encoding them differently.
    """
    preprocess = ColumnTransformer(
        [
            (
                "categorical",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical_columns,
            )
        ],
        remainder="passthrough",  # numeric columns go through unchanged
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    return Pipeline([("preprocess", preprocess), ("classifier", LGBMClassifier(**params))])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yml", help="Settings file")
    parser.add_argument("--data", default="data/processed", help="Output folder of prep")
    parser.add_argument(
        "--run-info", default="outputs/train_run.json", help="Where to write the run id"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    data_dir = Path(args.data)
    target = config["data"]["target"]

    train_df = pd.read_parquet(data_dir / "train.parquet")
    X, y = split_features_target(train_df, target)
    model = build_model(config["data"]["categorical_columns"], config["model"]["params"])

    setup_mlflow(config)
    with mlflow.start_run() as run:
        model.fit(X, y)

        # What we trained with: settings, data version and code version.
        mlflow.log_params(config["model"]["params"])
        mlflow.log_params({"target": target, "train_rows": len(X)})
        mlflow.set_tags({"data_path": str(data_dir), "git_commit": git_commit()})
        mlflow.set_tags({f"data_{k}": v for k, v in read_json(data_dir / "metadata.json").items()})

        # Score on the training data only as a sanity check; the real
        # comparison happens on the test set in evaluate.py.
        threshold = config["evaluation"]["threshold"]
        train_metrics = compute_metrics(y, model.predict_proba(X)[:, 1], threshold)
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})

        # The signature records the expected input columns and types. Fabric
        # PREDICT needs it, and it catches wrong inputs at scoring time.
        signature = infer_signature(X, model.predict(X))
        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=X.head(5),
            skops_trusted_types=TRUSTED_MODEL_TYPES,
        )

    write_json({"run_id": run.info.run_id, "model_uri": model_info.model_uri}, args.run_info)
    print(
        f"Run {run.info.run_id}: train {config['evaluation']['primary_metric']} = "
        f"{train_metrics[config['evaluation']['primary_metric']]:.3f}"
    )


if __name__ == "__main__":
    main()
