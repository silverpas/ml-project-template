"""Step 5: batch scoring. Use a registered model to predict on new data.

Usage:
    python -m src.score --input data/raw/ai4i2020.csv --output data/predictions/predictions.parquet

By default it uses the approved model (the version tagged status=champion).
Pass --model to use another one, e.g. models:/tool-wear/2.
The output has the input rows plus failure_probability, predicted_failure,
model_uri and scored_at.

Locally, predictions go to data/predictions/ (git-ignored). It stands in for
the predictions lakehouse in Fabric, where the Azure ML batch job writes them.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import pandas as pd

from src.common import load_config, setup_mlflow
from src.prep import clean_columns
from src.registry import champion_uri


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV or Parquet file."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def expected_columns(model_uri: str) -> list[str]:
    """The input columns the model was trained on, from its MLflow signature."""
    return mlflow.models.get_model_info(model_uri).signature.inputs.input_names()


def score(df: pd.DataFrame, model_uri: str, threshold: float) -> pd.DataFrame:
    """Add prediction columns to `df`. Columns the model does not use are kept as they are."""
    df = clean_columns(df)
    features = df[expected_columns(model_uri)]  # fails clearly if a column is missing
    model = mlflow.sklearn.load_model(model_uri)

    result = df.copy()
    result["failure_probability"] = model.predict_proba(features)[:, 1]
    result["predicted_failure"] = (result["failure_probability"] >= threshold).astype(int)
    result["model_uri"] = model_uri
    result["scored_at"] = datetime.now(UTC)
    return result


def write_table(df: pd.DataFrame, path: Path) -> None:
    """Write a CSV or Parquet file, creating the folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yml", help="Settings file")
    parser.add_argument("--model", help="MLflow model URI (default: the champion)")
    parser.add_argument("--input", required=True, help="CSV or Parquet file to score")
    parser.add_argument(
        "--output",
        default="data/predictions/predictions.parquet",
        help="CSV or Parquet file to write",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    setup_mlflow(config)
    model_uri = args.model or champion_uri(config["model"]["name"])

    predictions = score(read_table(Path(args.input)), model_uri, config["evaluation"]["threshold"])
    write_table(predictions, Path(args.output))
    print(
        f"Scored {len(predictions)} rows with {model_uri} -> {args.output} "
        f"({predictions['predicted_failure'].sum()} predicted failures)"
    )


if __name__ == "__main__":
    main()
