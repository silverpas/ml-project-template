"""Step 5: batch scoring. Use a registered model to predict on new data.

Usage:
    python -m src.score --input data/raw/ai4i2020.csv

By default it uses the approved model (the version tagged status=champion).
Pass --model to use another one, e.g. models:/tool-wear/2.
The output has the input rows plus failure_probability, predicted_failure,
model_uri and scored_at.

Where the predictions go (--output):
- default: a Delta table in data/predictions/<predictions_table>/ (git-ignored),
  the local stand-in for the Gold table in Fabric;
- in Azure ML: the Gold table itself, as an abfss://...onelake... address;
- a path ending in .csv or .parquet: a plain file (handy to open in Excel).

Delta is the table format Fabric uses: Parquet files plus a log of versions.
Each run replaces the table's rows; older versions stay readable (time travel).
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import pandas as pd
from deltalake import write_deltalake

from src.common import load_config, setup_mlflow
from src.data import is_onelake, read_data
from src.prep import clean_columns
from src.registry import champion_uri


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


def write_predictions(df: pd.DataFrame, output: str) -> None:
    """Write a CSV or Parquet file, or (any other path) replace a Delta table."""
    if is_onelake(output):
        # ASSUMPTION: the job runs as this project's identity (ml-project module),
        # which has write access to the Gold lakehouse in Fabric.
        write_deltalake(
            output, df, mode="overwrite", storage_options={"use_fabric_endpoint": "true"}
        )
        return

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        write_deltalake(path, df, mode="overwrite")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yml", help="Settings file")
    parser.add_argument("--model", help="MLflow model URI (default: the champion)")
    parser.add_argument("--input", required=True, help="CSV, Parquet or Delta table to score")
    parser.add_argument(
        "--output",
        help="Delta table folder or abfss:// address, or a .csv/.parquet file "
        "(default: data/predictions/<predictions_table>)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    setup_mlflow(config)
    model_uri = args.model or champion_uri(config["model"]["name"])
    output = args.output or f"data/predictions/{config['scoring']['predictions_table']}"

    predictions = score(read_data(args.input), model_uri, config["evaluation"]["threshold"])
    write_predictions(predictions, output)
    print(
        f"Scored {len(predictions)} rows with {model_uri} -> {output} "
        f"({predictions['predicted_failure'].sum()} predicted failures)"
    )


if __name__ == "__main__":
    main()
