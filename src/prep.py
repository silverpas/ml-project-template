"""Step 1: get the raw data, clean it and split it into train and test sets.

Usage:
    python -m src.prep --out data/processed
    python -m src.prep --input data/tables/sensors     # a Delta table instead of the CSV

The data is read with src/data.py (CSV, Parquet or Delta table).

Outputs (in --out):
    train.parquet, test.parquet   features + target
    metadata.json                 which data it was (source, format, version or
                                  fingerprint), row counts and file fingerprints
"""

from __future__ import annotations

import argparse
import io
import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.common import file_sha256, load_config, write_json
from src.data import describe_source, is_onelake, read_data


def download_raw(url: str, dest: Path) -> None:
    """Download the example zip from `url` and save the CSV inside it to `dest`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    csv_name = next(n for n in archive.namelist() if n.endswith(".csv"))
    dest.write_bytes(archive.read(csv_name))


def to_snake_case(name: str) -> str:
    """'Air temperature [K]' -> 'air_temperature_k'."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename all columns to snake_case.

    Simple names are easier to use in code and SQL, and LightGBM refuses
    feature names with characters like '[' or ']'.
    """
    return df.rename(columns=to_snake_case)


def prepare(df: pd.DataFrame, drop_columns: list[str]) -> pd.DataFrame:
    """Clean column names and drop the columns we must not train on."""
    df = clean_columns(df)
    return df.drop(columns=[c for c in drop_columns if c in df.columns])


def split(
    df: pd.DataFrame, target: str, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train and test, keeping the same share of failures in both (stratify)."""
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[target])
    return train.reset_index(drop=True), test.reset_index(drop=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yml", help="Settings file")
    parser.add_argument(
        "--input", help="CSV, Parquet or Delta table to read (default: data.source in config)"
    )
    parser.add_argument("--out", default="data/processed", help="Output folder")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    data_cfg = config["data"]

    source = args.input or data_cfg["source"]
    # The example data is downloaded the first time; your own data is just read.
    if not is_onelake(source) and not Path(source).exists() and data_cfg.get("source_url"):
        print(f"Downloading the example data to {source} ...")
        download_raw(data_cfg["source_url"], Path(source))

    df = prepare(read_data(source), data_cfg["drop_columns"])
    train, test = split(
        df, data_cfg["target"], config["split"]["test_size"], config["split"]["seed"]
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out / "train.parquet", index=False)
    test.to_parquet(out / "test.parquet", index=False)
    write_json(
        {
            **describe_source(source),
            "train_rows": len(train),
            "test_rows": len(test),
            "train_sha256": file_sha256(out / "train.parquet"),
            "test_sha256": file_sha256(out / "test.parquet"),
        },
        out / "metadata.json",
    )
    print(f"Wrote {len(train)} train and {len(test)} test rows to {out}")


if __name__ == "__main__":
    main()
