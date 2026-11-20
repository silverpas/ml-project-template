"""Step 1: get the raw data, clean it and split it into train and test sets.

Usage:
    python -m src.prep --out data/processed

Outputs (in --out):
    train.parquet, test.parquet   features + target
    metadata.json                 row counts and file fingerprints (data version)
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


def download_raw(url: str, dest: Path) -> None:
    """Download the zip from `url` and save the CSV inside it to `dest`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    csv_name = next(n for n in archive.namelist() if n.endswith(".csv"))
    dest.write_bytes(archive.read(csv_name))


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw CSV. 'utf-8-sig' strips the invisible BOM the AI4I file starts with."""
    return pd.read_csv(path, encoding="utf-8-sig")


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
    parser.add_argument("--raw", help="Raw CSV (default: data.raw_file in config)")
    parser.add_argument("--out", default="data/processed", help="Output folder")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    data_cfg = config["data"]

    raw_path = Path(args.raw or data_cfg["raw_file"])
    if not raw_path.exists():
        print(f"Downloading raw data to {raw_path} ...")
        download_raw(data_cfg["source_url"], raw_path)

    df = prepare(load_raw(raw_path), data_cfg["drop_columns"])
    train, test = split(
        df, data_cfg["target"], config["split"]["test_size"], config["split"]["seed"]
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out / "train.parquet", index=False)
    test.to_parquet(out / "test.parquet", index=False)
    write_json(
        {
            "raw_file": str(raw_path),
            "raw_sha256": file_sha256(raw_path),
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
