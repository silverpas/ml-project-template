"""Tests for src/data.py: every kind of input reads the same, and is traceable."""

from pathlib import Path

import pandas as pd
import pytest
from deltalake import write_deltalake

from src import prep
from src.common import read_json
from src.data import describe_source, read_data


@pytest.mark.parametrize("kind", ["csv", "parquet", "delta"])
def test_read_data_gives_the_same_table_for_each_format(tmp_path: Path, raw_data, kind):
    if kind == "csv":
        source = tmp_path / "raw.csv"
        raw_data.to_csv(source, index=False)
    elif kind == "parquet":
        source = tmp_path / "raw.parquet"
        raw_data.to_parquet(source, index=False)
    else:
        source = tmp_path / "raw_table"
        write_deltalake(source, raw_data)

    df = read_data(source)
    assert list(df.columns) == list(raw_data.columns)
    assert len(df) == len(raw_data)
    assert describe_source(source)["source_format"] == kind


def test_describe_source_gives_the_delta_version(tmp_path: Path, raw_data):
    table = tmp_path / "raw_table"
    write_deltalake(table, raw_data)
    write_deltalake(table, raw_data, mode="overwrite")
    assert describe_source(table)["source_version"] == 1  # versions 0 and 1


def test_describe_source_fingerprints_a_file(tmp_path: Path, raw_data):
    source = tmp_path / "raw.csv"
    raw_data.to_csv(source, index=False)
    assert len(describe_source(source)["source_sha256"]) == 64


def test_fabric_is_not_built_yet():
    with pytest.raises(NotImplementedError, match="section 9"):
        read_data("abfss://ws@onelake.dfs.fabric.microsoft.com/gold.Lakehouse/Tables/t")


def test_unknown_format_fails_clearly(tmp_path: Path):
    with pytest.raises(ValueError, match="Don't know how to read"):
        read_data(tmp_path / "data.xlsx")


def test_prep_reads_a_delta_table_and_records_its_version(config_path: Path, raw_data):
    write_deltalake("data/tables/sensors", raw_data)
    prep.main(["--config", str(config_path), "--input", "data/tables/sensors"])

    metadata = read_json("data/processed/metadata.json")
    assert metadata["source"] == "data/tables/sensors"
    assert metadata["source_format"] == "delta"
    assert metadata["source_version"] == 0
    assert metadata["train_rows"] + metadata["test_rows"] == len(raw_data)
    assert len(pd.read_parquet("data/processed/train.parquet")) == metadata["train_rows"]
