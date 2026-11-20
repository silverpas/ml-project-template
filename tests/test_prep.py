"""Tests for data preparation."""

import pandas as pd

from src.prep import clean_columns, prepare, split, to_snake_case

DROP = ["udi", "product_id", "twf", "hdf", "pwf", "osf", "rnf"]


def test_to_snake_case():
    assert to_snake_case("Air temperature [K]") == "air_temperature_k"
    assert to_snake_case("Product ID") == "product_id"
    assert to_snake_case("Machine failure") == "machine_failure"


def test_clean_columns_has_no_special_characters(raw_data: pd.DataFrame):
    cleaned = clean_columns(raw_data)
    for column in cleaned.columns:
        assert column == column.lower()
        assert "[" not in column and " " not in column


def test_prepare_drops_ids_and_leaking_columns(raw_data: pd.DataFrame):
    prepared = prepare(raw_data, DROP)
    assert not set(DROP) & set(prepared.columns)
    assert "machine_failure" in prepared.columns
    assert len(prepared) == len(raw_data)


def test_split_keeps_failure_rate_and_all_rows(raw_data: pd.DataFrame):
    df = prepare(raw_data, DROP)
    train, test = split(df, "machine_failure", test_size=0.25, seed=1)

    assert len(train) + len(test) == len(df)
    assert len(test) == 100
    # Stratified: both sets have (almost) the same share of failures.
    rate = df["machine_failure"].mean()
    assert abs(train["machine_failure"].mean() - rate) < 0.01
    assert abs(test["machine_failure"].mean() - rate) < 0.01


def test_split_is_reproducible(raw_data: pd.DataFrame):
    df = prepare(raw_data, DROP)
    first, _ = split(df, "machine_failure", test_size=0.2, seed=7)
    second, _ = split(df, "machine_failure", test_size=0.2, seed=7)
    pd.testing.assert_frame_equal(first, second)
