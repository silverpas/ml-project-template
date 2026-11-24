"""Reading data: the ONE place that knows where data comes from.

Every script that needs input data calls `read_data`, and `describe_source`
records which data it was (for MLflow). To read from a new kind of place, add
a branch here; nothing else in the code has to change.

What `read_data` understands today:
    data/raw/ai4i2020.csv            a CSV file
    data/raw/ai4i2020.parquet        a Parquet file
    data/tables/sensors/             a Delta table (a folder with a _delta_log/)

Later (design section 9, not built yet):
    abfss://<workspace>@onelake.dfs.fabric.microsoft.com/<lakehouse>.Lakehouse/Tables/<table>
    a table in Fabric, read with a token for the job's identity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from deltalake import DeltaTable

from src.common import file_sha256


def is_onelake(source: str) -> bool:
    """True for a OneLake (Fabric) address."""
    return str(source).startswith("abfss://")


def is_delta_table(source: str | Path) -> bool:
    """A Delta table is a folder with a _delta_log/ folder inside."""
    return (Path(source) / "_delta_log").is_dir()


def read_data(source: str | Path) -> pd.DataFrame:
    """Read a CSV file, a Parquet file or a Delta table into a DataFrame."""
    if is_onelake(str(source)):
        raise NotImplementedError(
            "Reading from Fabric is not built yet (see the design, section 9). "
            "Use a local file or an uploaded file for now."
        )
    path = Path(source)
    if is_delta_table(path):
        return DeltaTable(str(path)).to_pyarrow_table(engine="datafusion").to_pandas()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        # 'utf-8-sig' strips the invisible BOM some files (like AI4I) start with.
        return pd.read_csv(path, encoding="utf-8-sig")
    raise ValueError(f"Don't know how to read '{source}': use .csv, .parquet or a Delta table.")


def describe_source(source: str | Path) -> dict[str, Any]:
    """Which data this was, so a training run can be traced back to it.

    - Delta table: its version number (read it again later with time travel,
      as long as the table still keeps that version).
    - File: its SHA-256 fingerprint (same content, same fingerprint).
    """
    path = Path(source)
    if is_delta_table(path):
        return {
            "source": str(source),
            "source_format": "delta",
            "source_version": DeltaTable(str(path)).version(),
        }
    return {
        "source": str(source),
        "source_format": path.suffix.lstrip("."),
        "source_sha256": file_sha256(path),
    }
