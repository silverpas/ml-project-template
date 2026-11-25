"""Step 6: export the approved model as a standard MLflow model folder.

Usage:
    python -m src.export                 # the champion
    python -m src.export --version 2     # an older approved version (now archived)

Only approved versions (status champion or archived) can be exported.
The folder (default outputs/export/<model>-v<version>/) holds:
    MLmodel                   what the model is and how to load it (pyfunc flavor)
    model.pkl                 the model itself
    requirements.txt, conda.yaml, python_env.yaml   the exact packages it needs
    input_example.json        a few rows, to try it out
    export.json               which version, from which run, when, and its test metric

The same folder can be registered in Azure ML or Fabric, or used on a laptop:
    mlflow.pyfunc.load_model("outputs/export/tool-wear-v1").predict(df)
Before finishing, the script loads the folder and predicts on the example, so a
broken export fails here and not later.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import pandas as pd

from src.common import git_commit, load_config, setup_mlflow, write_json
from src.registry import ARCHIVED, CHAMPION, STATUS_TAG, find_version

APPROVED = {CHAMPION, ARCHIVED}


def approved_version(model_name: str, version: str | None) -> str:
    """The version to export: the champion, or `version` if it was ever approved."""
    if version is None:
        champion = find_version(model_name, CHAMPION)
        if champion is None:
            raise LookupError(f"No approved version of '{model_name}' yet: nothing to export.")
        return champion
    status = mlflow.MlflowClient().get_model_version(model_name, version).tags.get(STATUS_TAG)
    if status not in APPROVED:
        raise PermissionError(
            f"{model_name} version {version} has status '{status}': only approved "
            f"versions ({', '.join(sorted(APPROVED))}) can be exported."
        )
    return version


def export_model(model_name: str, version: str, out_dir: Path) -> Path:
    """Copy the registered model's files into out_dir/<name>-v<version>/."""
    folder = out_dir / f"{model_name}-v{version}"
    if folder.exists():
        shutil.rmtree(folder)  # exporting again gives a clean copy
    folder.parent.mkdir(parents=True, exist_ok=True)
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{model_name}/{version}", dst_path=str(folder)
    )
    flatten(folder)
    return folder


def flatten(folder: Path) -> None:
    """Put the model files (MLmodel, model.pkl, ...) directly in `folder`.

    The local registry downloads them there; the Azure ML registry puts them in
    a subfolder (folder/model/). Moving them up gives the same export everywhere.
    """
    if (folder / "MLmodel").exists():
        return
    found = list(folder.rglob("MLmodel"))
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one MLmodel file under {folder}, found {len(found)}")
    nested = found[0].parent
    for item in list(nested.iterdir()):
        shutil.move(str(item), folder / item.name)
    nested.rmdir()


def check_export(folder: Path) -> int:
    """Load the exported folder like any other user would, and predict on the example rows."""
    model = mlflow.pyfunc.load_model(str(folder))
    example = pd.read_json(folder / "input_example.json", orient="split")
    return len(model.predict(example))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yml", help="Settings file")
    parser.add_argument("--version", help="Approved version to export (default: the champion)")
    parser.add_argument("--out", default="outputs/export", help="Folder for the export")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    model_name = config["model"]["name"]
    setup_mlflow(config)

    version = approved_version(model_name, args.version)
    folder = export_model(model_name, version, Path(args.out))
    rows = check_export(folder)

    details = mlflow.MlflowClient().get_model_version(model_name, version)
    write_json(
        {
            "model_name": model_name,
            "version": version,
            "status": details.tags.get(STATUS_TAG),
            "run_id": details.run_id,
            "test_metrics": {k: v for k, v in details.tags.items() if k.startswith("test_")},
            "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "exported_by_commit": git_commit(),
        },
        folder / "export.json",
    )
    print(f"Exported {model_name} version {version} to {folder} (checked on {rows} example rows)")


if __name__ == "__main__":
    main()
