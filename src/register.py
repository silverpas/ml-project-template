"""Step 4: register the model, and promote approved versions.

Two modes:

    python -m src.register
        Reads outputs/evaluation.json. If the new model is a candidate, adds it
        to the MLflow model registry with the tag status=candidate.

    python -m src.register --promote 3
        Approval: tags version 3 status=champion (the previous champion becomes
        archived). Scoring uses the champion. In GitHub this runs only after a
        reviewer approves (deploy.yml). See src/registry.py.
"""

from __future__ import annotations

import argparse

import mlflow

from src.common import load_config, read_json, setup_mlflow, write_json
from src.registry import CANDIDATE, CHAMPION, promote, set_status


def register_candidate(evaluation: dict, model_name: str) -> str | None:
    """Register the evaluated model if it is a candidate. Returns the new version, or None."""
    if not evaluation["is_candidate"]:
        return None
    version = mlflow.register_model(evaluation["model_uri"], model_name)
    set_status(model_name, version.version, CANDIDATE)
    metric = evaluation["primary_metric"]
    mlflow.MlflowClient().set_model_version_tag(
        model_name, version.version, f"test_{metric}", str(evaluation["new_metrics"][metric])
    )
    return str(version.version)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yml", help="Settings file")
    parser.add_argument(
        "--evaluation", default="outputs/evaluation.json", help="Written by evaluate"
    )
    parser.add_argument(
        "--out", default="outputs/registration.json", help="Which version was registered"
    )
    parser.add_argument("--promote", metavar="VERSION", help="Make this version the champion")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    model_name = config["model"]["name"]
    setup_mlflow(config)

    if args.promote:
        promote(model_name, args.promote)
        print(f"{model_name} version {args.promote} is now '{CHAMPION}'")
        return

    version = register_candidate(read_json(args.evaluation), model_name)
    # Tells the next step (e.g. the approval job on GitHub) which version to approve.
    write_json({"model_name": model_name, "version": version}, args.out)
    if version is None:
        print("Not registered: the new model is not better than the champion.")
    else:
        print(
            f"Registered {model_name} version {version} as '{CANDIDATE}'. "
            f"To approve it: python -m src.register --promote {version}"
        )


if __name__ == "__main__":
    main()
