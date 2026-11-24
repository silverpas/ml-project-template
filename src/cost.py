"""Estimated cost of a job: how long the cloud machine ran × its price per hour.

Usage (first and last command of a job, see jobs/*.yml):
    python -m src.cost start     # writes outputs/job_start.json
    ...the pipeline steps...
    python -m src.cost end       # logs job_minutes and estimated_cost to MLflow

estimated_cost = (job minutes + extra_minutes) / 60 × price_per_hour

`extra_minutes` covers what the job itself cannot see: the machine starting up
and staying on for a while after the job (until the cluster scales down to 0).
On a laptop the cost is 0; only the duration is logged.

This is an ESTIMATE, to compare runs and spot expensive ones. The real bill is
in Azure Cost Management, per project thanks to the cluster's tags.
"""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from typing import Any

import mlflow

from src.common import load_config, read_json, setup_mlflow, write_json


def in_azure_ml_job() -> bool:
    """Inside an Azure ML job, Azure ML tells MLflow which run belongs to the job.

    ASSUMPTION: Azure ML sets MLFLOW_RUN_ID in every job (check in the sandbox).
    """
    return bool(os.environ.get("MLFLOW_RUN_ID"))


def estimate_cost(minutes: float, cost_cfg: dict[str, Any], on_azure: bool) -> float:
    """Billed minutes × price per hour. Zero on a laptop."""
    if not on_azure:
        return 0.0
    billed_minutes = minutes + cost_cfg["extra_minutes"]
    return round(billed_minutes / 60 * cost_cfg["price_per_hour"], 4)


def run_to_log_to(run_info: str) -> str | None:
    """The MLflow run that gets the cost.

    In Azure ML: the job's own run. On a laptop: the latest training run, if
    there is one (outputs/train_run.json); otherwise nothing is logged.
    """
    if in_azure_ml_job():
        return os.environ["MLFLOW_RUN_ID"]
    try:
        return read_json(run_info)["run_id"]
    except FileNotFoundError:
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["start", "end"], help="Start or end of the job")
    parser.add_argument("--config", default="config.yml", help="Settings file")
    parser.add_argument("--clock", default="outputs/job_start.json", help="Start time file")
    parser.add_argument("--run-info", default="outputs/train_run.json", help="Written by train")
    parser.add_argument("--out", default="outputs/cost.json", help="The estimate, as JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.action == "start":
        write_json({"started_at": datetime.now(UTC).isoformat()}, args.clock)
        return

    config = load_config(args.config)
    cost_cfg = config["cost"]
    started = datetime.fromisoformat(read_json(args.clock)["started_at"])
    minutes = round((datetime.now(UTC) - started).total_seconds() / 60, 2)
    on_azure = in_azure_ml_job()
    cost = estimate_cost(minutes, cost_cfg, on_azure)

    result = {
        "job_minutes": minutes,
        "estimated_cost": cost,
        "currency": cost_cfg["currency"],
        "vm_size": cost_cfg["vm_size"],
        "price_per_hour": cost_cfg["price_per_hour"],
        "extra_minutes": cost_cfg["extra_minutes"] if on_azure else 0,
        "where": "azure_ml" if on_azure else "local",
    }
    write_json(result, args.out)

    run_id = run_to_log_to(args.run_info)
    if run_id is not None:
        if not on_azure:  # in a job, Azure ML has already pointed MLflow at the workspace
            setup_mlflow(config)
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({"job_minutes": minutes, "estimated_cost": cost})
            mlflow.set_tags({f"cost_{k}": v for k, v in result.items() if k != "estimated_cost"})

    print(
        f"Job took {minutes} min on {result['where']}: "
        f"estimated cost {cost} {cost_cfg['currency']} ({cost_cfg['vm_size']})"
    )


if __name__ == "__main__":
    main()
