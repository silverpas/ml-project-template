# `src/`: the ML code

The pipeline steps. Each one is a small command-line program that reads
from a path and writes to a path, so it runs the same on a laptop and as an
Azure ML job.

| File | Step | Reads | Writes |
|---|---|---|---|
| `data.py` | Helper: **the one place that reads data** (CSV, Parquet, Delta table) and records which data it was | | |
| `prep.py` | Download the example (if needed), clean, split into train and test | `data.source` (via `data.py`) | `data/processed/` |
| `train.py` | Train the model, log everything to MLflow | `data/processed/train.parquet` | MLflow run, `outputs/train_run.json` |
| `evaluate.py` | Compare the new model with the approved one on the same test set | `data/processed/test.parquet` | `outputs/evaluation.json` |
| `register.py` | Register a better model as `candidate`; `--promote N` approves it as `champion` | `outputs/evaluation.json` | MLflow model registry |
| `registry.py` | Helper: the `status` tag (candidate, champion, archived) on model versions | | |
| `export.py` | Save the approved model as a standard MLflow folder, and check that it loads | MLflow model registry (champion) | `outputs/export/tool-wear-v<N>/` |
| `score.py` | Predict on new data with the approved model | a CSV, Parquet or Delta table (via `data.py`) | Delta table `data/predictions/ml_tool_wear_predictions/` (in Azure: the Gold table) |
| `cost.py` | `start` / `end` around a job: duration and estimated cost (config `cost`) | `outputs/job_start.json` | MLflow metrics `job_minutes`, `estimated_cost`; `outputs/cost.json` |
| `common.py` | Shared helpers: read config, set up MLflow, git commit, file fingerprint | | |

Run any step with `python -m src.<step>`. Add `--help` to see its options.
Settings come from `../config.yml`.

**For a new use case:** usually only `prep.py` changes, and only if your data
needs different cleaning (`prepare`). Data from a new kind of place (later:
Fabric tables) is a new branch in `data.py`, nowhere else. The model is saved with
MLflow 2 (`mlflow.sklearn.log_model`), the version Azure ML supports.
