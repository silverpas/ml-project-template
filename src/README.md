# `src/`: the ML code

The five pipeline steps. Each one is a small command-line program that reads
from a path and writes to a path, so it runs the same on a laptop and as an
Azure ML job.

| File | Step | Reads | Writes |
|---|---|---|---|
| `prep.py` | Download (if needed), clean, split into train and test | raw CSV | `data/processed/` |
| `train.py` | Train the model, log everything to MLflow | `data/processed/train.parquet` | MLflow run, `outputs/train_run.json` |
| `evaluate.py` | Compare the new model with the approved one on the same test set | `data/processed/test.parquet` | `outputs/evaluation.json` |
| `register.py` | Register a better model as `candidate`; `--promote N` approves it as `champion` | `outputs/evaluation.json` | MLflow model registry |
| `score.py` | Predict on new data with the approved model | any CSV or Parquet file | `data/predictions/predictions.parquet` |
| `common.py` | Shared helpers: read config, set up MLflow, git commit, file fingerprint | | |

Run any step with `python -m src.<step>`. Add `--help` to see its options.
Settings come from `../config.yml`.

**For a new use case:** usually only `prep.py` changes, and only if your data
needs different cleaning (`load_raw`, `prepare`). If you change the model type,
also update `TRUSTED_MODEL_TYPES` in `train.py`.
