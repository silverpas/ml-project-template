# AGENTS.md: ML project template

Instructions for the people who build a model in this repo, and for their AI
coding agents. Read this before changing anything.

## What this repo is for

One repo = one ML use case. It holds the code that turns data into an approved
model and then into predictions:

prepare data → train → evaluate → register → (human approval) → score.

Everything runs **locally first** (laptop, local MLflow, local files) and then
the same scripts run as **Azure ML jobs**. The example predicts machine
failures on the public AI4I 2020 dataset (use case `tool-wear`).

Jargon in one line each:
- **MLflow**: a logbook for training runs plus a registry (shelf) of model versions.
- **Azure ML job**: a script that runs on a cloud computer instead of your laptop.
- **Alias** `candidate` / `champion`: labels on model versions: "better, waiting
  for approval" / "approved, in use".
- **OIDC**: GitHub logs in to Azure without a stored password.

## Where this repo sits in the overall process

| Step | What happens | This repo? |
|---|---|---|
| 0. Explore | Try the idea anywhere, no request needed | No |
| 1. Request | Someone fills in the request form | No (ML requests repo) |
| 2. Qualify | Platform lead says go; Azure is set up; this repo is created | Created here |
| 3. Data and priority | Data access is granted | Uses it |
| 4. Build | Train, evaluate, register, approval gate | **Yes, the main work** |
| 5. Run and review | Nightly scoring, quarterly review | **Yes** (`score.yml`, `review-reminder.yml`) |

## Folder structure

```
config.yml          all settings (data, target, columns to drop, model, metric)
src/                prep.py, train.py, evaluate.py, register.py, score.py, common.py
tests/              pytest checks on small fake data
.github/workflows/  ci.yml, demo-local-flow.yml (demo only)
                    train.yml, deploy.yml, score.yml, review-reminder.yml  (milestone 3)
jobs/               train-register.yml, batch-score.yml   (Azure ML CLI v2, milestone 3)
environment/        conda.yml: packages for the Azure ML jobs   (milestone 3)
notebooks/          fabric_batch_score.ipynb: OPTIONAL option A   (milestone 3)
model-card.md       owner, purpose, data version, metrics, approvals   (milestone 3)
data/, outputs/, mlflow.db, mlruns/   local only, never committed
```

## Workflow for users

1. In `config.yml`, set your data source, target column, columns to drop and model name.
2. Adapt `src/prep.py` only if your data needs different cleaning.
3. Change the fake data in `tests/conftest.py` to your columns.
4. Run the pipeline locally (commands below) until tests pass and metrics make sense.
5. Push to `main`. CI checks the code. In Azure (milestone 3), `train.yml`
   trains and evaluates; if the new model is better it becomes `candidate`.
6. A reviewer approves in GitHub (`production` environment) → `champion`.
7. `score.yml` scores every night with the champion and writes predictions.
8. Every quarter `review-reminder.yml` opens an issue: keep, retrain or retire?

**Scoring:** the default is an **Azure ML batch job** (`jobs/batch-score.yml`)
that writes to the predictions lakehouse in Fabric. The notebook in
`notebooks/` (scoring inside Fabric with PREDICT) is an optional alternative.

**Demo:** `demo-local-flow.yml` runs train → approval → score on GitHub's own
computers with no Azure. It shows the approval gate (environment
`approval-demo`). It is **not** the production pipeline; delete it once Azure works.

## Commands

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt              # macOS: brew install libomp first
python -m src.prep                           # downloads the data if missing
python -m src.train
python -m src.evaluate                       # writes outputs/evaluation.json and .md
python -m src.register                       # registers as "candidate" if better
python -m src.register --promote <version>   # human approval: set "champion"
python -m src.score --input data/raw/ai4i2020.csv --output outputs/predictions.parquet
pytest
ruff check . && ruff format --check .
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 8080
```

Each script is a CLI: `python -m src.<step> --help` shows its options.

## Azure settings

No infrastructure code lives here. The platform team creates the Azure
resources and gives you these IDs; store them as **GitHub variables** (not secrets):
`AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`,
`AZURE_RESOURCE_GROUP`, `AZURE_ML_WORKSPACE`.

## Rules (what not to do)

- **No secrets** in code, config or commits. Azure login only via OIDC.
- **No real company data** in the repo; `data/` is git-ignored.
- **No Terraform or other infrastructure code** in this repo.
- **Azure ML SDK v2 / CLI v2 only** (`azure-ai-ml`, YAML job specs). Never SDK v1 (`azureml-core`).
- **No leaking features**: never use IDs or the failure-reason columns
  (TWF, HDF, PWF, OSF, RNF) as inputs; they are only known after a failure.
- **Not only accuracy**: failures are rare, so report recall, precision, F1 and
  PR-AUC. PR-AUC (`average_precision`) decides if a model is better.
- **Compare fairly**: `evaluate.py` compares the new model with the approved one
  on the same test set. Do not skip it.
- **Never promote without approval**: `champion` is set only after a reviewer approves.
- **Every training run logs** parameters, metrics, the data version and the git commit.
- Scripts read inputs from paths and write outputs to paths (`argparse`), so they
  run the same locally and in Azure ML. Settings go in `config.yml`, not in code.
- Ask before adding a dependency; pin its version in `requirements.txt`.
- Keep code simple: small functions, type hints, short docstrings.
- When you change this repo's structure or workflow, update this file in the same change.
