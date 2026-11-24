# AGENTS.md: ML project template

Instructions for the people who build a model in this repo, and for their AI
coding agents. Read this before changing anything.

## What this repo is for

One repo = one ML use case. It holds the code that turns data into an approved
model and then into predictions:

prepare data → train → evaluate → register → (human approval) → export → score.

Everything runs **locally first** (laptop, local MLflow, local files) and then
the same scripts run as **Azure ML jobs**. The example predicts machine
failures on the public AI4I 2020 dataset (use case `tool-wear`).

Jargon in one line each:
- **MLflow**: a logbook for training runs plus a registry (shelf) of model versions.
- **Azure ML job**: a script that runs on a cloud computer instead of your laptop.
- **Status tag** `candidate` / `champion` / `archived`: label on a model version:
  "better, waiting for approval" / "approved, in use" / "replaced" (`src/registry.py`).
- **Datastore**: Azure ML's named shortcut to a storage place, e.g. a OneLake lakehouse.
- **Delta table**: Parquet files plus a log of versions; how Fabric stores tables.
- **Gold table**: the clean, ready-to-use data layer in Fabric; tables written by ML start with `ml_`.
- **OIDC**: GitHub logs in to Azure without a stored password.

## Where this repo sits in the overall process

| Step | What happens | This repo? |
|---|---|---|
| 0. Explore | Try the idea on a laptop, no request needed | No |
| 1. Request | Someone fills in the request form | No (ML requests repo) |
| 2. Qualify | Platform lead says go | No |
| 3. Data and priority | Data access is granted; then Azure is set up and this repo is created | Created here |
| 4. Build | Train, evaluate, register, approval gate | **Yes, the main work** |
| 5. Run and review | Nightly scoring, quarterly review | **Yes** (`score.yml`, `review-reminder.yml`) |

## Folder structure

```
config.yml          all settings (data, target, columns to drop, model, metric,
                    predictions_table)
src/                prep.py, train.py, evaluate.py, register.py, export.py, score.py,
                    data.py (the ONE place that reads data), registry.py
                    (status tags), common.py (shared helpers)
tests/              pytest checks on small fake data
.github/workflows/  ci.yml, train.yml, deploy.yml, score.yml, review-reminder.yml,
                    demo-local-flow.yml (demo only)
jobs/               train-register.yml, batch-score.yml   (Azure ML CLI v2 job specs)
environment/        conda.yml: packages for the Azure ML jobs (same versions as requirements.txt)
notebooks/          fabric_batch_score.ipynb: OPTIONAL option A (PREDICT in Fabric)
model-card.md       owner, purpose, data version, metrics, approvals
requirements.txt    pinned packages; pyproject.toml: ruff settings
*/README.md         short guides inside src/, tests/ and .github/workflows/
data/, outputs/, mlflow.db, mlruns/   local only, never committed
                    (data/predictions/<predictions_table>/ = local stand-in for the Gold table)
```

## Workflow for users

1. In `config.yml`, set your data `source` (CSV, Parquet or Delta table), target
   column, columns to drop, model name (= the use-case name, e.g. `tool-wear`)
   and `predictions_table` (= the one in your registry file).
2. Adapt `src/prep.py` only if your data needs different cleaning.
3. Change the fake data in `tests/conftest.py` to your columns, and the column
   lists in `tests/test_rules.py` to your leaking and ID columns.
4. Run the pipeline locally (commands below) until tests pass and metrics make sense.
5. Fill in `model-card.md`. In `jobs/*.yml`, check every `# ASSUMPTION:` line
   (datastore names, data paths, the Gold table's OneLake address, compute)
   with the platform team.
6. Push to `main`: CI checks the code. Run **Train** (`train.yml`): Azure ML
   trains and evaluates; if the new model is better it becomes `candidate`.
7. Run **Deploy** (`deploy.yml`) with that version: a reviewer approves in
   GitHub (`production` environment) → `champion` → exported as a standard
   MLflow folder (`model-export` download). Add a row to `model-card.md`.
8. `score.yml` scores every night with the champion and replaces the rows of the Gold table.
9. Every quarter `review-reminder.yml` opens an issue: keep, retrain or retire?

**Scoring:** the default is an **Azure ML batch job** (`jobs/batch-score.yml`)
that writes the predictions as a **Delta table** (Fabric's table format) into
the Gold lakehouse, e.g. `ml_tool_wear_predictions`; Power BI reads it directly.
The notebook in `notebooks/` (scoring inside Fabric with PREDICT) is optional.

**Demo:** `demo-local-flow.yml` runs train → approval → score on GitHub's own
computers with no Azure. It shows the approval gate (environment
`approval-demo`). It runs only when started by hand ("Run workflow" on the
Actions tab). It is **not** the production pipeline; delete it once Azure works.

## Commands

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt              # macOS: brew install libomp first
python -m src.prep                           # downloads the data if missing
python -m src.train
python -m src.evaluate                       # writes outputs/evaluation.json and .md
python -m src.register                       # registers as "candidate" if better
python -m src.register --promote <version>   # human approval: set "champion"
python -m src.export                         # champion -> outputs/export/tool-wear-v<N>/
az ml job create --file jobs/train-register.yml   # same steps in Azure ML (after az login)
python -m src.score --input data/raw/ai4i2020.csv   # -> data/predictions/ml_tool_wear_predictions/
pytest
ruff check . && ruff format --check .
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 8080
```

Each script is a CLI: `python -m src.<step> --help` shows its options.

## Azure settings

No infrastructure code lives here. The platform team creates the Azure
resources and gives you these IDs; store them as **GitHub variables** (not secrets):
`AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`,
`AZURE_RESOURCE_GROUP`, `AZURE_ML_WORKSPACE`. Without them the Azure workflows
are skipped, so a new repo stays green. Also create the `production` environment
with the reviewers as required reviewers. The platform team also grants the
project's identity read access to the input data and write access to the Gold
table in Fabric.

## Rules (what not to do)

- **No secrets** in code, config or commits. Azure login only via OIDC.
- **No real company data** in the repo; `data/` is git-ignored.
- **No Terraform or other infrastructure code** in this repo.
- **Azure ML SDK v2 / CLI v2 only** (`azure-ai-ml`, YAML job specs). Never SDK v1 (`azureml-core`).
- **No leaking features**: never use IDs or the failure-reason columns
  (TWF, HDF, PWF, OSF, RNF) as inputs; they are only known after a failure.
  `tests/test_rules.py` fails if `config.yml` stops dropping them.
- **Not only accuracy**: failures are rare, so report recall, precision, F1 and
  PR-AUC. PR-AUC (`average_precision`) decides if a model is better; never accuracy.
- **Compare fairly**: `evaluate.py` compares the new model with the approved one
  on the same test set. Do not skip it.
- **Never promote without approval**: `champion` is set only after a reviewer approves.
  Only approved versions can be exported (`src/export.py` refuses the others).
- **Every training run logs** parameters, metrics, the data version and the git commit.
- **Read data only through `src/data.py`** (`read_data`), so a new source is one change.
- **Same packages everywhere**: a version change goes in `requirements.txt` AND
  `environment/conda.yml` (`tests/test_azure_files.py` checks they match).
- **Predictions go to an `ml_` Gold table**; `tests/test_rules.py` checks the name
  and that `jobs/batch-score.yml` writes to the same table.
- **Mark guesses** about the platform (datastore names, paths) with `# ASSUMPTION:`.
- Scripts read inputs from paths and write outputs to paths (`argparse`), so they
  run the same locally and in Azure ML. Settings go in `config.yml`, not in code.
- Ask before adding a dependency; pin its version in `requirements.txt`.
- Keep code simple: small functions, type hints, short docstrings.
- When you change this repo's structure or workflow, update this file in the same change.
