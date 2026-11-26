# Guide: building an ML project on the platform

This repo gives you a working example (use case `tool-wear`), but you don't
have to copy it line by line. This guide explains how a project is organised,
the few rules every project follows, and **three ways to run your code**.
Pick the way that suits you; you can mix them.

## 1. Your project at a glance

Your project already has a home on Azure: its own resource group with an Azure
ML workspace, a compute cluster and an identity. The platform team set it up
from your request. This repo is where your code lives.

```
config.yml          settings: data, target column, model, metrics, table names
src/                your code, one small script per step
tests/              automatic checks (pytest)
jobs/               Azure ML job files: what to run, on which data and machine
environment/        conda.yml: the Python packages for Azure ML
.github/workflows/  automation: train, approve, score every night
model-card.md       what the model is for, its data, metrics and approvals
```

The example code in `src/` is a complete project. Read it, keep what helps,
replace the rest.

## 2. Writing your code

Write your code the way you like: scripts, a package, notebooks turned into
scripts. The platform only cares about what goes in and what comes out.

### 2a. What your code must do

1. **Run as a command**, for example `python train.py` or
   `python -m mypackage.train`. That is what an Azure ML job and the GitHub
   workflows run.
2. **Read and write platform data through the shared data helper**
   (`src/data.py`), not with your own connection code. Today it reads files
   and Delta tables; later it also reads and writes Fabric tables.
3. **Log every training run to MLflow**: parameters, metrics, the data
   version, the git commit, and the model with its input signature.
4. **Save the model in MLflow format** (for example `mlflow.sklearn.log_model`).
   This is what lets the platform register it, approve it, **export** it as a
   standard folder and use it in Azure ML, Fabric or on a laptop.
5. **Register a better model as `candidate`**, and never set `champion`
   yourself: that happens only after approval (section 3).

### 2b. One way to do it: the example in this repo

The repo comes with a complete project that meets all of the above. Use it as
it is, adapt it, or replace it.

| Step | Script | What it does |
|---|---|---|
| Prepare | `src/prep.py` | Read the data, clean it, split it into train and test |
| Train | `src/train.py` | Train the model and log everything to MLflow |
| Evaluate | `src/evaluate.py` | Compare the new model with the approved one on the same test data |
| Register | `src/register.py` | Save a better model as a new version, marked `candidate` |
| Export | `src/export.py` | Save an approved model as a standard MLflow folder |
| Score | `src/score.py` | Make predictions with the approved model |

It comes with everything to run it: settings in `config.yml`, tests in
`tests/`, Azure ML job files in `jobs/` (`train-register.yml`,
`batch-score.yml`) and GitHub workflows in `.github/workflows/` (train,
approve, score every night). If you keep the same commands, all of it works
without changes. If you write your own code, change the `command` lines in
`jobs/*.yml` to call it. Two things the workflows expect from the code:
`train.yml` shows `outputs/evaluation.md` and reads the new version number
from `outputs/registration.json`, and `deploy.yml` calls
`python -m src.register --promote <version>` and `python -m src.export`. Keep
those (they work with any MLflow model) or change the workflows to match.

## 3. The rules every project follows

However you run your code, these stay the same. They are what makes a model
safe to use.

1. **Every training run is tracked in MLflow**: data version, code version,
   parameters, metrics and the model.
2. **A model is used only after approval.** A new model version starts as
   `candidate`. Only a reviewer can make it `champion`, and only the champion
   makes predictions. Approve through the GitHub workflow `deploy.yml`, never
   by changing the tag by hand.
3. **Compare fairly.** A new model must beat the current champion on the same
   test data before it becomes a candidate.
4. **No secrets and no real data in the repo.** Logins go through Azure, and
   `data/` is never committed.

## 4. Three ways to run your code

| | A. GitHub workflows | B. Your terminal | C. Azure ML studio |
|---|---|---|---|
| **What** | Buttons and schedules in the repo's Actions tab | Commands on your laptop | The Azure ML website |
| **Best for** | The real thing: training you want to keep, approvals, nightly predictions | Daily development and quick tries | Exploring, looking at results, trying something by clicking |
| **Who logs in** | The project's identity (no password) | You (`az login`) | You (browser) |
| **Leaves a record** | Yes: who ran what, when, with which code | In MLflow, not in GitHub | In MLflow, not in GitHub |

### A. GitHub workflows (for the real runs)

This is the ready-made way: the workflows in `.github/workflows/` run the job
files in `jobs/`, which run the example code (or yours, see 2b). Open the
repo's **Actions** tab:

- **Train (Azure ML)** → *Run workflow*: trains in Azure ML. If the new model
  is better, the run page says which version to approve.
- **Deploy (approval gate)** → *Run workflow* with that version: waits for a
  reviewer, then makes it the champion and exports it.
- **Score (nightly, Azure ML)**: makes predictions every night, or with *Run workflow*.

These workflows run the same job files (`jobs/*.yml`) you can run yourself in
B. They need the five Azure variables, which the platform team set up for you.

### B. Your terminal

**On your laptop, without Azure.** Good for writing and testing code:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.prep && python -m src.train && python -m src.evaluate && python -m src.register
python -m src.register --promote 1           # approve a local model (local only)
python -m src.score --input data/raw/ai4i2020.csv
pytest
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 8080   # see your runs
```

**In Azure ML, from your laptop.** The same code on the project's cluster:

```bash
az login
az configure --defaults group=<your resource group> workspace=<your workspace>
az ml job create --file jobs/train-register.yml
az ml job create --file jobs/batch-score.yml
```

The job files say what to run (`command`), on which data (`inputs`), with
which packages (`environment`) and on which machine (`compute`). Follow the
job in Azure ML studio. Step by step, with checks: [docs/azure-sandbox.md](docs/azure-sandbox.md).

### C. Azure ML studio (the website)

Open [ml.azure.com](https://ml.azure.com) and choose your project's workspace.

- **See your work:** *Jobs* shows every run with its metrics and logs;
  *Models* shows the registered versions and their `status` tag.
- **Run a job by clicking:** *Jobs* → create a new job. Upload your project
  folder as the code, give the command (for example
  `python -m src.prep --input ${{inputs.training_data}} && python -m src.train`),
  choose the environment that your earlier jobs created (or create one from
  `environment/conda.yml`), and choose your project's cluster.
- **Notebooks:** studio notebooks need a *compute instance*, a personal
  machine that costs money every hour it is on. Ask the platform team before
  you create one, and always turn on auto-shutdown. For exploring, a notebook
  on your laptop is usually enough.

Whatever you run in studio is tracked in MLflow like the rest. Approval still
goes through `deploy.yml` (rule 2).

## 5. Moving between the three ways

The code is the same everywhere. Only two things change:

- **Where MLflow writes.** On your laptop: `mlflow.db`. In Azure ML: your
  workspace. To work on the workspace's models from your laptop, install
  `requirements-azure.txt` and run
  `export MLFLOW_TRACKING_URI=$(az ml workspace show --query mlflow_tracking_uri -o tsv)`.
- **Where the data comes from.** On your laptop: a file in `data/`. In Azure
  ML: the job's `inputs`. Later: tables in Fabric, through `src/data.py`.

## 6. Starting a new use case

1. Put your data source, target column and columns to drop in `config.yml`.
2. Change `src/prep.py` if your data needs other cleaning.
3. Make the fake data in `tests/conftest.py` look like yours.
4. Run it on your laptop until `pytest` passes and the metrics make sense.
5. Run it once in Azure ML (B), then use the workflows (A).
6. Fill in `model-card.md`, and add a row there at every approval.
