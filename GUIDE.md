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

**One small script per step.** The example has:

| Step | Script | What it does |
|---|---|---|
| Prepare | `src/prep.py` | Read the data, clean it, split it into train and test |
| Train | `src/train.py` | Train the model and log everything to MLflow |
| Evaluate | `src/evaluate.py` | Compare the new model with the approved one on the same test data |
| Register | `src/register.py` | Save a better model as a new version, marked `candidate` |
| Export | `src/export.py` | Save the approved model as a standard MLflow folder |
| Score | `src/score.py` | Make predictions with the approved model |

You can have fewer or more steps. A few habits make your code run the same
everywhere (laptop, terminal, Azure ML, GitHub):

- **Paths in, paths out.** Each script reads its inputs from paths and writes
  its outputs to paths given on the command line (`argparse`), for example
  `python -m src.prep --input data/raw/file.csv --out data/processed`.
- **Settings in `config.yml`**, not in the code: column names, model
  parameters, the metric that decides which model is better.
- **Read data in one place** (`src/data.py`), so a new data source is one change.
- **Log with MLflow**: parameters, metrics, the data version, the git commit
  and the model. You don't need to set anything up: locally MLflow writes to
  `mlflow.db`, and inside Azure ML it writes to your workspace by itself.
- **Write tests** for your own logic in `tests/`, and run `pytest` before you push.

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

Open the repo's **Actions** tab:

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
