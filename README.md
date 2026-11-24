# ML project template

A starting point for every new machine learning use case on a Fabric data
platform with Azure Machine Learning. One GitHub repo per use case: create it
from this template, swap the example for your own data, and keep the rest.

The example inside (use case `tool-wear`, department `manufacturing`) predicts
**machine failures** from sensor readings.

**Data credit:** the example uses the public
[AI4I 2020 Predictive Maintenance dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
(UCI Machine Learning Repository, id 601, CC BY 4.0).

## How a model goes from data to report

```
Fabric lakehouse (training data snapshot)
  → Azure ML: prep → train → evaluate        only a better model continues
  → registered as "candidate"
  → reviewer approves in GitHub               the model becomes "champion"
  → exported as a standard MLflow folder      runs in Azure ML, Fabric or locally
  → Azure ML: nightly batch scoring with the champion   (the default)
  → Fabric lakehouse (predictions table) → Power BI
```

Everything also runs **on your laptop**, without Azure: local files instead of
the lakehouse, and a local MLflow store instead of the Azure ML workspace.

The Azure infrastructure (workspace, compute, login identity) is **not** in
this repo. It already exists when your project starts; the platform team sets
it up and gives you five IDs to store as GitHub variables.

## Run it locally

Needs Python 3.11. On macOS, LightGBM also needs `brew install libomp`.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.prep          # download and prepare the data
python -m src.train         # train a model
python -m src.evaluate      # compare it with the approved model
python -m src.register      # register it as "candidate" if it is better
python -m src.register --promote 1     # approve version 1 as "champion"
python -m src.score --input data/raw/ai4i2020.csv   # -> data/predictions/ml_tool_wear_predictions/

pytest                      # run the tests
ruff check . && ruff format --check .   # check code style
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 8080   # browse runs at localhost:8080
```

## Demo: the whole flow on GitHub, no Azure needed

`demo-local-flow.yml` is **a demo, not the production pipeline**. It runs
train → approval → scoring on GitHub's own computers, so you can see the
approval gate work before your project has Azure access:

1. *Actions* tab → *Demo flow (no Azure, not production)* → **Run workflow**.
2. The run page shows the new model next to the current champion, and the decision.
3. If the new model is better, the run pauses at **Review deployments**
   (GitHub environment `approval-demo`): a reviewer approves or rejects.
4. After approval the model becomes champion; the predictions can be
   downloaded from the run page.

To see a second model compete, change a setting under `model.params` in
`config.yml`, push, and run the demo again. Delete the demo once your project
runs in Azure.

## Run it in Azure ML

Once the platform team has set up Azure and you have stored the five GitHub
variables and created the `production` environment:

1. *Actions* → **Train (Azure ML)** → Run workflow. Azure ML runs
   `jobs/train-register.yml`; the run page shows the comparison and, if the
   model is better, the version to approve.
2. *Actions* → **Deploy (approval gate)** → Run workflow with that version.
   A reviewer approves; the version becomes champion.
3. **Score (nightly)** then runs every night and writes predictions to the
   Gold table `ml_tool_wear_predictions` in Fabric. **Review reminder** opens an
   issue every quarter.

Before the first run, check every `# ASSUMPTION:` line in `jobs/*.yml` with the
platform team (datastore names, data paths, compute name).

## Folders

| Path | What it is | Change it for a new use case? |
|---|---|---|
| `config.yml` | All settings: data, target column, columns to drop, model, metric | **Yes**, the main place |
| [`src/`](src/README.md) | The five pipeline steps | Only `prep.py`, if your data needs different cleaning |
| [`tests/`](tests/README.md) | Automatic checks on fake data | Adapt the fake data, add tests for your cleaning |
| `requirements.txt` | Python packages with exact versions | Rarely |
| `pyproject.toml` | Settings for pytest and ruff | No |
| [`.github/workflows/`](.github/workflows/README.md) | CI, the Azure workflows (train, deploy, score, review) and the demo | No |
| `jobs/`, `environment/` | Azure ML job files and packages | Check the `# ASSUMPTION:` lines |
| `notebooks/` | Optional: scoring inside Fabric instead of Azure ML | Only if you use it |
| `model-card.md` | Owner, purpose, data version, metrics, approvals | **Yes**, fill it in |

Created locally and never committed: `data/` (including `data/predictions/`, the local
stand-in for the Gold predictions table), `outputs/`, `mlflow.db`, `mlruns/`, `.venv/`.

## Start a new use case

1. Your request is approved (via the ML requests process) and the platform
   team has set up Azure for your use case.
2. Create a repo from this template ("Use this template" on GitHub).
3. In `config.yml`, change the `data` section (source, target, columns to drop)
   and `model.name`.
4. If your data needs other cleaning, adapt `prepare` in `src/prep.py`. Reading
   the data (file or Delta table) happens in `src/data.py`.
5. Change the fake data in `tests/conftest.py` to your columns.
6. Run the pipeline and `pytest` locally until the tests pass and the metrics make sense.
7. Set the five GitHub variables and add reviewers to the `production` environment.

## Rules

- **No secrets** in code, config or commits. Azure login goes through OIDC.
- **No real company data** in the repo. `data/` is never committed.
- **No infrastructure code** here: Terraform lives in the platform team's repo.
- **A model is used only after approval**: it must beat the current model on
  the same test set, and a reviewer must approve it.
- **Every run is traceable**: MLflow records the data fingerprint, the git
  commit, the settings and the metrics.
