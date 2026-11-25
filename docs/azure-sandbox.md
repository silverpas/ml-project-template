# Running the project in Azure ML, by hand

The whole flow in an Azure ML workspace, started from your laptop: train,
approve, score, export, and check the cost. No GitHub workflows and no Fabric
needed. Good for a first try in a test workspace, or to see what the
workflows do step by step.

| Step | Command | Runs where |
|---|---|---|
| 1. Train | `az ml job create --file jobs/train-register.yml` | Azure ML, the project's cluster |
| 2. Approve | `python -m src.register --promote <version>` | your laptop, on the Azure ML registry |
| 3. Score | `az ml job create --file jobs/batch-score.yml` | Azure ML, the project's cluster |
| 4. Export | `python -m src.export` | your laptop, from the Azure ML registry |
| 5. Cost | `estimated_cost` metric on each job | Azure ML studio |

The data is the example CSV on your laptop: each job uploads it. Reading from
Fabric comes later (design section 9).

## Before you start

- An Azure ML workspace and the project's cluster (`ml-project` module; its
  `cluster` output gives the name, e.g. `cpu-tool-wear`). If the name differs,
  change `compute:` in both `jobs/*.yml`.
- The project identity may need **AcrPull** on the workspace's container
  registry, so the cluster can load the job environment. If the module ran with
  `create_role_assignments = false`, someone with the right assigns it (see its
  `roles_to_assign_by_hand` output).
- Quota for the cluster's machine type in that region (Azure portal → the
  workspace → Quotas).
- The Azure CLI with the ML extension: `az extension add --name ml`.

```bash
source .venv/bin/activate
pip install -r requirements-azure.txt   # adds azureml-mlflow (MLflow ↔ Azure ML)
python -m src.prep                      # downloads data/raw/ai4i2020.csv once

az login
az configure --defaults group=<resource group> workspace=<workspace>
```

In `config.yml`, set `cost.vm_size` to the cluster's machine type and
`cost.price_per_hour` to its price in your region (Azure pricing page).

## 1. Train

```bash
az ml job create --file jobs/train-register.yml \
  --set environment_variables.GIT_COMMIT=$(git rev-parse HEAD) --stream
```

The first job is slow (10-20 minutes): Azure ML builds the Python environment
once and starts a machine. Later jobs reuse the environment.

In Azure ML studio (https://ml.azure.com) → **Jobs** → `tool-wear`: parameters,
metrics (`test_average_precision`, `job_minutes`, `estimated_cost`), the data
fingerprint (`data_source_sha256`) and the logs. → **Models** → `tool-wear`:
version 1 with the tag `status: candidate`.

## 2. Approve (the approval gate, by hand)

Point MLflow on your laptop at the workspace. From then on, the same commands
work on the Azure ML registry instead of the local `mlflow.db`:

```bash
export MLFLOW_TRACKING_URI=$(az ml workspace show --query mlflow_tracking_uri -o tsv)
python -m src.register --promote 1
```

In studio, version 1 now has `status: champion`. (In a real project this step
runs in `deploy.yml`, only after a reviewer approves in GitHub.)

## 3. Score

```bash
az ml job create --file jobs/batch-score.yml --stream
az ml job download --name <job name> --output-name predictions --download-path outputs/azure
```

The predictions are a Delta table, `ml_tool_wear_predictions`, in the job's
output `predictions`. Read the download on your laptop:

```bash
python -c "from src.data import read_data; import glob; \
print(read_data(glob.glob('outputs/azure/**/ml_tool_wear_predictions', recursive=True)[0]).head())"
```

## 4. Export

```bash
python -m src.export        # still with MLFLOW_TRACKING_URI set
```

`outputs/export/tool-wear-v1/` is a standard MLflow model folder: it loads with
`mlflow.pyfunc.load_model(...)` on any computer, and can be registered in Azure
ML or Fabric.

## 5. Cost

- Each job's `estimated_cost` (studio → the job → Metrics) is an estimate:
  job minutes plus `cost.extra_minutes`, times `cost.price_per_hour`.
- The real cost shows up in the Azure portal → **Cost Management** a day later.
  Filter by the tag `use_case = tool-wear`: the project's cluster carries it.

## Back to local

```bash
unset MLFLOW_TRACKING_URI   # MLflow uses the local mlflow.db again
```

The cluster switches its machines off by itself after the idle time (scales to
0). Nothing else keeps running.

## What a first run confirmed

Checked in a test workspace (CPU cluster, AI4I data, November 2026):

- [x] The cluster loads the environment with the project identity (AcrPull).
- [x] Azure ML sets `MLFLOW_RUN_ID` in a job, so `src/cost.py` logs
      `job_minutes` and `estimated_cost` on the job's run.
- [x] The `status` tags (candidate, champion) work in the Azure ML registry.
- [x] MLflow 3 cannot save models in Azure ML (404 on `logged-models`): the
      template uses MLflow 2.16, the newest version Azure ML supports.
- [x] `src/export.py` downloads the champion from the Azure ML registry. Azure
      puts the files in a subfolder `model/`; the script moves them up.
- [x] Same data and code give the same model as on the laptop (PR-AUC 0.838,
      same data fingerprint, 360 predicted failures).

## When something fails

| Message mentions | Likely cause | What to do |
|---|---|---|
| quota, `QuotaExceeded` | Not enough cores for the machine type | A smaller `vm_size`, or ask for quota |
| `ImagePull`, `unauthorized`, `acr` | The cluster identity cannot read the registry | Ask for AcrPull on the registry for the project identity |
| `compute ... not found` | Another cluster name | Fix `compute:` in `jobs/*.yml` |
| `No approved version` | Nothing promoted yet, or MLflow points at local | Step 2; check `echo $MLFLOW_TRACKING_URI` |
| the input path | `data/raw/ai4i2020.csv` missing | `python -m src.prep` once |
| `logged-models` … `404` | MLflow 3 in the environment: Azure ML supports MLflow up to 2.16 | Keep `mlflow==2.16.2` in `requirements.txt` and `environment/conda.yml` |
