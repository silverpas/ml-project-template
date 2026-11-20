# Model card: tool-wear

One page that says what this model is, who owns it and whether it may be used.
Fill it in when the project starts (step 2) and update it at every approval and
quarterly review. The values below are the example on public data; replace them.

## Basics

| | |
|---|---|
| Use case | `tool-wear`: predict machine failures before they happen |
| Department | `manufacturing` |
| Model owner | [name] |
| Reviewer (approval gate) | [name] |
| Risk tier | Decision support: predictions shown in reports, a person decides |
| Request | [link to the request issue in the ml-requests repo] |

Risk tiers: *exploration* (cannot go live), *decision support* (owner + reviewer
approve), *production-critical* (also operations sign-off and a fallback plan).

## Purpose and use

- **What it predicts:** the probability that a machine fails, per row of sensor data.
- **Who uses it, how:** maintenance planners see `failure_probability` and
  `predicted_failure` in a Power BI report and decide which machines to check.
- **Not for:** stopping machines automatically. That would make it production-critical.

## Data

| | |
|---|---|
| Source | AI4I 2020 Predictive Maintenance (UCI, id 601, public, CC BY 4.0) |
| Training data version | [OneLake path, e.g. `Files/ml/tool-wear/2026-11-20/`, and `raw_sha256` from the MLflow run] |
| Target | `machine_failure` (about 3% of rows are failures) |
| Excluded columns | IDs (`udi`, `product_id`) and failure reasons (`twf`, `hdf`, `pwf`, `osf`, `rnf`): known only after a failure |

## Model and metrics

| | |
|---|---|
| Algorithm | LightGBM classifier, failures weighted up (`class_weight: balanced`) |
| Deciding metric | PR-AUC (`average_precision`) on the test set; never accuracy |
| Threshold | 0.5 on the probability → `predicted_failure` |

Results of the approved version (from `outputs/evaluation.md` or the Train run page):

| Version | PR-AUC | ROC-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| 1 (example, local run) | 0.838 | 0.975 | 0.779 | 0.679 | 0.726 |

## Known limits

- Trained on synthetic public data: it says nothing about real machines.
- About 1 in 3 failure alarms is false (precision 0.68), and about 1 in 5 real failures is missed (recall 0.78).
- New machine types, sensors or products: retrain before trusting the predictions.

## Approvals and reviews

Add a row at every approval (Deploy run) and every quarterly review.

| Date | Version | Decision | By | Link |
|---|---|---|---|---|
| [date] | 1 | Approved as champion | [reviewer] | [Deploy run] |
