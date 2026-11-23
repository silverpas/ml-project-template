# `.github/workflows/`: automation

Workflows are recipes GitHub runs by itself, on its own computers, when
something happens (a push, a button click, a schedule). See them run in the
repo's **Actions** tab.

| File | Starts when | What it does | Needs Azure? |
|---|---|---|---|
| `ci.yml` | Every push and pull request | Runs ruff and pytest; ✅ or ❌ on the commit | No |
| `train.yml` | "Run workflow" button | Submits `jobs/train-register.yml` to Azure ML; shows the comparison and the version to approve | Yes |
| `deploy.yml` | "Run workflow" button, with a version | Approval gate (`production` environment), then promotes the version to champion | Yes |
| `score.yml` | Every night at 02:00 UTC, or the button | Submits `jobs/batch-score.yml` to Azure ML; predictions go to the Gold table in Fabric | Yes |
| `review-reminder.yml` | 1 Jan, Apr, Jul, Oct, or the button | Opens an issue: "review this model: keep, retrain or retire?" (not in the template repo itself) | No |
| `demo-local-flow.yml` | Only the "Run workflow" button (manual) | **Demo only.** The whole flow on GitHub's computers: train → approval → score | No |

## The approval gate

A deploy job uses a GitHub **environment** with **required reviewers**. GitHub
pauses the run there and shows a **Review deployments** button to the
reviewers. Nothing is promoted until one of them approves.

- `production`: the real gate, used by `deploy.yml`.
- `approval-demo`: the demo gate, used only by `demo-local-flow.yml`.

Set them up once under *Settings → Environments*. On private repositories
required reviewers need a paid GitHub plan; on public ones they are free.

## Azure login

The Azure workflows log in with **OIDC** (GitHub proves who it is to Azure; no
password is stored). They read these **GitHub variables** (IDs, not secrets),
which the platform team gives you:
`AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`,
`AZURE_RESOURCE_GROUP`, `AZURE_ML_WORKSPACE`. Until they are set, the Azure
workflows are skipped (shown as grey, not red).

**For a new use case:** add your reviewers to the `production` environment and
set the five variables. Delete `demo-local-flow.yml` once Azure works.
