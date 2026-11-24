# `tests/`: automatic checks

Run them with `pytest` from the repo root. They need no internet and no Azure:
they use small fake data and a temporary MLflow store.

| File | Checks |
|---|---|
| `conftest.py` | Shared setup: fake data shaped like the real dataset, and a test config |
| `test_data.py` | Reading CSV, Parquet and Delta tables gives the same data; the data version is recorded |
| `test_prep.py` | Column cleaning, leaking columns are dropped, the split keeps the failure rate |
| `test_evaluate.py` | Metrics, and the rule "only a better model becomes a candidate" |
| `test_pipeline.py` | All five steps end to end: prep → train → evaluate → register → score; the model has no leaking inputs |
| `test_export.py` | Only approved versions can be exported; the folder loads with plain MLflow |
| `test_registry.py` | Status tags: one champion at a time, the previous one is archived |
| `test_azure_files.py` | Azure files without Azure: job specs complete, same packages as `requirements.txt`, OIDC only, no secrets |
| `test_rules.py` | The real `config.yml` follows the rules: leaking and ID columns dropped, a supported primary metric (never accuracy) |

**For a new use case:** change `make_raw_data()` in `conftest.py` so the fake
data has your columns, update `LEAKING_COLUMNS` and `ID_COLUMNS` in `test_rules.py`,
and add a test for any new cleaning you write in `prep.py`.
