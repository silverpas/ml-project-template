# `tests/`: automatic checks

Run them with `pytest` from the repo root. They need no internet and no Azure:
they use small fake data and a temporary MLflow store.

| File | Checks |
|---|---|
| `conftest.py` | Shared setup: fake data shaped like the real dataset, and a test config |
| `test_prep.py` | Column cleaning, leaking columns are dropped, the split keeps the failure rate |
| `test_evaluate.py` | Metrics, and the rule "only a better model becomes a candidate" |
| `test_pipeline.py` | All five steps end to end: prep → train → evaluate → register → score |

**For a new use case:** change `make_raw_data()` in `conftest.py` so the fake
data has your columns, and add a test for any new cleaning you write in `prep.py`.
