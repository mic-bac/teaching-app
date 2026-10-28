# Environments

This workspace is a Streamlit orchestrator app (this repo) plus several **independent**
sibling teaching repos. **Every repo manages its own Python environment with
[uv](https://docs.astral.sh/uv/), standardized on Python 3.13.**

## Why per-repo (not one shared env)

- **Independence** — each sibling is a standalone repo ("microservice") that must run on its
  own and stay independently reproducible.
- **No real duplication** — uv stores packages once in a global cache (`~/.cache/uv`) and
  **hardlinks** them into each `.venv`, so separate environments share bytes on disk. A shared
  env would save little space.
- **Isolation** — separate resolutions avoid cross-lesson dependency conflicts
  (e.g. `streamlit` vs `torch` vs `scikit-surprise`).

## Standard files per repo

| File | Role | Git |
|------|------|-----|
| `pyproject.toml` | dependencies + `requires-python = ">=3.13"` (source of truth) | commit |
| `uv.lock` | fully pinned resolution (reproducible installs) | commit |
| `.python-version` | pins the interpreter to `3.13` | commit |
| `.venv/` | the actual environment | **ignore** |

## Commands (run inside a repo)

```bash
uv sync                        # create/update .venv from the lockfile
uv run <cmd>                   # run in the env, e.g. `uv run pytest`
uv run streamlit run Home.py   # (app only) launch the app
uv add <pkg> / uv remove <pkg> # change deps (updates pyproject.toml + uv.lock)
```

## Git: ignore vs commit

- **Ignore:** `.venv/`, `__pycache__/`, `*.py[cod]`
- **Commit:** `pyproject.toml`, `uv.lock`, `.python-version`

## R note

`parallelization/` includes `parallel/parallel.R`, shown **side by side** with the Python
version in the app for comparison but **never executed**. R is therefore not part of any uv
environment; install R separately (system/apt) only if you want to run a `.R` script
standalone.

## Status

| Repo | Python | Env state |
|------|--------|-----------|
| app (this repo) | 3.13 | synced ✅ (97 tests pass) |
| `segmentation` | 3.13 | synced ✅ (`openpyxl` for the one-time `fetch_data.py`; surfaced as the Segmentation page, whose `src/` modules the app imports directly) |
| `parallelization` | 3.13 | synced ✅ (R excluded) |
| `object-detection` | 3.13 | locked (run `uv sync` — large torch install) |
| `predictions` | 3.13 | synced ✅ (`scikit-survival`/`xgboost` ship wheels; `prophet` builds — only needed for `timeseries.py`/`forecasting.py`; surfaced as the Propensity + Survival pages) |
| `recommender` | 3.13 | synced ✅ (pure-Python: pandas/scikit-learn/mlxtend/plotly; surfaced as the Recommender page) |
