# Tests

Automated tests for the Streamlit teaching app.

## What is covered

- **`test_app_smoke.py`** — runs each page through Streamlit's headless
  `AppTest` framework and asserts **no exceptions** are raised (`Home.py`,
  `pages/1_Database_Basics.py`, `pages/2_Parallelization.py`). Also verifies the
  landing page renders, the DB page shows a backend banner + sales table, and the
  benchmark actually runs on button click.
- **`test_utils.py`** — unit tests for the shared utilities:
  - `data_generator.generate_dataset` — shape, column names, value range.
  - `compute_utils` — serial/parallel/vectorized results agree, and
    `run_benchmark` returns the expected keys/types.
  - `db_utils` — full CRUD (init → insert → fetch → aggregate) against a
    temporary SQLite database, so it runs anywhere without Docker/PostgreSQL.

## Run

From the workspace root:

```bash
uv sync            # installs the dev group (pytest) too
uv run pytest -q
```

The DB smoke test uses PostgreSQL if the Docker container is running, otherwise
the app's built-in SQLite fallback — both are expected to load cleanly.
