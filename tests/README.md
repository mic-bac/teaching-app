# Tests

Automated tests for the Streamlit teaching app.

## What is covered

- **`test_app_smoke.py`** — runs each page through Streamlit's headless
  `AppTest` framework and asserts **no exceptions** are raised (`Home.py`,
  `pages/1_Database_Basics.py`, `pages/2_Parallelization.py`). Also verifies the
  landing page renders, the DB page shows a backend banner + sales table, the SQL
  playground runs a query without error, and the benchmark runs on button click.
- **`test_utils.py`** — unit tests for the shared utilities:
  - `data_generator.generate_dataset` — shape, column names, value range.
  - `compute_utils` — serial/parallel/vectorized results agree, and
    `run_benchmark` returns the expected keys/types.
  - `db_utils` — full CRUD (init → insert → fetch → aggregate) plus the teaching
    helpers (`get_schema`, `count_sales`, `seed_sample_sales`, `delete_recent_sales`,
    `resync_id_sequence` / contiguous-id behaviour, `masked_url`, and `run_query`'s
    read/write guard + error reporting), against a temporary SQLite database, so it
    runs anywhere without Docker/PostgreSQL.
  - `teaching` — `source_of`, `postgres_docker_command` (reads the sibling
    `postgresql/README.md`), and `architecture_dot`.

## Run

From the workspace root:

```bash
uv sync            # installs the dev group (pytest) too
uv run pytest -q
```

The DB smoke test uses PostgreSQL if the Docker container is running, otherwise
the app's built-in SQLite fallback — both are expected to load cleanly.
