# Tests

Automated tests for the Streamlit teaching app.

## What is covered

- **`test_app_smoke.py`** — runs **every** page through Streamlit's headless
  `AppTest` framework and asserts **no exceptions** are raised (`Home.py` plus
  `pages/1_Database_Basics.py` … `pages/6_Segmentation.py`). On top of the load
  check, one or two interactions per page are driven at their cheapest setting:
  the landing page renders, the DB page shows a backend banner + sales table and
  the SQL playground runs a query, the CPU and I/O benchmarks run on button
  click, the recommender returns recommendations and trains its matrix
  factorization, the propensity page fits a model / cross-validates / tunes /
  draws its learning and validation curves, and the survival page reports
  Kaplan-Meier readouts, trains all three models, scores them over time, and
  states that the ranking-only SVM has no survival curve, and the segmentation
  page reacts to its retention-rate slider without reloading the transaction log.
- **`test_utils.py`** — unit tests for the shared utilities:
  - `segmentation_utils` — the CLTV multiplier against the lecture formula and
    its convexity in the retention rate, frequency counted as invoices rather
    than line items, the tenure floor that keeps a one-order customer's
    annualised contribution finite, and cohort frames that keep unobserved
    periods missing instead of silently reading them as zero retention.
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
  - `recommender_utils` — content-based, neighbourhood and matrix-factorization
    recommenders plus association rules, on tiny hand-built fixtures with a known
    right answer.
  - `propensity_utils` — the synthetic-data fallback (used when the sibling's
    git-ignored churn CSVs are absent), feature preparation, the stratified
    split, metrics, cross-validation, the tuning cost report, learning/validation
    curves, feature importance across model families, and risk segments.
  - `survival_utils` — the `(event, time)` structured target, Kaplan-Meier
    (monotone, inside its confidence band), per-segment curves, the censoring
    illustration, c-index / time-dependent AUC / integrated Brier score, and the
    honest "this model has no S(t)" path for the ranking-only Survival SVM.

  All of these run on small seeded fixtures, so they are deterministic and fast —
  no Kaggle download and no Docker required.

## Run

From the workspace root:

```bash
uv sync            # installs the dev group (pytest) too
uv run pytest -q
```

The DB smoke test uses PostgreSQL if the Docker container is running, otherwise
the app's built-in SQLite fallback — both are expected to load cleanly.
