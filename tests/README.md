# Tests

Automated tests for the Streamlit teaching app.

## What is covered

- **`test_app_smoke.py`** — runs **every** page through Streamlit's headless
  `AppTest` framework and asserts **no exceptions** are raised (`Home.py` plus
  `pages/1_Database_Basics.py` … `pages/6_Segmentation.py`). On top of the load
  check, one or two interactions per page are driven at their cheapest setting:
  the landing page renders, the DB page shows a backend banner + sales table, the
  Excel-vs-database box and the line-by-line SQL explanation, reads the region chart, and
  the SQL playground runs a query, the CPU and I/O benchmarks run on button
  click (the CPU result is scaled to the 1-million-row file, and both tabs show their everyday
  analogies), the recommender returns recommendations and trains its matrix
  factorization, the propensity page fits a model / cross-validates / tunes /
  draws its learning and validation curves, and the survival page reports
  Kaplan-Meier readouts, trains all three models, scores them over time, and
  states that the ranking-only SVM has no survival curve, and the segmentation
  page reacts to its retention-rate slider without reloading the transaction log. The time-series page reproduces the lecture's Excel smoothing example and fits SARIMA on click.
- **`test_sibling_alignment.py`** — a drift guard for the utils that *re-implement* a
  sibling script (propensity, survival, recommender — and the inline steps of the time-series
  and forecasting scripts: CV folds, the band's coverage, the depth sweep, the CO₂ horizon).
  Each hyperparameter is regex-read from
  the sibling's source and compared with what the app runs. Entries are either `"same"` (the
  app must match) or `"intentional"` (a recorded live-demo deviation); either way the
  sibling's value is pinned, so a sibling edit fails here instead of drifting silently. It
  also checks that the segmentation action table only names segments the sibling's RFM ladder
  defines. Skips when a sibling repo is not checked out.
- **`test_deployment.py`** — every sibling the app reads from is listed in `siblings.toml`
  (https URL, git- and docker-ignored), and the branch rule resolves correctly: app
  `develop` → sibling `develop`, `main` → `main`, environment variables first.
- **`test_utils.py`** — unit tests for the shared utilities:
  - New output-quality guards: user-based CF needs enough neighbours (no 5.0 ties from a
    single rating), basket suggestions ignore bystander and lift ≤ 1 rules, the synthetic
    Groceries baskets recover a planted mission while staples stay at lift ≈ 1, the clustering
    verdict reads stability *and* separation, and the learning-curve reading compares against
    the best model (high bias) and recognises a flat validation curve.
  - `segmentation_utils` — the CLTV multiplier against the lecture formula and
    its convexity in the retention rate, frequency counted as invoices rather
    than line items, the tenure floor that keeps a one-order customer's
    annualised contribution finite, cohort frames that keep unobserved
    periods missing instead of silently reading them as zero retention, the
    year-by-year CLTV table summing to the closed form, per-band retention
    mapping, cluster-size readings (balanced vs extreme) and the campaign
    break-even maths.
  - `forecasting_utils` — the slides' Excel smoothing example reproduced exactly (MAE 10.20,
    MSE 184.89, RMSE 13.60 at α = 0.4), rolling-origin folds that only move forward, every
    Christmas lying before the first week a 52-week model can be tested on, the promotion
    break, the SARIMA label derived from the sibling, the growth baseline (last year × growth),
    Easter weeks ending on Good Friday, misses labelled with the moved holiday, the CV band
    and the plain-language readings, and `predictions/src` coexisting with
    `segmentation/src` in one process.
  - forecasting: the feature table drops the first year per store (`lag_52`), the vectorised
    per-store trend in `fit_hybrid` keeps stores apart, and one-Holt-Winters-per-store reports
    accuracy and cost; `with_easter` adds next year's Easter too, and the Prophet component
    reading finds the peaks. The Forecasting page smoke test presses all four experiment
    buttons and checks the Easter calendar test and the worked readings.
  - `detection_utils` — on a tiny fake `results/` export (no sibling, no torch): flattening
    the stored boxes, the confidence threshold (stricter never keeps more boxes), boxes really
    drawn on the photo, the YOLO label format, IoU from 0 to 1, the first epoch reaching a
    mAP level, the before/after table, credit parsing, the notebook's code-line count, and the
    plain-number readings (precision/recall as counts, the threshold trade-off, before vs after). The
    Object Detection smoke test moves the threshold slider (skipped if the sibling results are
    absent).
  - The plain-language readings shown under results: `benchmark_reading` / `io_reading` (both
    Parallelization races, incl. multiprocessing explained as overlapped waiting, and the CPU
    race scaled to the real 1-million-row file), `region_reading` (more sales or bigger ones?),
    `explain_schema` (column types in plain words), the example SQL queries (each runs, and every
    explained clause appears in its query) and
    `read_rule` (the top association rule as a sentence).
  - From the holistic review: plain names instead of code names (`plain_feature`, `plain_attribute`,
    the censoring-table headings), customers / clusters / folds counted from 1, `big_number`
    (MSE as "1.3 trillion", not `1.26e+12`), whole numbers in the survival customer table, and the
    new readings — content-based ties and taste-profile blends, the item-based list, the
    correlation chart, the churn climb across risk segments (incl. "Medium ≈ High"), when the
    contract-length gap opens, neighbouring risk strata that overlap, which k each clustering
    score prefers (inertia has none), the silhouette in words, cluster nicknames, and the labels a
    pretrained detector saw on one photo. Also: the tuning results as a plain table with a "did
    tuning matter?" reading, and the integrated Brier score read against the 0.25 guessing line.
  - `test_sibling_alignment` also checks that `object-detection/results/` was exported with the
    notebook's current EPOCHS / BATCH / IMG_SIZE (skipped if the sibling is absent).
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
