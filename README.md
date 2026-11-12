# Teaching App 🎓

A multi-page **Streamlit** app that unifies the teaching lessons from this workspace into
one interactive tool:

- **🗄️ Database Basics** — a hands-on relational-database lesson organized into three tabs:
  an **architecture diagram** of the setup (PostgreSQL-in-Docker ← Python/SQLAlchemy, with the
  active backend highlighted), an **Explore & Add Data** tab (schema viewer, seed / remove
  sample-data buttons, insert form, live table, and a `SUM(amount) GROUP BY region` chart — each
  paired with a collapsible box showing the *actual function that runs*), and a **SQL Playground** for running
  raw queries. Falls back to a local SQLite database automatically if the Postgres container isn't
  running.
- **⚡ Parallelization** — a "benchmark race" comparing **Serial vs Parallel vs Vectorized**
  row computations in Python, plus a side-by-side view of the equivalent Python and R code.
- **🎯 Recommender Systems** — content-based filtering (TF-IDF + cosine similarity),
  collaborative filtering (neighbourhood, a bias baseline, and from-scratch matrix
  factorization trained live), and market-basket **association rules** via Apriori.
- **🔮 Propensity Models** — look-alike modeling for churn: the observation → buffer →
  outcome time design, three model families (logistic regression, XGBoost, neural net),
  stratified cross-validation and grid/randomized search, the metrics imbalanced data
  actually needs (confusion matrix, precision/recall/F1, ROC-AUC, PR-AUC), learning and
  validation curves, and propensity scores cut into risk segments.
- **⏳ Survival Analysis** — the same business question in time: **censoring**, Kaplan-Meier
  survival curves, Cox / Random Survival Forest / Survival SVM, and the metrics censoring
  forces on us (c-index, time-dependent AUC, integrated Brier score), ending in individual
  survival curves and risk strata.
- **🧩 Customer Segmentation** — **RFM** scoring (the R/F/M cube and a business-rule ladder),
  **customer lifetime value** from a retention rate *measured* on cohorts (and how sharply the
  value moves with that assumption), and **clustering** on demographics and behaviour,
  evaluated without labels (silhouette, stability, a 2D projection).

## Quickstart

Requires [**uv**](https://docs.astral.sh/uv/) (and, optionally, Docker for PostgreSQL).
Common tasks are wrapped in a `Makefile` — run `make help` to list them:

```bash
make setup   # install the uv env (Python 3.13) + start PostgreSQL in Docker
make run     # launch the app at http://localhost:8501
make test    # run the test suite
```

Under the hood these are just uv commands — `make install` is `uv sync`, `make run` is
`uv run streamlit run Home.py`. Every repo in the workspace uses its own uv environment on
Python 3.13; see [ENVIRONMENTS.md](ENVIRONMENTS.md) for the full strategy.

Docker is optional: if you skip `make db-up` (or Docker isn't running), the app automatically
falls back to a local SQLite database, so `make install && make run` works on its own.

> Not using uv? A `requirements.txt` (exported from the lockfile) is also provided:
> `pip install -r requirements.txt && streamlit run Home.py`.

### Make targets

| Target | Description |
|--------|-------------|
| `make install` | Create/update the uv env (Python 3.13) from the lockfile |
| `make setup` | `install` + start the PostgreSQL container |
| `make run` | Launch the Streamlit app |
| `make test` | Run the test suite |
| `make db-up` / `make db-stop` | Start / stop PostgreSQL in Docker |
| `make lock` / `make export-reqs` | Refresh `uv.lock` / regenerate `requirements.txt` |
| `make clean` | Remove the virtualenv and Python caches |

## Deployment (Railway)

The repo ships a `Dockerfile` and `railway.json`; connect the GitHub repo in Railway and it
builds the image (Railway injects `$PORT`, health check on `/_stcore/health`).

The sibling lesson repos are **not** part of this repo. The container clones them **when it
starts**, on the branch matching the app's (`siblings.toml`):

| App branch | Sibling branch |
|---|---|
| `develop` | `develop` |
| `main` | `main` |
| anything else | `main` |

Siblings are maintained first, so a restart/redeploy simply runs their latest commit on that
branch — a sibling-only change needs no app commit, just a redeploy. The branch comes from
Railway's `$RAILWAY_GIT_BRANCH` (override with a `SIBLING_BRANCH` service variable). Public
datasets listed in `siblings.toml` are downloaded too (MovieLens from GroupLens); Kaggle-only
datasets are not — those lessons fall back to synthetic data and say so on screen. PostgreSQL
is not provisioned; the Database page uses its SQLite fallback.

```bash
make docker-build      # build exactly what Railway builds
make docker-run        # run it at http://localhost:8501 (siblings on develop)
make siblings-status   # which branch each local sibling is on vs. expected
make siblings          # clone any missing sibling on the matching branch (fresh clone)
```

## Project layout

```
Home.py                     # Landing page (cards linking to every lesson)
Makefile                    # Common tasks: make install / setup / run / test
pyproject.toml              # Dependencies (source of truth), Python >= 3.13
uv.lock                     # Pinned, reproducible resolution
.python-version             # Pins the interpreter to 3.13
requirements.txt            # Exported from the lock (pip/deploy fallback)
Dockerfile, railway.json    # Deployment image + Railway config
scripts/start.sh            # Container start: fetch siblings for this branch, then serve
siblings.toml               # Sibling repos, branch rule (develop↔develop, main↔main), public data
scripts/siblings.py         # fetch / data / status for the siblings (stdlib only)
ENVIRONMENTS.md             # Workspace-wide uv environment strategy
.streamlit/config.toml      # Theme
pages/
  1_Database_Basics.py      # DB lesson UI
  2_Parallelization.py      # Parallelization lesson UI
  3_Recommender.py          # Recommender Systems lesson UI
  4_Propensity.py           # Propensity / look-alike modeling lesson UI
  5_Survival.py             # Survival analysis lesson UI
  6_Segmentation.py         # Customer segmentation lesson UI (RFM, CLTV, clustering)
utils/
  data_generator.py         # @st.cache_data random dataset generator
  db_utils.py               # SQLAlchemy connection + CRUD/introspection/run_query (Postgres → SQLite fallback)
  compute_utils.py          # Serial / Parallel / Vectorized benchmark functions
  io_utils.py               # Sequential / threaded / async / multiprocess I/O benchmarks
  teaching.py               # UI-free teaching helpers: source_of, postgres_docker_command, architecture_dot
  recommender_utils.py      # TF-IDF, collaborative filtering, matrix factorization, Apriori
  propensity_utils.py       # Churn classification, tuning, metrics, risk segments (+ synthetic fallback)
  survival_utils.py         # Kaplan-Meier, Cox / RSF / Survival SVM, censoring-aware metrics
  segmentation_utils.py     # Thin adapter that imports segmentation/src/* (RFM, CLTV, clustering)
tests/                      # pytest + Streamlit AppTest suite
data/                       # Local SQLite fallback DB lives here (gitignored)
```

## Database

The Database Basics page connects to PostgreSQL using the parameters from the
`postgresql/` setup guide (`localhost:5432`, db `mydatabase`, user `myuser`). Start/stop the
container with the Makefile:

```bash
make db-up     # start PostgreSQL in Docker (creates the container on first run)
make db-stop   # stop it (data is preserved)
```

`make db-up` runs the equivalent `docker run --name postgres-db … -d postgres`. If Postgres
is unreachable, the app transparently uses a SQLite file at `data/fallback.db`, so the DB
lesson works with or without Docker.

The `docker run` command shown on the page is **read from `postgresql/README.md` at runtime**
(via `teaching.postgres_docker_command`) rather than duplicated, so the app and the setup guide
can never drift apart. The **SQL Playground** is read-only by default — only `SELECT`/`WITH`
run until you tick *Allow write/DDL statements*, so a stray `DROP` can't wipe the demo table.

Ids stay **contiguous**: every write realigns PostgreSQL's id sequence to `MAX(id)`
(`db_utils.resync_id_sequence`), so after deleting rows the next insert continues at
`MAX(id) + 1` instead of jumping. Existing ids are never mutated; the call is a no-op on SQLite,
which already reuses `MAX(id) + 1`.

## Notes

- The Database Basics page shows the **real executed source** of each helper via
  `inspect.getsource` (so displayed code can't drift from what runs), and draws the architecture
  with `st.graphviz_chart` fed a DOT string — **no Graphviz Python package required**.
- The Parallelization page **generates data on the fly** at the slider-selected size; it does
  not load the large `parallelization/data/parallel_big_data.csv`.
- The R script (`parallelization/parallel/parallel.R`) is **displayed only, never executed**.
- The Propensity and Survival pages fit their models on a **slider-selected sample** of the
  ~505,000-customer churn dataset, so a live demo never waits minutes for a hyperparameter
  search. If the sibling repo's git-ignored Kaggle CSVs are missing, both pages fall back to a
  **synthetic** customer base with the same columns and say so on screen.
- The Segmentation page **imports the sibling's own `segmentation/src/` modules** instead of
  re-implementing them, so the code it shows is literally the code the lesson repo runs. If the
  sibling's data files are missing, its loaders fall back to synthetic data and the page says so.
- Adapted from the existing `parallelization/`, `postgresql/`, `recommender/`, `predictions/`
  and `segmentation/` teaching repos — the app reads from them, it never copies them.
