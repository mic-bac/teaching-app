# CLAUDE.md — Teaching App

Project-local guide for making changes in the right context. Not committed (gitignored).

## What this is

A multi-page **Streamlit** app that turns a set of independent data-science repos into one
**interactive teaching tool**. The audience is **learners**: every page should not only *work*
but *teach* — it shows the real code that runs, the SQL/commands generated, and the runtime
architecture. Favor clarity and pedagogy over cleverness.

**Usage model: instructor-led demo.** The instructor runs the app and drives it live while
presenting; students **watch**, they do not operate it themselves. It is a single-trusted-user
demo, not a deployed/multi-user product. So **do not invest in defensive hardening** — no auth,
no input sanitization against misuse, no guarding against hostile/careless input, no
production-safety concerns. Spend that effort on **clarity and teaching value** instead.

This repo (`teaching-app/`) is the **orchestrator**. Each sibling folder is an independent
lesson repo that stands alone *and* is surfaced as a page here.

## Workspace architecture

```
teaching-app/            ← this repo (Streamlit orchestrator; git repo, branch `develop`)
├── Home.py, pages/, utils/, tests/
├── object-detection/    ┐
├── parallelization/     │  independent sibling repos — each its OWN git repo + uv env,
├── postgresql/          │  gitignored from this repo, consumed read-only by the app
├── predictions/         │
├── recommender/         │
└── segmentation/        ┘
```

- **Siblings are independent repos**, `.gitignore`d from the root — never `git add`/commit
  them here; edits to a sibling are committed in *that* repo. New teaching topic ⇒ new
  sibling repo added to the root `.gitignore`.
- **Siblings are the single source of truth.** The app *reads from* them, it does not copy
  them. Examples in code: `utils/teaching.postgres_docker_command()` extracts the `docker run`
  from `postgresql/README.md`; the Parallelization page displays the real
  `parallelization/parallel/parallel.py` and `.R`.

### Lessons

| Page | Source repo | Status |
|------|-------------|--------|
| `pages/1_Database_Basics.py` | `postgresql/` | ✅ built |
| `pages/2_Parallelization.py` | `parallelization/` | ✅ built |
| `pages/3_Recommender.py` | `recommender/` | ✅ built |
| `pages/4_Propensity.py` | `predictions/propensity.py` | ✅ built |
| `pages/5_Survival.py` | `predictions/survival.py` | ✅ built |
| `pages/6_Segmentation.py` | `segmentation/` (rfm, cltv, clustering) | ✅ built |
| object-detection | respective repo | 🔜 **will become its own `pages/N_*.py`** |

`predictions/timeseries.py` (analysing a series + classical models) and
`predictions/forecasting.py` (Prophet/XGBoost) are a third topic in that sibling repo and
are **not** yet surfaced — they belong to a different lecture block. Unlike the two churn
scripts, their logic lives in importable modules (`predictions/src/{sales_data,ts_core,
forecast_models,control_series}.py`), so a future page can **import** the sibling like
`segmentation_utils.py` does rather than re-implement it.

## Environments & tooling

- **uv everywhere, Python 3.13**, one env **per repo** (not shared — uv's cache hardlink-dedups
  so there's no real duplication; separate envs keep lessons independent & conflict-free).
- Per repo: `pyproject.toml` (deps, `requires-python = ">=3.13"`) + `uv.lock` + `.python-version`
  (=3.13), all committed; `.venv/` gitignored. Full policy: `ENVIRONMENTS.md`.
- **Common tasks via the `Makefile`** (`make help`): `make install` (`uv sync`), `make run`
  (`uv run streamlit run Home.py`), `make test` (`uv run pytest`), `make db-up`/`db-stop`
  (Postgres in Docker), `make lock`, `make export-reqs`, `make clean`.
- Change deps with `uv add`/`uv remove` (updates pyproject + lock); then `make export-reqs`
  to refresh the pip-fallback `requirements.txt`. Never hand-edit `requirements.txt`.
- **R is display-only** — never installed, never executed. `.R` scripts are shown for
  comparison. Install R separately only to run one standalone.

## App structure & core conventions

- **`st.*` lives only in `Home.py` and `pages/`.** All logic goes in **`utils/` as UI-free,
  importable, unit-testable functions.** `utils/teaching.py` is deliberately Streamlit-free.
- Imports are `from utils.x import ...` — the app always runs from the repo root.
- **First call in every page is `st.set_page_config(...)`**, `layout="wide"`.
- Existing utils: `data_generator.py` (`@st.cache_data`), `db_utils.py` (SQLAlchemy Core,
  Postgres→SQLite), `compute_utils.py` (serial/parallel/vectorized; `expensive_row_op` must
  stay **module-top-level** so `multiprocessing` can pickle it), `teaching.py` (`source_of`,
  `postgres_docker_command`, `architecture_dot`), `io_utils.py`, `recommender_utils.py`,
  `propensity_utils.py` (churn classification; its `load_churn` synthetic fallback is shared
  with), `survival_utils.py` (Kaplan-Meier, Cox/RSF/SVM, censoring-aware metrics),
  `segmentation_utils.py` (RFM, cohort-measured CLTV, clustering).

- **Two ways a util relates to its sibling.** Most *re-implement* the sibling's logic, because
  `predictions/{propensity,survival}.py` are `# %%` scripts with top-level side effects and
  cannot be imported (the newer `predictions/src/*` modules are pure, and should be imported).
  `segmentation_utils.py` instead **imports** `segmentation/src/*` directly — those modules are
  already pure and importable, so copying them would duplicate the source of truth and
  guarantee drift. Prefer importing whenever a sibling exposes a clean `src/`; re-implement
  only when the sibling is a side-effecting script. Either way the util stays Streamlit-free.

### Teaching patterns to reuse (this is what makes it a *teaching* app)

- **Show the real code.** Use `teaching.source_of(fn)` / the page-level `show_source(fn, label)`
  to render the *actual* function that runs, so displayed code can't drift from behavior.
- **Reveal the generated artifact**, not just the result: e.g. `st.code(str(select(sales)...))`
  to show the SQL SQLAlchemy emits; `st.graphviz_chart(architecture_dot(...))` for a live,
  state-aware diagram (raw DOT string — no graphviz Python dep).
- **Graceful degradation so lessons always work**: DB falls back Postgres→SQLite; benchmarks
  **generate data on the fly** at a slider size rather than loading big files. (The SQL
  playground's write-toggle is a *presenter convenience* so a stray `DROP` doesn't derail a
  live demo — not a security control. Don't add hardening beyond that.)
- **Explain the "why" in docstrings/captions** (see `_resync_id_sequence` for the model).
- Reuse the sibling repo's real content instead of duplicating it.

## Testing (required for every change)

- Run with `make test` / `uv run pytest`. Currently 84 tests, all must stay green.
- **`tests/test_utils.py`** — unit-test every `utils/` helper. Keep them **deterministic**:
  DB tests use a temp **SQLite** file (`create_engine(f"sqlite:///{tmp}")`), never Docker.
- **`tests/test_app_smoke.py`** — drive each page with `streamlit.testing.v1.AppTest` and
  assert `not at.exception`, plus a key interaction (button click / widget) per page.
- Add tests alongside any new helper or page in the same PR/change.

## How to add a new lesson page (primary workflow)

1. **Read the sibling repo** (`<topic>/`) — its real code + README are the source of truth.
2. **`utils/<topic>_utils.py`** — adapt the sibling's logic into UI-free, importable functions
   (return data, don't render). Keep anything used by `multiprocessing` at top level.
3. **`pages/N_<Topic>.py`** — `set_page_config` first; structure with `st.tabs`; use
   `show_source(...)` to reveal the real functions; display the sibling's real source via
   `pathlib` (read-only; never execute R); add graceful fallbacks and on-the-fly data.
4. **Tests** — unit tests for the utils + an `AppTest` smoke test asserting no exceptions.
5. **`Home.py`** — add an `st.container(border=True)` card + `st.page_link("pages/N_...")`.
6. **Deps** — `uv add <pkg>`, then `make export-reqs`; update `ENVIRONMENTS.md` status if the
   sibling env changes.

## Git & safety

- Root repo tracks **only the app**; branch is **`develop`**. Don't commit or push unless
  asked. Don't touch sibling repos' git.
- Don't load the 184 MB `parallelization/data/parallel_big_data.csv`; it's gitignored.
- Postgres creds (from `postgresql/`): `localhost:5432`, db `mydatabase`, user `myuser`,
  password `mypassword`, container `postgres-db`.
