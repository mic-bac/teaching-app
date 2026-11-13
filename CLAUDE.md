# CLAUDE.md — Teaching App

Project-local guide for making changes in the right context (tracked in git).

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
| `pages/7_Time_Series.py` | `predictions/timeseries.py` | ✅ built |
| `pages/8_Forecasting.py` | `predictions/forecasting.py` | ✅ built |
| `pages/9_Object_Detection.py` | `object-detection/results/` | ✅ built — **results only** (no torch/YOLO in the app) |

The time-series lessons import the sibling's pure modules (`predictions/src/{sales_data,
ts_core,forecast_models,forecast_panel,control_series}.py`) through `utils/forecasting_utils.py`, loaded as
a package with `teaching.load_sibling_package` (their modules import each other relatively,
and the name `src` would otherwise collide with `segmentation/src`).

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
- Width: use `width="stretch"` (not the removed `use_container_width`). Process pools go
  through `compute_utils.fork_pool` (explicit **fork**): Streamlit runs each page as `__main__`,
  and spawn/forkserver workers re-execute `__main__` — the whole page — so the app hangs.
- **First call in every page is `st.set_page_config(...)`**, `layout="wide"`.
- Existing utils: `data_generator.py` (`@st.cache_data`), `db_utils.py` (SQLAlchemy Core,
  Postgres→SQLite), `compute_utils.py` (serial/parallel/vectorized; `expensive_row_op` must
  stay **module-top-level** so `multiprocessing` can pickle it), `teaching.py` (`source_of`,
  `postgres_docker_command`, `architecture_dot`), `io_utils.py`, `recommender_utils.py`,
  `propensity_utils.py` (churn classification; its `load_churn` synthetic fallback is shared
  with), `survival_utils.py` (Kaplan-Meier, Cox/RSF/SVM, censoring-aware metrics),
  `segmentation_utils.py` (RFM, cohort-measured CLTV, clustering — imports `segmentation/src/*`
  incl. `cltv_core`), `forecasting_utils.py` (time series + forecasting — imports
  `predictions/src/*` as a package; adds the lesson's CV loop, Christmas check and the slides'
  Excel smoothing example), `detection_utils.py` (reads `object-detection/results/` — exported
  boxes, training metrics, photo credits — and redraws boxes with Pillow; no torch).

- **Object detection is results-only.** The sibling (AGPL-3.0, Ultralytics YOLO) exports with
  `export_results.py`; the app never imports `ultralytics`/torch. Only openly licensed photos
  (credited in `results/CREDITS.md`) are published — the original training photos and weights
  stay local and git-ignored.

- **Two ways a util relates to its sibling.** Most *re-implement* the sibling's logic, because
  `predictions/{propensity,survival}.py` are `# %%` scripts with top-level side effects and
  cannot be imported (the newer `predictions/src/*` modules are pure, and should be imported).
  `segmentation_utils.py` instead **imports** `segmentation/src/*` directly — those modules are
  already pure and importable, so copying them would duplicate the source of truth and
  guarantee drift (the Recommender likewise imports `recommender/src/groceries_data.py`, via
  `teaching.load_sibling_module`). Prefer importing whenever a sibling exposes a clean `src/`; re-implement
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
- **Write for business students, not mathematicians.** Plain language first, short sentences, no
  unexplained jargon; formulas from the slides go into a collapsed *"🔎 The maths, if you're
  curious"* expander. Every lesson page opens with a **💼 Where you'll use this** box (2–3 real job
  situations) and ends with **🎯 What to remember** (three bullets). The goal is curiosity and
  confidence — students should leave wanting to use these tools at work.
- Reuse the sibling repo's real content instead of duplicating it.

## Testing (required for every change)

- Run with `make test` / `uv run pytest`. All tests must stay green (203 as of 2026-11-13).
- **`tests/test_sibling_alignment.py`** — drift guard: sibling hyperparameters are regex-read and
  compared with the app. Parameter differences for live-demo runtime are fine but must be
  recorded there as `"intentional"`; logic differences are not.
- **Sibling first.** Lesson-content changes start in the sibling repo, then the app follows.
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

## Deployment & the sibling branch rule

- **Siblings are maintained first, and branches track branches:** app `develop` runs against
  every sibling's `develop`, app `main` against every sibling's `main` (single-branch siblings
  like `postgresql` stay on `main`). Merge sibling `develop → main` before or together with
  the app's `develop → main`. Defined in `siblings.toml`; enforced by `tests/test_deployment.py`.
- Locally, keep each sibling checked out on the branch matching the app's — check with
  `make siblings-status`.
- Railway builds the `Dockerfile`; `scripts/start.sh` clones the siblings **at container
  start** on the matching branch (`SIBLING_BRANCH`, else `$RAILWAY_GIT_BRANCH`, else `main`), plus
  public data. A sibling-only change therefore needs just a redeploy, no app commit.
- **CI/CD** (`.github/workflows/ci-cd.yml`): every push to `develop`/`main` (no separate PR
  trigger, to avoid testing a commit twice) runs the tests against freshly cloned siblings of the
  same branch, then builds and
  health-checks the Docker image; only a push to `main` deploys, via `railway up` (secret
  `RAILWAY_TOKEN`, variable `RAILWAY_SERVICE`, GitHub environment `production`). Railway's own
  GitHub auto-deploy must stay off. CI has no Kaggle data, so the synthetic fallbacks are tested too.
- On Railway, Kaggle datasets are absent → every page needs a synthetic fallback. A new
  sibling the app reads from must be added to `siblings.toml`.
- Importing a sibling module: use `utils.teaching.load_sibling_module(repo, "src/x.py")` —
  several siblings name their package `src`, so plain `from src import` would collide.

## Git & safety

- Root repo tracks **only the app**; branch is **`develop`**. Don't commit or push unless
  asked. Don't touch sibling repos' git.
- Don't load the 184 MB `parallelization/data/parallel_big_data.csv`; it's gitignored.
- Postgres creds (from `postgresql/`): `localhost:5432`, db `mydatabase`, user `myuser`,
  password `mypassword`, container `postgres-db`.
