# Teaching App 🎓

A multi-page **Streamlit** app that unifies two teaching lessons from this workspace into
one interactive tool:

- **🗄️ Database Basics** — running PostgreSQL in Docker, with a form to insert fake sales,
  a live data table, and a `SUM(amount) GROUP BY region` bar chart. Falls back to a local
  SQLite database automatically if the Postgres container isn't running.
- **⚡ Parallelization** — a "benchmark race" comparing **Serial vs Parallel vs Vectorized**
  row computations in Python, plus a side-by-side view of the equivalent Python and R code.

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

## Project layout

```
Home.py                     # Landing page (links to the two lessons)
Makefile                    # Common tasks: make install / setup / run / test
pyproject.toml              # Dependencies (source of truth), Python >= 3.13
uv.lock                     # Pinned, reproducible resolution
.python-version             # Pins the interpreter to 3.13
requirements.txt            # Exported from the lock (pip/deploy fallback)
ENVIRONMENTS.md             # Workspace-wide uv environment strategy
.streamlit/config.toml      # Theme
pages/
  1_Database_Basics.py      # DB lesson UI
  2_Parallelization.py      # Parallelization lesson UI
utils/
  data_generator.py         # @st.cache_data random dataset generator
  db_utils.py               # SQLAlchemy connection + CRUD (Postgres → SQLite fallback)
  compute_utils.py          # Serial / Parallel / Vectorized benchmark functions
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

## Notes

- The Parallelization page **generates data on the fly** at the slider-selected size; it does
  not load the large `parallelization/data/parallel_big_data.csv`.
- The R script (`parallelization/parallel/parallel.R`) is **displayed only, never executed**.
- Adapted from the existing `parallelization/` and `postgresql/` teaching repos.
