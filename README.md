# Teaching App 🎓

A multi-page **Streamlit** app that unifies two teaching lessons from this workspace into
one interactive tool:

- **🗄️ Database Basics** — running PostgreSQL in Docker, with a form to insert fake sales,
  a live data table, and a `SUM(amount) GROUP BY region` bar chart. Falls back to a local
  SQLite database automatically if the Postgres container isn't running.
- **⚡ Parallelization** — a "benchmark race" comparing **Serial vs Parallel vs Vectorized**
  row computations in Python, plus a side-by-side view of the equivalent Python and R code.

## Run it

This app uses [**uv**](https://docs.astral.sh/uv/) on **Python 3.13** for its environment.
Every repo in the workspace has its own uv environment on the same Python version — see
[ENVIRONMENTS.md](ENVIRONMENTS.md) for the full strategy.

```bash
uv sync                       # create the locked environment (.venv)
uv run streamlit run Home.py  # launch the app
```

Then open http://localhost:8501.

Run the tests with `uv run pytest`.

> Not using uv? A `requirements.txt` (exported from the lockfile) is also provided:
> `pip install -r requirements.txt && streamlit run Home.py`.

## Project layout

```
Home.py                     # Landing page (links to the two lessons)
requirements.txt            # streamlit, pandas, numpy, SQLAlchemy, psycopg2-binary
.streamlit/config.toml      # Theme
pages/
  1_Database_Basics.py      # DB lesson UI
  2_Parallelization.py      # Parallelization lesson UI
utils/
  data_generator.py         # @st.cache_data random dataset generator
  db_utils.py               # SQLAlchemy connection + CRUD (Postgres → SQLite fallback)
  compute_utils.py          # Serial / Parallel / Vectorized benchmark functions
data/                       # Local SQLite fallback DB lives here (gitignored)
```

## Database

The Database Basics page connects to PostgreSQL using the parameters from the
`postgresql/` setup guide (`localhost:5432`, db `mydatabase`, user `myuser`). To start the
container:

```bash
docker run --name postgres-db \
  -e POSTGRES_PASSWORD=mypassword \
  -e POSTGRES_USER=myuser \
  -e POSTGRES_DB=mydatabase \
  -p 5432:5432 \
  -v postgres-data:/var/lib/postgresql/data \
  -d postgres
```

If Postgres is unreachable, the app transparently uses a SQLite file at `data/fallback.db`.

## Notes

- The Parallelization page **generates data on the fly** at the slider-selected size; it does
  not load the large `parallelization/data/parallel_big_data.csv`.
- The R script (`parallelization/parallel/parallel.R`) is **displayed only, never executed**.
- Adapted from the existing `parallelization/` and `postgresql/` teaching repos.
