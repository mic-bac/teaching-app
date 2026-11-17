"""Database Basics page.

A hands-on relational-database lesson backed by PostgreSQL (in Docker) with an
automatic SQLite fallback when Postgres is not reachable. Beyond the working demo,
the page shows learners the *actual Python functions* that run behind each action,
sketches the runtime architecture, and offers a live SQL playground.
"""

import streamlit as st
from sqlalchemy import insert, select

from utils.db_utils import (
    count_sales,
    EXAMPLE_QUERIES,
    delete_recent_sales,
    explain_query,
    explain_schema,
    fetch_sales,
    get_engine,
    get_schema,
    init_db,
    insert_sale,
    masked_url,
    region_reading,
    run_query,
    sales,
    seed_sample_sales,
    total_sales_by_region,
)
from utils.teaching import architecture_dot, postgres_docker_command, source_of

st.set_page_config(page_title="Database Basics", layout="wide", page_icon="🗄️")


def show_source(obj, label: str) -> None:
    """Render an expander containing the real source code of ``obj``.

    The code shown is read from the running module, so it can never drift out of
    sync with what actually executes.
    """
    with st.expander(label):
        st.code(source_of(obj), language="python")


st.title("🗄️ Database Basics")
with st.container(border=True):
    st.markdown(
        "💼 **Where you'll use this** — almost everywhere: sales, customers and orders live in databases. "
        "A few lines of SQL get you the numbers yourself — *revenue by region* for the Monday meeting, "
        "*which orders above €250 came in this week* for a sales manager, *how many customers bought twice* "
        "for marketing — instead of waiting days for someone else's report."
    )

# ---------------------------------------------------------------------------
# Connect once (with graceful fallback) and make sure the schema exists.
# ---------------------------------------------------------------------------
engine, backend = get_engine()
init_db(engine)

if backend == "PostgreSQL":
    st.success("Connected to PostgreSQL 🐘")
else:
    st.warning("PostgreSQL not reachable — using local SQLite fallback.")

tab_setup, tab_data, tab_sql = st.tabs(
    ["🧭 Setup & Architecture", "📊 Explore & Add Data", "🧪 SQL Playground"]
)

# ===========================================================================
# TAB 1 — Setup & Architecture
# ===========================================================================
with tab_setup:
    with st.container(border=True):
        st.markdown(
            """
**Why a database, and not one big Excel file?**

| | Excel file | Database |
|---|---|---|
| Who uses it | one person at a time, copies by e-mail | hundreds of people and programs **at once**, one shared version |
| Size | slows down around a million rows | billions of rows |
| Rules | any cell can hold anything | each column has a fixed type — `amount` *must* be a number |
| Asking questions | filters and pivot tables, by hand | **SQL**: written once, re-run every Monday |
"""
        )

    st.markdown(
        """
This page talks to a database from Python. The database is **PostgreSQL**, a free
database used by many companies, running inside **Docker** — think of Docker as a
sealed box with the database already installed, so it runs the same on every laptop.

Here's the whole setup at a glance — the highlighted box is the backend you're
connected to **right now**:
"""
    )

    st.graphviz_chart(architecture_dot(backend), width="stretch")

    st.markdown(
        """
How to read it, left to right — like a shop:
- **Your browser** is the shop window: it shows the page and passes on your clicks.
- **The Python app** is the clerk: it turns a click (*"show all sales"*) into SQL,
  using a library called **SQLAlchemy**, and opens one connection it reuses for every question.
- **PostgreSQL** is the warehouse where the rows are kept. If its box isn't running,
  the clerk uses a small backup warehouse instead — **SQLite**, a database that is
  just one file (`data/fallback.db`). The highlighted box shows which one is in use;
  the lesson works either way.
"""
    )

    st.subheader("Start PostgreSQL in Docker")
    st.markdown(
        "The command below comes straight from the "
        "[`postgresql/` setup guide](../postgresql/README.md) — Docker will download the "
        "`postgres` image if needed, expose port `5432`, and persist data in a named volume."
    )
    st.code(postgres_docker_command(), language="bash")

    st.subheader("Active connection")
    col_a, col_b = st.columns(2)
    col_a.metric("Backend", backend)
    col_b.metric("Rows in `sales`", count_sales(engine))
    st.caption("Connection URL (password hidden):")
    st.code(masked_url(engine), language="text")

    show_source(get_engine, "🐍 How the app connects (with fallback)")


# ===========================================================================
# TAB 2 — Explore & Add Data
# ===========================================================================
with tab_data:
    # ----- Schema ---------------------------------------------------------
    st.subheader("The `sales` table")
    st.caption(
        "Every table has a fixed list of named columns, each with a type — this list is "
        "read live from the database. The type is a promise: the database refuses a "
        "sale whose `amount` is the word *\"ten\"*, so every total you calculate later can be trusted."
    )
    st.dataframe(explain_schema(get_schema(engine)), width="stretch", hide_index=True)

    # ----- Seed / remove sample data -------------------------------------
    st.subheader("Need some data?")
    st.caption(
        "Seeding **persists** rows to the database — use *Remove newest* to undo, "
        "deleting the most recently added rows. Ids stay **contiguous**: the sequence "
        "is realigned on every write, so the next insert continues at `MAX(id) + 1` "
        "instead of jumping after deletes."
    )
    seed_col1, seed_col2, seed_col3 = st.columns([1, 1, 2])
    n_seed = seed_col1.number_input(
        "Number of rows", min_value=1, max_value=500, value=20, step=5
    )
    if seed_col2.button("🌱 Seed random sales"):
        added = seed_sample_sales(engine, int(n_seed))
        st.toast(f"Inserted {added} random sales.")
        st.rerun()
    if seed_col3.button("🗑️ Remove newest rows"):
        removed = delete_recent_sales(engine, int(n_seed))
        st.toast(f"Removed {removed} row(s).")
        st.rerun()
    show_source(seed_sample_sales, "🐍 The function that bulk-inserts fake rows")
    show_source(delete_recent_sales, "🐍 The function that removes the newest rows")

    st.divider()

    # ----- Add a sale -----------------------------------------------------
    st.subheader("Add a sale")
    with st.form("add_sale"):
        product = st.text_input("Product", placeholder="e.g. Widget")
        region = st.selectbox("Region", ["North", "South", "East", "West"])
        quantity = st.number_input("Quantity", min_value=1, step=1, value=1)
        amount = st.number_input(
            "Amount", min_value=0.0, step=0.01, value=0.0, format="%.2f"
        )
        submitted = st.form_submit_button("Add sale")

        if submitted:
            if not product.strip():
                st.error("Please enter a product name.")
            else:
                insert_sale(engine, product.strip(), region, int(quantity), float(amount))
                st.rerun()

    show_source(insert_sale, "🐍 The function that writes a row")
    with st.expander("🔎 The SQL SQLAlchemy generates for an insert"):
        st.code(str(insert(sales)), language="sql")
        st.caption(
            "Read it as: *put a new row into `sales` with these columns.* The `:product`, "
            "`:amount`, … are empty slots — the values you typed in the form are sent "
            "separately and dropped into them. You never write this line yourself; "
            "the Python function above builds it."
        )

    st.divider()

    # ----- Live table -----------------------------------------------------
    st.subheader("All sales")
    st.dataframe(fetch_sales(engine), width="stretch")
    show_source(fetch_sales, "🐍 The function that reads the table")
    with st.expander("🔎 The SQL SQLAlchemy generates for the read"):
        st.code(str(select(sales).order_by(sales.c.id)), language="sql")
        st.caption(
            "Read it as: *take every column of every row in `sales`, sorted by `id`* — "
            "the same as opening a sheet and sorting it by its first column."
        )

    st.divider()

    # ----- Aggregate view -------------------------------------------------
    st.subheader("Total sales by region")
    if st.toggle("Show chart"):
        df = total_sales_by_region(engine)
        if df.empty:
            st.info("No sales recorded yet — add or seed some rows above.")
        else:
            st.bar_chart(df, x="region", y="total_sales")
            st.markdown(region_reading(df))
            st.caption(
                "The seeded sales are random, so the winner changes when you seed again — "
                "the question to ask stays the same: *more sales, or bigger ones?*"
            )
        show_source(
            total_sales_by_region, "🐍 The function that aggregates, then plots"
        )


# ===========================================================================
# TAB 3 — SQL Playground
# ===========================================================================
with tab_sql:
    st.markdown(
        "Write SQL and run it against the live database. This is how data is "
        "retrieved (and changed) without any Python in the middle."
    )

    choice = st.selectbox("Load an example", list(EXAMPLE_QUERIES))
    query = st.text_area(
        "SQL query", value=EXAMPLE_QUERIES[choice]["sql"], height=140, key=choice
    )
    st.markdown("**The example, line by line** — every SQL step has an Excel twin:")
    st.dataframe(explain_query(choice), width="stretch", hide_index=True)
    st.caption(
        "SQL runs in a fixed order that differs from how it's written: first **FROM** "
        "(which table), then **WHERE** (filter rows), **GROUP BY** (bundle them), "
        "**SELECT** (pick columns), **ORDER BY** (sort) and **LIMIT** (cut off). "
        "Try it: change `250` to `400` in *Big sales in the North* and run again."
    )

    allow_writes = st.checkbox(
        "Allow write/DDL statements (INSERT/UPDATE/DELETE/DROP)",
        value=False,
        help="Off by default so a stray DROP can't wipe the demo table.",
    )

    if st.button("▶️ Run query"):
        result_df, message = run_query(engine, query, allow_writes=allow_writes)
        if message.startswith("Error"):
            st.error(message)
        elif result_df is not None:
            st.success(message)
            st.dataframe(result_df, width="stretch")
        else:
            st.success(message)

    show_source(run_query, "🐍 How the query runs (and how writes are guarded)")

    with st.container(border=True):
        st.markdown(
            """
🎯 **What to remember**
- **A database is a set of shared tables with fixed column types** — SQL is the language for asking them questions.
- **SQL steps are Excel steps written down:** `WHERE` = filter, `GROUP BY` = pivot table, `ORDER BY` = sort — and a written query can be re-run every Monday.
- **A total is half an answer:** ask whether it comes from *more* sales or *bigger* ones (`COUNT` and `SUM` together tell you).
"""
        )
