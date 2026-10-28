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
    delete_recent_sales,
    fetch_sales,
    get_engine,
    get_schema,
    init_db,
    insert_sale,
    masked_url,
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
    st.markdown(
        """
This page talks to a relational database from Python using **SQLAlchemy**. The
recommended backend is **PostgreSQL** running inside a **Docker** container, which
keeps the database isolated and reproducible.

Here's the whole setup at a glance — the highlighted box is the backend you're
connected to **right now**:
"""
    )

    st.graphviz_chart(architecture_dot(backend), use_container_width=True)

    st.markdown(
        """
- Your **browser** shows the Streamlit UI.
- The **Python app** creates one SQLAlchemy *engine* and reuses it for every query.
- It first tries **PostgreSQL** on `localhost:5432`. If the container isn't running,
  it transparently falls back to a local **SQLite** file (`data/fallback.db`), so the
  lesson works with or without Docker.
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
        "Every relational table has a schema: named, typed columns. This one is "
        "reflected live from the database via SQLAlchemy's inspector."
    )
    st.dataframe(get_schema(engine), use_container_width=True, hide_index=True)

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

    st.divider()

    # ----- Live table -----------------------------------------------------
    st.subheader("All sales")
    st.dataframe(fetch_sales(engine), use_container_width=True)
    show_source(fetch_sales, "🐍 The function that reads the table")
    with st.expander("🔎 The SQL SQLAlchemy generates for the read"):
        st.code(str(select(sales).order_by(sales.c.id)), language="sql")

    st.divider()

    # ----- Aggregate view -------------------------------------------------
    st.subheader("Total sales by region")
    if st.toggle("Show chart"):
        df = total_sales_by_region(engine)
        if df.empty:
            st.info("No sales recorded yet — add or seed some rows above.")
        else:
            st.bar_chart(df, x="region", y="total_sales")
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

    EXAMPLES = {
        "Totals by region": (
            "SELECT region, COUNT(*) AS n, SUM(amount) AS total\n"
            "FROM sales\nGROUP BY region\nORDER BY total DESC;"
        ),
        "Ten biggest sales": (
            "SELECT product, region, amount\n"
            "FROM sales\nORDER BY amount DESC\nLIMIT 10;"
        ),
        "Everything": "SELECT * FROM sales;",
    }

    choice = st.selectbox("Load an example", list(EXAMPLES))
    query = st.text_area("SQL query", value=EXAMPLES[choice], height=140, key=choice)

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
            st.dataframe(result_df, use_container_width=True)
        else:
            st.success(message)

    show_source(run_query, "🐍 How the query runs (and how writes are guarded)")
