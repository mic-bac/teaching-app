"""Database Basics page.

Demonstrates a simple sales database backed by PostgreSQL (in Docker) with an
automatic SQLite fallback when Postgres is not reachable.
"""

import streamlit as st

from utils.db_utils import (
    fetch_sales,
    get_engine,
    init_db,
    insert_sale,
    total_sales_by_region,
)

st.set_page_config(page_title="Database Basics", layout="wide", page_icon="🗄️")

DOCKER_RUN_CMD = """docker run --name postgres-db \\
  -e POSTGRES_PASSWORD=mypassword \\
  -e POSTGRES_USER=myuser \\
  -e POSTGRES_DB=mydatabase \\
  -p 5432:5432 \\
  -v postgres-data:/var/lib/postgresql/data \\
  -d postgres"""

st.title("🗄️ Database Basics")

st.markdown(
    """
This page shows how to work with a relational database from Python using
**SQLAlchemy**. The recommended backend is **PostgreSQL** running inside a
**Docker** container, which keeps the database isolated and reproducible.

Start the PostgreSQL container with the command below. Docker will download the
`postgres` image automatically if it is not present, expose the database on port
`5432`, and persist data in a named volume so it survives container restarts.
"""
)

st.code(DOCKER_RUN_CMD, language="bash")

# ---------------------------------------------------------------------------
# Connect (with graceful fallback) and make sure the schema exists.
# ---------------------------------------------------------------------------
engine, backend = get_engine()
init_db(engine)

if backend == "PostgreSQL":
    st.success("Connected to PostgreSQL 🐘")
else:
    st.warning("PostgreSQL not reachable — using local SQLite fallback.")

st.caption(f"Active backend: **{backend}**")

# ---------------------------------------------------------------------------
# Add a new sale.
# ---------------------------------------------------------------------------
st.subheader("Add a sale")

with st.form("add_sale"):
    product = st.text_input("Product", placeholder="e.g. Widget")
    region = st.selectbox("Region", ["North", "South", "East", "West"])
    quantity = st.number_input("Quantity", min_value=1, step=1, value=1)
    amount = st.number_input("Amount", min_value=0.0, step=0.01, value=0.0, format="%.2f")
    submitted = st.form_submit_button("Add sale")

    if submitted:
        if not product.strip():
            st.error("Please enter a product name.")
        else:
            insert_sale(engine, product.strip(), region, int(quantity), float(amount))
            st.rerun()

# ---------------------------------------------------------------------------
# Live table.
# ---------------------------------------------------------------------------
st.subheader("Sales table")
st.dataframe(fetch_sales(engine), use_container_width=True)

# ---------------------------------------------------------------------------
# Aggregate view.
# ---------------------------------------------------------------------------
if st.toggle("Show Total Sales by Region"):
    df = total_sales_by_region(engine)
    if df.empty:
        st.info("No sales recorded yet — add a sale above to see the chart.")
    else:
        st.bar_chart(df, x="region", y="total_sales")
