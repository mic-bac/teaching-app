"""Database utilities for the teaching app.

Provides a small, backend-agnostic data layer built on SQLAlchemy Core so the
same code works against PostgreSQL (running in Docker) and a local SQLite
fallback. The engine is created lazily via :func:`get_engine`; nothing connects
to a database at import time.
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    func,
    insert,
    select,
)

# ---------------------------------------------------------------------------
# Schema definition (portable across PostgreSQL and SQLite)
# ---------------------------------------------------------------------------
metadata = MetaData()

sales = Table(
    "sales",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("product", String),
    Column("region", String),
    Column("quantity", Integer),
    Column("amount", Float),
    Column("created_at", DateTime, default=lambda: datetime.now(timezone.utc)),
)


# ---------------------------------------------------------------------------
# Engine / connection handling
# ---------------------------------------------------------------------------
@st.cache_resource
def get_engine():
    """Return ``(engine, backend_name)``.

    Attempts to connect to the PostgreSQL container described in
    ``postgresql/README.md``. If that fails for any reason (e.g. Docker is not
    running), fall back to a local SQLite database under ``data/fallback.db``.

    A short connect timeout keeps the app responsive when Postgres is absent.
    """
    postgres_url = "postgresql+psycopg2://myuser:mypassword@localhost:5432/mydatabase"
    try:
        engine = create_engine(
            postgres_url,
            connect_args={"connect_timeout": 3},
        )
        # Actually test the connection so we don't silently defer failures.
        with engine.connect():
            pass
        return engine, "PostgreSQL"
    except Exception:
        fallback_path = Path(__file__).resolve().parents[1] / "data" / "fallback.db"
        engine = create_engine(f"sqlite:///{fallback_path}")
        return engine, "SQLite (fallback)"


# ---------------------------------------------------------------------------
# CRUD / query helpers
# ---------------------------------------------------------------------------
def init_db(engine) -> None:
    """Create the ``sales`` table if it does not already exist."""
    metadata.create_all(engine)


def insert_sale(
    engine, product: str, region: str, quantity: int, amount: float
) -> None:
    """Insert a single sale row."""
    stmt = insert(sales).values(
        product=product,
        region=region,
        quantity=quantity,
        amount=amount,
    )
    with engine.begin() as conn:
        conn.execute(stmt)


def fetch_sales(engine) -> pd.DataFrame:
    """Return all sales rows ordered by id as a DataFrame."""
    stmt = select(sales).order_by(sales.c.id)
    with engine.connect() as conn:
        return pd.read_sql(stmt, conn)


def total_sales_by_region(engine) -> pd.DataFrame:
    """Return total sales amount per region, highest first.

    Columns: ``region``, ``total_sales``.
    """
    stmt = (
        select(
            sales.c.region.label("region"),
            func.sum(sales.c.amount).label("total_sales"),
        )
        .group_by(sales.c.region)
        .order_by(func.sum(sales.c.amount).desc())
    )
    with engine.connect() as conn:
        return pd.read_sql(stmt, conn)
