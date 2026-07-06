"""Database utilities for the teaching app.

Provides a small, backend-agnostic data layer built on SQLAlchemy Core so the
same code works against PostgreSQL (running in Docker) and a local SQLite
fallback. The engine is created lazily via :func:`get_engine`; nothing connects
to a database at import time.
"""

import random
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
    delete,
    func,
    insert,
    inspect,
    select,
    text,
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


def _resync_id_sequence(conn) -> None:
    """Realign PostgreSQL's id sequence to ``MAX(id)`` on an open connection.

    Postgres keeps the auto-increment sequence independent of the rows, so after
    deletes the next inserted id can "jump" (e.g. …3 → 105). This resets the
    sequence so the next insert continues contiguously at ``MAX(id) + 1`` (or at 1
    when the table is empty). Existing ids are never changed.

    No-op on SQLite, whose ``INTEGER PRIMARY KEY`` already reuses ``MAX(id) + 1``.
    """
    if conn.engine.dialect.name != "postgresql":
        return
    conn.execute(
        text(
            "SELECT setval("
            "  pg_get_serial_sequence('sales', 'id'),"
            "  COALESCE(MAX(id), 1),"
            "  MAX(id) IS NOT NULL"
            ") FROM sales"
        )
    )


def resync_id_sequence(engine) -> None:
    """Public wrapper around :func:`_resync_id_sequence` (opens its own transaction)."""
    with engine.begin() as conn:
        _resync_id_sequence(conn)


def insert_sale(
    engine, product: str, region: str, quantity: int, amount: float
) -> None:
    """Insert a single sale row.

    Realigns the id sequence first so the new row's id follows on contiguously
    (no jumps after prior deletes).
    """
    stmt = insert(sales).values(
        product=product,
        region=region,
        quantity=quantity,
        amount=amount,
    )
    with engine.begin() as conn:
        _resync_id_sequence(conn)
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


# ---------------------------------------------------------------------------
# Introspection / teaching helpers
# ---------------------------------------------------------------------------
REGIONS = ["North", "South", "East", "West"]
SAMPLE_PRODUCTS = ["Widget", "Gadget", "Gizmo", "Doohickey", "Sprocket", "Thingamajig"]


def count_sales(engine) -> int:
    """Return the number of rows currently in the ``sales`` table."""
    stmt = select(func.count()).select_from(sales)
    with engine.connect() as conn:
        return int(conn.execute(stmt).scalar_one())


def get_schema(engine) -> pd.DataFrame:
    """Return the ``sales`` table schema as a DataFrame.

    Columns: ``column``, ``type``, ``nullable``, ``primary_key``. Uses SQLAlchemy's
    dialect-aware inspector, so it reflects whichever backend is live.
    """
    inspector = inspect(engine)
    rows = [
        {
            "column": col["name"],
            "type": str(col["type"]),
            "nullable": bool(col["nullable"]),
            "primary_key": bool(col.get("primary_key", False)),
        }
        for col in inspector.get_columns("sales")
    ]
    return pd.DataFrame(rows, columns=["column", "type", "nullable", "primary_key"])


def seed_sample_sales(engine, n: int = 20) -> int:
    """Insert ``n`` random fake sales in one bulk statement. Returns ``n``.

    Handy for making the table and charts non-empty so the lesson is immediately
    hands-on.
    """
    rows = [
        {
            "product": random.choice(SAMPLE_PRODUCTS),
            "region": random.choice(REGIONS),
            "quantity": random.randint(1, 20),
            "amount": round(random.uniform(5.0, 500.0), 2),
        }
        for _ in range(n)
    ]
    with engine.begin() as conn:
        _resync_id_sequence(conn)
        conn.execute(insert(sales), rows)
    return n


def delete_recent_sales(engine, n: int = 20) -> int:
    """Delete the ``n`` most recently added sales (highest ``id`` first).

    Returns the number of rows actually deleted (may be fewer than ``n`` if the
    table has fewer rows). Handy for undoing a ``seed_sample_sales`` call.
    """
    ids_stmt = select(sales.c.id).order_by(sales.c.id.desc()).limit(n)
    with engine.begin() as conn:
        ids = [row[0] for row in conn.execute(ids_stmt)]
        if not ids:
            return 0
        conn.execute(delete(sales).where(sales.c.id.in_(ids)))
        # Roll the sequence back so the next insert continues without a jump.
        _resync_id_sequence(conn)
    return len(ids)


def masked_url(engine) -> str:
    """Return the engine's connection URL with the password hidden."""
    return engine.url.render_as_string(hide_password=True)


def run_query(engine, sql: str, allow_writes: bool = False):
    """Execute a raw SQL string for the SQL-playground lesson.

    Returns ``(dataframe_or_none, message)``:

    - ``SELECT`` / ``WITH`` queries → ``(DataFrame, "N rows")``.
    - Other statements are refused unless ``allow_writes`` is True; when allowed
      they run in a transaction and return ``(None, "OK — <rowcount> rows affected")``.
    - SQL/database errors are caught and returned as ``(None, "Error: ...")`` so the
      page can display them as a teaching moment rather than crashing.
    """
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        return None, "Enter a SQL statement to run."

    first_word = stripped.split(None, 1)[0].lower()
    is_read = first_word in ("select", "with")

    if not is_read and not allow_writes:
        return (
            None,
            f"Read-only mode: '{first_word.upper()}' statements are blocked. "
            "Tick 'Allow write/DDL statements' to run this.",
        )

    try:
        if is_read:
            with engine.connect() as conn:
                df = pd.read_sql(text(stripped), conn)
            return df, f"Returned {len(df)} row(s)."
        with engine.begin() as conn:
            result = conn.execute(text(stripped))
            affected = result.rowcount if result.rowcount is not None else 0
        # Best-effort: keep ids contiguous after a raw INSERT/DELETE, but never let
        # this undo the user's committed statement (e.g. if they dropped the table).
        try:
            resync_id_sequence(engine)
        except Exception:
            pass
        return None, f"OK — {affected} row(s) affected."
    except Exception as exc:  # surface DB errors to the learner
        return None, f"Error: {exc}"
