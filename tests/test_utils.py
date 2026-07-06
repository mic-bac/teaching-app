"""Unit tests for the shared ``utils`` modules.

These exercise the pure logic (data generation, benchmarks, DB CRUD) without a
running Streamlit server. The DB tests use a temporary SQLite file so they are
fully deterministic and never depend on Docker/PostgreSQL.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from utils import data_generator, compute_utils, db_utils, teaching


# ---------------------------------------------------------------------------
# data_generator
# ---------------------------------------------------------------------------
def test_generate_dataset_shape_and_columns():
    df = data_generator.generate_dataset(50, 4)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (50, 4)
    assert list(df.columns) == ["col1", "col2", "col3", "col4"]
    # All values are random floats in [0, 1).
    assert df.to_numpy().min() >= 0.0
    assert df.to_numpy().max() < 1.0


def test_generate_dataset_default_cols():
    df = data_generator.generate_dataset(10)
    assert df.shape == (10, 10)


# ---------------------------------------------------------------------------
# compute_utils
# ---------------------------------------------------------------------------
def test_run_serial_and_vectorized_agree():
    data = np.random.rand(200, 6)
    serial_results, serial_t = compute_utils.run_serial(data)
    vec_results, vec_t = compute_utils.run_vectorized(data)
    assert len(serial_results) == 200
    assert vec_results.shape == (200,)
    assert serial_t >= 0 and vec_t >= 0
    # Serial and vectorized compute the same row means.
    np.testing.assert_allclose(np.array(serial_results), vec_results, rtol=1e-9)


def test_run_parallel_returns_results():
    data = np.random.rand(100, 5)
    results, elapsed = compute_utils.run_parallel(data, n_cpu=2)
    assert len(results) == 100
    assert elapsed >= 0


def test_run_benchmark_keys_and_types():
    res = compute_utils.run_benchmark(500)
    assert set(res.keys()) == {"Serial", "Parallel", "Vectorized"}
    assert all(isinstance(v, float) for v in res.values())
    assert all(v >= 0 for v in res.values())
    assert res["Serial"] > 0  # the sleep-based serial loop always takes real time


# ---------------------------------------------------------------------------
# db_utils — CRUD against a temporary SQLite database
# ---------------------------------------------------------------------------
def _temp_engine(tmp_path):
    return create_engine(f"sqlite:///{tmp_path / 'test.db'}")


def test_db_init_insert_fetch(tmp_path):
    engine = _temp_engine(tmp_path)
    db_utils.init_db(engine)

    db_utils.insert_sale(engine, "Widget", "North", 3, 10.0)
    db_utils.insert_sale(engine, "Gadget", "South", 1, 99.5)

    df = db_utils.fetch_sales(engine)
    assert list(df.columns) == ["id", "product", "region", "quantity", "amount", "created_at"]
    assert len(df) == 2
    assert set(df["product"]) == {"Widget", "Gadget"}


def test_db_total_sales_by_region(tmp_path):
    engine = _temp_engine(tmp_path)
    db_utils.init_db(engine)
    db_utils.insert_sale(engine, "A", "North", 1, 40.0)
    db_utils.insert_sale(engine, "B", "North", 1, 10.0)
    db_utils.insert_sale(engine, "C", "South", 1, 99.5)

    agg = db_utils.total_sales_by_region(engine)
    assert list(agg.columns) == ["region", "total_sales"]
    # Ordered by total_sales descending: South (99.5) before North (50.0).
    assert list(agg["region"]) == ["South", "North"]
    totals = dict(zip(agg["region"], agg["total_sales"]))
    assert totals["North"] == 50.0
    assert totals["South"] == 99.5


def test_db_total_sales_empty(tmp_path):
    engine = _temp_engine(tmp_path)
    db_utils.init_db(engine)
    agg = db_utils.total_sales_by_region(engine)
    assert agg.empty


# ---------------------------------------------------------------------------
# db_utils — teaching helpers (schema, count, seed, masked url, run_query)
# ---------------------------------------------------------------------------
def test_get_schema_lists_columns(tmp_path):
    engine = _temp_engine(tmp_path)
    db_utils.init_db(engine)
    schema = db_utils.get_schema(engine)
    assert list(schema.columns) == ["column", "type", "nullable", "primary_key"]
    assert set(schema["column"]) == {
        "id",
        "product",
        "region",
        "quantity",
        "amount",
        "created_at",
    }
    # The primary key is flagged.
    assert bool(schema.loc[schema["column"] == "id", "primary_key"].iloc[0]) is True


def test_count_and_seed(tmp_path):
    engine = _temp_engine(tmp_path)
    db_utils.init_db(engine)
    assert db_utils.count_sales(engine) == 0
    added = db_utils.seed_sample_sales(engine, 7)
    assert added == 7
    assert db_utils.count_sales(engine) == 7
    # Seeded rows use the known regions/products.
    df = db_utils.fetch_sales(engine)
    assert set(df["region"]).issubset(set(db_utils.REGIONS))


def test_delete_recent_undoes_seed(tmp_path):
    engine = _temp_engine(tmp_path)
    db_utils.init_db(engine)
    db_utils.insert_sale(engine, "Keep", "North", 1, 1.0)  # a pre-existing row
    db_utils.seed_sample_sales(engine, 10)
    assert db_utils.count_sales(engine) == 11

    removed = db_utils.delete_recent_sales(engine, 10)
    assert removed == 10
    # Only the newest 10 (the seeded rows) were removed; the original remains.
    df = db_utils.fetch_sales(engine)
    assert len(df) == 1
    assert df["product"].iloc[0] == "Keep"


def test_delete_recent_caps_at_available_rows(tmp_path):
    engine = _temp_engine(tmp_path)
    db_utils.init_db(engine)
    db_utils.seed_sample_sales(engine, 3)
    # Asking for more than exist deletes only what's there and reports the count.
    removed = db_utils.delete_recent_sales(engine, 50)
    assert removed == 3
    assert db_utils.count_sales(engine) == 0


def test_resync_is_noop_on_sqlite(tmp_path):
    # SQLite already reuses MAX(id)+1, so resync must run cleanly and change nothing.
    engine = _temp_engine(tmp_path)
    db_utils.init_db(engine)
    db_utils.seed_sample_sales(engine, 3)
    db_utils.resync_id_sequence(engine)  # must not raise
    assert db_utils.count_sales(engine) == 3


def test_ids_stay_contiguous_after_delete_then_insert(tmp_path):
    engine = _temp_engine(tmp_path)
    db_utils.init_db(engine)
    db_utils.seed_sample_sales(engine, 5)  # ids 1..5
    db_utils.delete_recent_sales(engine, 2)  # removes 4,5 → max id now 3
    db_utils.insert_sale(engine, "Next", "North", 1, 1.0)

    df = db_utils.fetch_sales(engine)
    # No jump: the new row continues at 4, not 6.
    assert list(df["id"]) == [1, 2, 3, 4]


def test_masked_url_hides_password():
    engine = create_engine(
        "postgresql+psycopg2://myuser:secret@localhost:5432/mydatabase"
    )
    url = db_utils.masked_url(engine)
    assert "secret" not in url
    assert "myuser" in url


def test_run_query_select_returns_dataframe(tmp_path):
    engine = _temp_engine(tmp_path)
    db_utils.init_db(engine)
    db_utils.insert_sale(engine, "Widget", "North", 3, 10.0)
    df, msg = db_utils.run_query(engine, "SELECT * FROM sales;")
    assert df is not None
    assert len(df) == 1
    assert "1 row" in msg


def test_run_query_blocks_writes_by_default(tmp_path):
    engine = _temp_engine(tmp_path)
    db_utils.init_db(engine)
    db_utils.insert_sale(engine, "Widget", "North", 3, 10.0)
    df, msg = db_utils.run_query(engine, "DELETE FROM sales;")
    assert df is None
    assert "Read-only" in msg
    # Nothing was deleted.
    assert db_utils.count_sales(engine) == 1


def test_run_query_allows_writes_when_enabled(tmp_path):
    engine = _temp_engine(tmp_path)
    db_utils.init_db(engine)
    db_utils.insert_sale(engine, "Widget", "North", 3, 10.0)
    df, msg = db_utils.run_query(engine, "DELETE FROM sales;", allow_writes=True)
    assert df is None
    assert "OK" in msg
    assert db_utils.count_sales(engine) == 0


def test_run_query_reports_sql_errors(tmp_path):
    engine = _temp_engine(tmp_path)
    db_utils.init_db(engine)
    df, msg = db_utils.run_query(engine, "SELECT * FROM nonexistent_table;")
    assert df is None
    assert msg.startswith("Error")


# ---------------------------------------------------------------------------
# teaching helpers
# ---------------------------------------------------------------------------
def test_source_of_returns_code():
    src = teaching.source_of(db_utils.fetch_sales)
    assert "def fetch_sales" in src


def test_postgres_docker_command_from_repo():
    cmd = teaching.postgres_docker_command()
    assert "docker run" in cmd
    assert "postgres-db" in cmd


def test_postgres_docker_command_fallback(tmp_path):
    missing = Path(tmp_path) / "nope.md"
    cmd = teaching.postgres_docker_command(missing)
    assert "postgresql/README.md" in cmd


def test_architecture_dot_highlights_active_backend():
    pg = teaching.architecture_dot("PostgreSQL")
    sqlite = teaching.architecture_dot("SQLite (fallback)")
    assert pg.startswith("digraph")
    # The highlighted backend gets a thicker border (penwidth=3).
    assert "penwidth=3" in pg and "penwidth=3" in sqlite
    assert pg != sqlite
