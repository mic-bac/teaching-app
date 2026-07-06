"""Unit tests for the shared ``utils`` modules.

These exercise the pure logic (data generation, benchmarks, DB CRUD) without a
running Streamlit server. The DB tests use a temporary SQLite file so they are
fully deterministic and never depend on Docker/PostgreSQL.
"""

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from utils import data_generator, compute_utils, db_utils


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
