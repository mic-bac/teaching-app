"""Unit tests for the shared ``utils`` modules.

These exercise the pure logic (data generation, benchmarks, DB CRUD) without a
running Streamlit server. The DB tests use a temporary SQLite file so they are
fully deterministic and never depend on Docker/PostgreSQL.
"""

import http.server
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests
from sqlalchemy import create_engine

from utils import data_generator, compute_utils, db_utils, io_utils, teaching
from utils import recommender_utils


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


def test_split_script_blocks_python_notebook():
    source = (
        "# %%\n"
        "# --- 1. Imports ---\n"
        "import numpy as np\n"
        "\n"
        "# %% [markdown]\n"
        "# ## Narrative that must not leak into a code block\n"
        "\n"
        "# %%\n"
        "# --- 2. Serial ---\n"
        "results = [f(row) for row in data]\n"
    )
    blocks = teaching.split_script_blocks(source)
    assert [b["number"] for b in blocks] == [1, 2]
    assert blocks[0]["title"] == "Imports"
    # Cell boundary (# %%) stops the block: the markdown narrative is excluded.
    assert blocks[0]["code"] == "import numpy as np"
    assert "Narrative" not in blocks[0]["code"]
    assert blocks[1]["code"] == "results = [f(row) for row in data]"


def test_split_script_blocks_real_scripts_align():
    root = Path(__file__).resolve().parents[1]
    py_blocks = teaching.split_script_blocks(
        (root / "parallelization" / "parallel" / "parallel.py").read_text()
    )
    r_blocks = teaching.split_script_blocks(
        (root / "parallelization" / "parallel" / "parallel.R").read_text()
    )
    py_nums = {b["number"] for b in py_blocks}
    r_nums = {b["number"] for b in r_blocks}
    # The two languages share the numbered steps 1..7 so they can be paired.
    assert {1, 2, 3, 4, 5, 6, 7} <= (py_nums & r_nums)
    # Real code made it into the parsed blocks.
    py_by_num = {b["number"]: b["code"] for b in py_blocks}
    assert "Pool" in py_by_num[6]
    assert "np.mean" in py_by_num[7]


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


# ---------------------------------------------------------------------------
# io_utils
#
# These exercise the *real* download paths (requests, aiohttp, multiprocessing)
# against a throwaway HTTP server bound to localhost, so the tests are fully
# deterministic and never touch the public internet.
# ---------------------------------------------------------------------------
class _QuietHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"hello world"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence per-request logging
        pass


@pytest.fixture(scope="module")
def local_url():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _QuietHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _, port = server.server_address
    try:
        yield f"http://127.0.0.1:{port}/"
    finally:
        server.shutdown()
        server.server_close()


def test_run_sequential_downloads(local_url):
    results, elapsed = io_utils.run_sequential([local_url] * 5)
    assert len(results) == 5
    assert all(r > 0 for r in results)  # real bytes were read back
    assert elapsed >= 0


def test_run_threaded_downloads(local_url):
    results, elapsed = io_utils.run_threaded([local_url] * 5)
    assert len(results) == 5
    assert all(r > 0 for r in results)
    assert elapsed >= 0


def test_run_async_downloads(local_url):
    results, elapsed = io_utils.run_async([local_url] * 5)
    assert len(results) == 5
    assert all(r > 0 for r in results)
    assert elapsed >= 0


def test_download_site_unreachable_returns_zero():
    # Graceful degradation: a dead endpoint yields 0 bytes instead of raising,
    # so a flaky network can never crash the live demo.
    with requests.Session() as session:
        assert io_utils.download_site(session, "http://127.0.0.1:1/") == 0


def test_run_io_benchmark_keys_and_types(local_url):
    res = io_utils.run_io_benchmark(local_url, count=5)
    assert set(res.keys()) == {"Sequential", "Threaded", "Async", "Multiprocessing"}
    assert all(isinstance(v, float) for v in res.values())
    assert all(v >= 0 for v in res.values())


# ---------------------------------------------------------------------------
# recommender_utils — deterministic tests on tiny in-memory fixtures
# ---------------------------------------------------------------------------
# Content-Based Filtering
def _tiny_movies():
    # Genres are already space-joined (as build_tfidf expects).
    return pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4],
            "title": ["A", "B", "C", "D"],
            "genres": ["Action Adventure", "Action Adventure", "Comedy Romance", "Comedy"],
        }
    )


def test_cbf_recommendations_exclude_input_and_rank_similar():
    movies = _tiny_movies()
    _, tfidf, title_to_idx = recommender_utils.build_tfidf(movies)
    recs = recommender_utils.get_recommendations("A", tfidf, movies, title_to_idx, top_n=10)
    titles = list(recs["title"])
    assert "A" not in titles  # never recommend the input itself
    assert len(titles) == 3  # only three other movies exist
    # "B" shares both genres with "A", so it must rank first.
    assert titles[0] == "B"
    assert recs["similarity"].iloc[0] == pytest.approx(1.0, abs=1e-6)


def test_cbf_unknown_title_returns_message():
    movies = _tiny_movies()
    _, tfidf, title_to_idx = recommender_utils.build_tfidf(movies)
    out = recommender_utils.get_recommendations("Nope", tfidf, movies, title_to_idx)
    assert isinstance(out, str) and "not found" in out


def test_cbf_intra_list_similarity_range():
    movies = _tiny_movies()
    _, tfidf, title_to_idx = recommender_utils.build_tfidf(movies)
    ils = recommender_utils.intra_list_similarity(["A", "B"], tfidf, title_to_idx)
    # A and B are identical in genre space → similarity 1.0.
    assert ils == pytest.approx(1.0, abs=1e-6)
    assert recommender_utils.intra_list_similarity(["A"], tfidf, title_to_idx) == 0.0


# Neighborhood collaborative filtering
def test_user_based_recommendations_are_weighted_and_exclude_seen():
    user_item = pd.DataFrame(
        {"M1": [5.0, 4.0, np.nan], "M2": [np.nan, 5.0, 1.0], "M3": [1.0, np.nan, 5.0]},
        index=[1, 2, 3],
    )
    user_sim = recommender_utils.user_similarity(user_item)
    recs = recommender_utils.get_user_based_recommendations(1, user_item, user_sim)
    # User 1 has already rated M1 and M3, so only M2 can be recommended.
    assert "M2" in recs.index
    assert "M1" not in recs.index and "M3" not in recs.index


def test_item_based_recommendations_on_demand():
    user_item = pd.DataFrame(
        {"M1": [5.0, 4.0, np.nan], "M2": [5.0, 4.0, np.nan], "M3": [1.0, np.nan, 5.0]},
        index=[1, 2, 3],
    )
    sparse, labels = recommender_utils.build_item_user_sparse(user_item)
    recs = recommender_utils.get_item_based_recommendations("M1", sparse, labels)
    # M2 is co-rated identically to M1 → most similar; M1 itself is excluded.
    assert recs.index[0] == "M2"
    assert "M1" not in recs.index


# Matrix completion: bias baseline + factorization
def test_fit_baseline_learns_item_bias_direction():
    # Item 0 is always loved (5), item 1 always panned (1) by all users.
    rows = []
    for u in range(5):
        rows.append([u, 0, 5.0])
        rows.append([u, 1, 1.0])
    ratings = np.array(rows, dtype=float)
    mu, b_u, b_i = recommender_utils.fit_baseline(ratings, n_users=5, n_items=2)
    assert b_i[0] > b_i[1]  # loved item has higher bias than panned one
    preds = recommender_utils.predict_baseline(mu, b_u, b_i, [0], [0])
    assert 0.5 <= preds[0] <= 5.0  # clipped to the valid range


def test_matrix_factorization_reduces_training_error():
    # Structured data: rating depends on a user offset + an item offset.
    rng = np.random.default_rng(0)
    u_off = rng.uniform(-1, 1, 8)
    i_off = rng.uniform(-1, 1, 8)
    rows = [[u, i, float(np.clip(3 + u_off[u] + i_off[i], 0.5, 5.0))]
            for u in range(8) for i in range(8)]
    ratings = np.array(rows, dtype=float)

    model = recommender_utils.MatrixFactorization(8, 8, n_factors=3, n_epochs=15)
    history = []
    model.fit(ratings, on_epoch_end=lambda e, r: history.append(r))
    assert history[-1] < history[0]  # SGD lowers the training RMSE
    preds = model.predict(ratings[:, 0], ratings[:, 1])
    assert np.all((preds >= 0.5) & (preds <= 5.0))


# Association rules
def _tiny_groceries():
    # Members 1-4 buy milk+bread together; member 5 buys soda alone.
    return pd.DataFrame(
        {
            "Member_number": [1, 1, 2, 2, 3, 3, 4, 4, 5],
            "Date": ["d"] * 9,
            "itemDescription": ["milk", "bread", "milk", "bread",
                                 "milk", "bread", "milk", "bread", "soda"],
        }
    )


def test_association_rules_and_basket_recommendation():
    trans = recommender_utils.build_transactions(_tiny_groceries())
    assert trans.shape[0] == 5  # five baskets
    itemsets = recommender_utils.run_apriori(trans, min_support=0.1)
    rules = recommender_utils.make_rules(itemsets, min_confidence=0.5)
    # milk -> bread is a certainty here.
    assert (rules["rule_str"] == "milk → bread").any()

    recs = recommender_utils.get_basket_recommendations("milk", rules)
    assert "bread" in list(recs["recommendation"])


def test_basket_recommendation_unknown_item():
    trans = recommender_utils.build_transactions(_tiny_groceries())
    rules = recommender_utils.make_rules(
        recommender_utils.run_apriori(trans, min_support=0.1), min_confidence=0.5
    )
    out = recommender_utils.get_basket_recommendations("caviar", rules)
    assert isinstance(out, str) and "No rules" in out


def test_taxonomy_dot_is_graphviz():
    dot = recommender_utils.taxonomy_dot()
    assert dot.startswith("digraph")
    assert "Collaborative" in dot and "Content-Based" in dot
