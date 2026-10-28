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
from utils import propensity_utils, survival_utils
from utils import segmentation_utils


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


# ---------------------------------------------------------------------------
# propensity_utils — deterministic tests on tiny synthetic customer bases
# ---------------------------------------------------------------------------
def _tiny_churn(n_customers: int = 400) -> pd.DataFrame:
    """A small, seeded customer base with a real (learnable) churn signal."""
    return propensity_utils.make_synthetic_churn(n_customers, random_state=0)


def test_synthetic_churn_has_the_kaggle_columns_and_a_signal():
    frame = _tiny_churn(600)
    assert list(frame.columns) == propensity_utils.CHURN_COLUMNS
    assert frame["Churn"].isin([0, 1]).all()
    # Both classes must be present, otherwise nothing downstream can be fitted.
    assert 0.1 < frame["Churn"].mean() < 0.9
    # The planted signal: churners call support more often than the rest.
    assert frame.loc[frame["Churn"] == 1, "Support Calls"].mean() > frame.loc[
        frame["Churn"] == 0, "Support Calls"
    ].mean()


def test_load_churn_falls_back_to_synthetic_when_csvs_are_missing(tmp_path):
    frame, source = propensity_utils.load_churn(data_dir=tmp_path, sample_size=300)
    assert source == "synthetic"  # the page shows a warning on this branch
    assert len(frame) == 300
    assert list(frame.columns) == propensity_utils.CHURN_COLUMNS


def test_stratified_sample_keeps_the_churn_rate():
    frame = _tiny_churn(1000)
    sampled = propensity_utils._stratified_sample(frame, 200, random_state=0)
    assert len(sampled) == 200
    assert sampled["Churn"].mean() == pytest.approx(frame["Churn"].mean(), abs=0.02)


def test_prepare_features_drops_the_id_and_encodes_categoricals():
    X, y, encoders = propensity_utils.prepare_features(_tiny_churn())
    assert "CustomerID" not in X.columns  # a key, not a feature
    assert "Churn" not in X.columns
    assert set(encoders) == {"Gender", "Subscription Type", "Contract Length"}
    # Every column must be numeric for scikit-learn to accept it.
    assert all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns)
    assert len(y) == len(X)


def test_split_and_scale_is_stratified_and_standardised():
    X, y, _ = propensity_utils.prepare_features(_tiny_churn())
    split = propensity_utils.split_and_scale(X, y, test_size=0.25, random_state=42)
    assert len(split["X_train"]) == 300 and len(split["X_test"]) == 100
    assert split["y_train"].mean() == pytest.approx(split["y_test"].mean(), abs=0.05)
    # Scaled training features are centred on 0 with unit spread.
    np.testing.assert_allclose(split["X_train_scaled"].mean(axis=0), 0, atol=1e-9)
    np.testing.assert_allclose(split["X_train_scaled"].std(axis=0), 1, atol=1e-9)


def test_fit_and_score_returns_every_metric_in_range():
    X, y, _ = propensity_utils.prepare_features(_tiny_churn(500))
    split = propensity_utils.split_and_scale(X, y)
    result = propensity_utils.fit_and_score(
        propensity_utils.build_model("Logistic Regression"),
        split["X_train_scaled"],
        split["y_train"],
        split["X_test_scaled"],
        split["y_test"],
    )
    assert set(result["metrics"]) == {
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
        "PR-AUC",
    }
    assert all(0.0 <= v <= 1.0 for v in result["metrics"].values())
    # The planted signal is learnable, so this must beat coin-flipping clearly.
    assert result["metrics"]["ROC-AUC"] > 0.7
    assert set(np.unique(result["predictions"])) <= {0, 1}


def test_build_model_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="Unknown model"):
        propensity_utils.build_model("Random Guessing")


def test_threshold_sweep_trades_precision_against_recall():
    X, y, _ = propensity_utils.prepare_features(_tiny_churn(500))
    split = propensity_utils.split_and_scale(X, y)
    result = propensity_utils.fit_and_score(
        propensity_utils.build_model("Logistic Regression"),
        split["X_train_scaled"],
        split["y_train"],
        split["X_test_scaled"],
        split["y_test"],
    )
    sweep = propensity_utils.threshold_sweep(split["y_test"], result["probabilities"], steps=9)
    assert list(sweep.columns) == ["threshold", "Precision", "Recall", "F1-Score"]
    # Recall can only fall as the bar for "predict churn" is raised.
    assert (sweep["Recall"].diff().dropna() <= 1e-9).all()


def test_confusion_frame_is_labelled_and_totals_the_test_set():
    confusion = propensity_utils.confusion_frame([0, 0, 1, 1], [0, 1, 1, 1])
    assert list(confusion.index) == ["Actually stayed", "Actually churned"]
    assert list(confusion.columns) == ["Predicted stay", "Predicted churn"]
    assert confusion.to_numpy().sum() == 4
    assert confusion.loc["Actually stayed", "Predicted churn"] == 1  # one false alarm


def test_cross_validation_returns_one_score_per_fold():
    X, y, _ = propensity_utils.prepare_features(_tiny_churn(400))
    split = propensity_utils.split_and_scale(X, y)
    scores = propensity_utils.cross_validate_model(
        propensity_utils.build_model("Logistic Regression"),
        split["X_train_scaled"],
        split["y_train"],
        n_splits=3,
    )
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_tune_model_reports_what_the_search_cost():
    X, y, _ = propensity_utils.prepare_features(_tiny_churn(400))
    split = propensity_utils.split_and_scale(X, y)
    tuned = propensity_utils.tune_model(
        "Logistic Regression", split["X_train_scaled"], split["y_train"], cv=2
    )
    grid_size = len(propensity_utils.MODEL_SPECS["Logistic Regression"]["grid"]["C"])
    assert tuned["n_candidates"] == grid_size  # a grid search tries all of them
    assert tuned["n_fits"] == grid_size * 2
    assert "C" in tuned["best_params"]
    assert len(tuned["results"]) == grid_size


def test_learning_and_validation_curves_have_both_series():
    X, y, _ = propensity_utils.prepare_features(_tiny_churn(400))
    split = propensity_utils.split_and_scale(X, y)
    model = propensity_utils.build_model("Logistic Regression")

    learning = propensity_utils.learning_curve_frame(
        model, split["X_train_scaled"], split["y_train"], cv=2, n_points=3
    )
    assert len(learning) == 3
    assert learning["train_size"].is_monotonic_increasing
    assert (learning["validation_mean"] > 0.5).all()

    validation = propensity_utils.validation_curve_frame(
        model, split["X_train_scaled"], split["y_train"], "C", [0.01, 1.0], cv=2
    )
    assert list(validation["value"]) == [0.01, 1.0]
    assert {"train_mean", "validation_mean"} <= set(validation.columns)


def test_feature_importance_switches_on_what_the_model_exposes():
    X, y, _ = propensity_utils.prepare_features(_tiny_churn(400))
    split = propensity_utils.split_and_scale(X, y)

    linear = propensity_utils.build_model("Logistic Regression").fit(
        split["X_train_scaled"], split["y_train"]
    )
    ranked = propensity_utils.feature_importance(linear, split["feature_names"], top_n=3)
    assert list(ranked.columns) == ["feature", "weight", "kind"]
    assert ranked["kind"].iloc[0] == "|coefficient|"
    assert ranked["weight"].is_monotonic_decreasing

    trees = propensity_utils.build_model("XGBoost", n_estimators=10).fit(
        split["X_train"], split["y_train"]
    )
    assert propensity_utils.feature_importance(trees, split["feature_names"])["kind"].iloc[0] == "split gain"

    # A neural net offers no built-in attribution: an empty frame, not a crash.
    net = propensity_utils.build_model("Neural Network", max_iter=5)
    net.fit(split["X_train_scaled"], split["y_train"])
    assert propensity_utils.feature_importance(net, split["feature_names"]).empty


def test_risk_segments_and_summary():
    probabilities = np.array([0.05, 0.35, 0.5, 0.65, 0.85, 0.95])
    actual = np.array([0, 0, 0, 1, 1, 1])
    segments = propensity_utils.risk_segments(probabilities, actual, low=0.4, high=0.7)
    assert list(segments["Segment"]) == [
        "Low risk",
        "Low risk",
        "Medium risk",
        "Medium risk",
        "High risk",
        "High risk",
    ]

    summary = propensity_utils.segment_summary(segments)
    assert list(summary["Customers"]) == [2, 2, 2]
    assert summary["Share"].sum() == pytest.approx(1.0)
    # The realised churn rate must climb across the segments — that climb is the value.
    assert summary["Actual churn rate"].is_monotonic_increasing


def test_lookalike_timeline_windows_are_contiguous():
    timeline = propensity_utils.lookalike_timeline(12, 2, 4)
    assert list(timeline["window"]) == ["Observation", "Buffer", "Outcome"]
    assert list(timeline["weeks"]) == [12, 2, 4]
    # No gaps and no overlaps: each window starts where the previous one ended.
    assert list(timeline["start"])[1:] == list(timeline["end"])[:-1]
    assert timeline["end"].iloc[-1] == 18


def test_propensity_pipeline_dot_is_graphviz():
    dot = propensity_utils.propensity_pipeline_dot()
    assert dot.startswith("digraph")
    assert "Risk segments" in dot and "Cross-validate" in dot


# ---------------------------------------------------------------------------
# survival_utils — deterministic tests on tiny synthetic customer bases
# ---------------------------------------------------------------------------
def test_prepare_survival_builds_the_structured_target():
    frame = _tiny_churn(300)
    X, y, names = survival_utils.prepare_survival(frame)
    assert y.dtype.names == ("event", "time")
    assert y["event"].dtype == bool
    assert len(y) == len(frame)
    # Tenure is the time being modelled, so it must never become a feature.
    assert "Tenure" not in names and "CustomerID" not in names and "Churn" not in names
    assert all(str(dtype) == "float64" for dtype in X.dtypes)


def test_kaplan_meier_starts_high_and_never_rises():
    frame = _tiny_churn(400)
    curve = survival_utils.kaplan_meier(frame["Tenure"], frame["Churn"])
    assert {"time", "survival", "lower", "upper"} <= set(curve.columns)
    assert curve["survival"].iloc[0] <= 1.0
    # S(t) is a non-increasing staircase by construction.
    assert (curve["survival"].diff().dropna() <= 1e-12).all()
    assert (curve["lower"] <= curve["survival"] + 1e-9).all()
    assert (curve["survival"] <= curve["upper"] + 1e-9).all()


def test_km_by_group_returns_one_curve_per_level():
    frame = _tiny_churn(400)
    curves = survival_utils.km_by_group(frame, "Contract Length")
    assert set(curves["Contract Length"]) == {"Annual", "Monthly", "Quarterly"}
    for _, group in curves.groupby("Contract Length"):
        assert (group["survival"].diff().dropna() <= 1e-12).all()


def test_censoring_example_marks_survivors_as_censored():
    example = survival_utils.censoring_example(n_customers=6, seed=0)
    assert len(example) == 6
    assert example.attrs["cutoff"] == 20.0
    # Censored customers are exactly those still running at the cut-off.
    censored = example[~example["event"]]
    assert (censored["observed_until"] == 20.0).all()
    assert (example["duration"] > 0).all()


def test_survival_models_rank_better_than_chance_and_report_metrics():
    frame = _tiny_churn(600)
    X, y, names = survival_utils.prepare_survival(frame)
    X_train, X_test, y_train, y_test = survival_utils.split_survival(X, y, random_state=42)

    cox = survival_utils.fit_cox(X_train, y_train)
    assert survival_utils.c_index(cox, X_test, y_test) > 0.5

    times, in_window = survival_utils.evaluation_window(y_train, y_test)
    assert len(times) > 0
    assert times.max() < y_test["time"].max()  # inside the observed follow-up

    auc_frame, auc_mean = survival_utils.time_dependent_auc(
        cox, y_train, y_test[in_window], X_test[in_window], times
    )
    assert len(auc_frame) == len(times)
    assert 0.0 <= auc_mean <= 1.0

    brier = survival_utils.integrated_brier(cox, y_train, y_test[in_window], X_test[in_window], times)
    assert isinstance(brier, float) and 0.0 <= brier <= 0.25

    coefficients = survival_utils.cox_coefficients(cox, names)
    assert len(coefficients) == len(names)
    np.testing.assert_allclose(
        coefficients["hazard_ratio"], np.exp(coefficients["coefficient"]), rtol=1e-9
    )


def test_score_only_model_gets_a_message_instead_of_a_brier_score():
    frame = _tiny_churn(400)
    X, y, _ = survival_utils.prepare_survival(frame)
    X_train, X_test, y_train, y_test = survival_utils.split_survival(X, y, random_state=42)

    svm = survival_utils.fit_svm(X_train, y_train, max_iter=10)
    times, in_window = survival_utils.evaluation_window(y_train, y_test)

    # It CAN be ranked (that only needs a risk score) ...
    assert 0.0 <= survival_utils.c_index(svm, X_test, y_test) <= 1.0
    # ... but it has no S(t), so both of these explain themselves instead of guessing.
    brier = survival_utils.integrated_brier(svm, y_train, y_test[in_window], X_test[in_window], times)
    assert isinstance(brier, str) and "c-index" in brier
    curves = survival_utils.individual_survival_curves(svm, X_test, [0])
    assert isinstance(curves, str) and "risk score" in curves


def test_individual_survival_curves_are_labelled_and_monotone():
    frame = _tiny_churn(400)
    X, y, _ = survival_utils.prepare_survival(frame)
    X_train, X_test, y_train, _ = survival_utils.split_survival(X, y, random_state=42)

    cox = survival_utils.fit_cox(X_train, y_train)
    curves = survival_utils.individual_survival_curves(cox, X_test, [0, 1], labels=["Ada", "Grace"])
    assert set(curves["customer"]) == {"Ada", "Grace"}
    for _, group in curves.groupby("customer"):
        assert (group["survival"].diff().dropna() <= 1e-12).all()


def test_risk_strata_split_the_test_set_and_separate_the_groups():
    frame = _tiny_churn(600)
    X, y, _ = survival_utils.prepare_survival(frame)
    X_train, X_test, y_train, y_test = survival_utils.split_survival(X, y, random_state=42)

    cox = survival_utils.fit_cox(X_train, y_train)
    curves, summary = survival_utils.risk_strata(cox, X_test, y_test)
    assert list(summary["Risk group"]) == ["Low", "Medium", "High", "Very high"]
    assert summary["Customers"].sum() == len(y_test)
    # A working model puts more churners in the high-risk groups than the low one.
    assert summary["Observed churn rate"].iloc[-1] > summary["Observed churn rate"].iloc[0]
    assert set(curves["risk_group"]) == {"Low", "Medium", "High", "Very high"}


def test_survival_taxonomy_dot_is_graphviz():
    dot = survival_utils.survival_taxonomy_dot()
    assert dot.startswith("digraph")
    assert "Survival analysis" in dot and "Classification" in dot


# ---------------------------------------------------------------------------
# segmentation_utils — the app's adapter over the sibling ``segmentation/`` repo
#
# These deliberately avoid loading the 640,000-row transaction log: the pure
# helpers are exercised on tiny hand-built frames so the suite stays fast and
# deterministic. The one test that touches the sibling only checks that it is
# importable at all.
# ---------------------------------------------------------------------------
def _tiny_transactions() -> pd.DataFrame:
    """Six invoices across three customers, with known recency/frequency/spend."""
    rows = [
        # customer, invoice, day, quantity, price
        (1, "A1", "2011-01-01", 2, 10.0),
        (1, "A1", "2011-01-01", 3, 10.0),  # same invoice — one order, two lines
        (1, "A2", "2011-06-01", 1, 50.0),
        (2, "B1", "2011-02-01", 5, 4.0),
        (3, "C1", "2010-01-01", 1, 7.0),
        (3, "C2", "2010-02-01", 1, 3.0),
    ]
    frame = pd.DataFrame(rows, columns=["Customer ID", "Invoice", "InvoiceDate",
                                        "Quantity", "Price"])
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"])
    frame["Revenue"] = frame["Quantity"] * frame["Price"]
    return frame


def test_sibling_repo_is_importable():
    # The page degrades gracefully without it, but in this workspace it must be there.
    assert segmentation_utils.SIBLING_AVAILABLE, segmentation_utils.SIBLING_ERROR


def test_build_rfm_counts_invoices_not_line_items():
    rfm = segmentation_utils.build_rfm(_tiny_transactions())
    customer_1 = rfm[rfm["Id"] == 1].iloc[0]
    # Customer 1 has three *lines* but only two *orders* — the distinction that
    # makes big baskets look like fake loyalty when you get it wrong.
    assert customer_1["Frequency"] == 2
    assert customer_1["Monetary"] == pytest.approx(2 * 10 + 3 * 10 + 50)
    # Recency is measured from the day after the last invoice, so it is never 0.
    assert rfm["Recency"].min() >= 1


def test_dataset_roles_covers_the_three_methods():
    roles = segmentation_utils.dataset_roles()
    assert list(roles["Method"]) == ["RFM", "CLTV", "Clustering"]
    # The whole point: the two data shapes are not interchangeable.
    assert roles.loc[roles["Method"] == "Clustering", "Needs"].iloc[0] == "Customer attributes"
    assert roles.loc[roles["Method"] == "RFM", "Needs"].iloc[0] == "Transaction log"


def test_segmentation_architecture_dot_is_graphviz_and_reports_the_source():
    dot = segmentation_utils.segmentation_architecture_dot("retail", "kaggle")
    assert dot.strip().startswith("digraph")
    assert "Online Retail II" in dot and "Customer Personality" in dot
    # A synthetic fallback must be visible in the diagram, not painted over.
    fallback = segmentation_utils.segmentation_architecture_dot("synthetic", "synthetic")
    assert "SYNTHETIC" in fallback


def test_clv_multiplier_matches_the_lecture_formula():
    # Rr / (1 - Rr + d); at Rr=0.75, d=0.10 -> 0.75 / 0.35
    assert segmentation_utils.clv_multiplier(0.75, 0.10) == pytest.approx(0.75 / 0.35)
    # Convex and increasing in retention: the same +10 points is worth far more
    # at the top of the range than the bottom.
    low_step = segmentation_utils.clv_multiplier(0.6, 0.1) - segmentation_utils.clv_multiplier(0.5, 0.1)
    high_step = segmentation_utils.clv_multiplier(0.9, 0.1) - segmentation_utils.clv_multiplier(0.8, 0.1)
    assert high_step > low_step > 0


def test_sensitivity_frame_rises_with_retention_and_falls_with_discount():
    frame = segmentation_utils.sensitivity_frame([0.5, 0.7, 0.9], [0.05, 0.10, 0.20])
    assert list(frame.index) == ["Rr=50%", "Rr=70%", "Rr=90%"]
    for column in frame.columns:
        assert frame[column].is_monotonic_increasing
    for _, row in frame.iterrows():
        assert row.is_monotonic_decreasing


def test_equity_curve_grows_with_retention():
    gc = pd.Series([100.0, 200.0, 300.0])
    curve = segmentation_utils.equity_curve(gc, np.array([0.5, 0.7, 0.9]), 0.10, 10.0)
    assert curve["CustomerEquity"].is_monotonic_increasing
    # Equity = total GC * multiplier - AC per customer.
    expected = 600 * segmentation_utils.clv_multiplier(0.7, 0.10) - 3 * 10.0
    assert curve.loc[1, "CustomerEquity"] == pytest.approx(expected)


def test_annual_contribution_survives_zero_tenure():
    # A customer who ordered once has a tenure of exactly 0 days; without the
    # floor this is a division by zero dressed up as a business metric.
    customers = pd.DataFrame({"Revenue": [500.0, 500.0], "TenureDays": [0, 730]})
    gc = segmentation_utils.annual_contribution(customers, gross_margin=0.30)
    assert np.isfinite(gc).all()
    # The one-off buyer is credited with one year, so their GC is the smaller of
    # the two per-year figures only because the other had twice as long.
    assert gc.iloc[0] == pytest.approx(500 * 0.30 / (365 / 365.25), rel=1e-3)
    assert gc.iloc[1] < gc.iloc[0]


def test_cohort_frames_keep_unobserved_periods_missing():
    matrix = pd.DataFrame(
        {0: [1.0, 1.0], 1: [0.5, 0.4], 2: [0.3, np.nan]},
        index=pd.Index(["2010-01", "2010-02"], name="Cohort"),
    )
    heat = segmentation_utils.cohort_matrix_frame(matrix)
    assert (heat[0] == 100.0).all()
    # The unobserved cell must stay NaN — painting it as 0 would invent a
    # collapse in retention that never happened.
    assert np.isnan(heat.loc["2010-02", 2])

    curve = segmentation_utils.retention_curve_frame(matrix)
    assert 0 not in set(curve["Period"])  # period 0 is 100% by construction
    assert "average" in set(curve["Cohort"])


def test_k_sweep_frame_puts_every_metric_on_an_up_is_better_scale():
    sweep = pd.DataFrame(
        {
            "k": [2, 3, 4],
            "silhouette": [0.3, 0.2, 0.1],
            "calinski_harabasz": [300.0, 200.0, 100.0],
            "davies_bouldin": [1.0, 1.5, 2.0],  # lower is better
            "inertia": [300.0, 200.0, 100.0],  # lower is better
        }
    )
    frame = segmentation_utils.k_sweep_frame(sweep)
    for column in ["silhouette", "calinski_harabasz", "davies_bouldin", "inertia"]:
        assert frame[column].between(0, 1).all()
    # Davies-Bouldin is flipped, so its best (lowest raw) value scores 1.
    assert frame.loc[0, "davies_bouldin"] == pytest.approx(1.0)
    assert frame.loc[2, "davies_bouldin"] == pytest.approx(0.0)


def test_value_tiers_and_action_table():
    clv = pd.Series(range(100), dtype=float)
    tiers = segmentation_utils.value_tiers(clv)
    assert list(tiers.cat.categories) == ["Low Value", "Medium Value", "High Value", "Premium"]
    assert tiers.value_counts().nunique() == 1  # quartiles are equal-sized

    cross_tab = pd.DataFrame(
        {"Premium": [12, 0], "Low Value": [0, 30]}, index=["At Risk", "Lost"]
    )
    cross_tab.index.name = "Segment"
    actions = segmentation_utils.segment_action_table(cross_tab)
    assert set(actions["RFM segment"]) == {"At Risk", "Lost"}
    # Highest-count cell first, and the urgent group is described as urgent.
    assert actions.iloc[0]["Customers"] == 30
    urgent = actions[actions["RFM segment"] == "At Risk"].iloc[0]["Recommended action"]
    assert "Urgent" in urgent
