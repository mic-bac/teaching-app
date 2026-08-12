"""Smoke tests that actually run the Streamlit app and assert no errors.

Uses Streamlit's official headless testing framework
(``streamlit.testing.v1.AppTest``): each script is executed exactly as Streamlit
would run it, and ``at.exception`` captures any uncaught error. If a page raises,
these tests fail — which is the "verify there are no errors" guarantee.

The DB page connects to PostgreSQL if the Docker container is up, otherwise it
transparently uses the SQLite fallback; either way the page must load cleanly.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]

PAGES = [
    "Home.py",
    "pages/1_Database_Basics.py",
    "pages/2_Parallelization.py",
    "pages/3_Recommender.py",
    "pages/4_Propensity.py",
    "pages/5_Survival.py",
    "pages/6_Segmentation.py",
]


def _run(script: str) -> AppTest:
    at = AppTest.from_file(str(ROOT / script), default_timeout=60)
    at.run()
    return at


@pytest.mark.parametrize("script", PAGES)
def test_page_loads_without_exception(script):
    at = _run(script)
    assert not at.exception, f"{script} raised: {[str(e) for e in at.exception]}"


def test_home_has_title_and_links():
    at = _run("Home.py")
    assert not at.exception
    # Landing page renders a title and the two navigation cards.
    assert any("Teaching App" in t.value for t in at.title)


def test_db_page_shows_backend_and_table():
    at = _run("pages/1_Database_Basics.py")
    assert not at.exception
    # Either a success (PostgreSQL) or warning (SQLite fallback) banner is shown.
    assert at.success or at.warning
    # Schema + live sales table are rendered (across the tabs).
    assert len(at.dataframe) >= 1


def test_db_page_sql_playground_runs():
    at = _run("pages/1_Database_Basics.py")
    assert not at.exception
    run_buttons = [b for b in at.button if b.label == "▶️ Run query"]
    assert run_buttons, "SQL playground run button not found"
    run_buttons[0].click().run()
    assert not at.exception, f"SQL playground raised: {[str(e) for e in at.exception]}"


def test_parallel_page_benchmark_runs():
    at = AppTest.from_file(str(ROOT / "pages/2_Parallelization.py"), default_timeout=120)
    at.run()
    assert not at.exception
    # Tabs mean several sliders/buttons exist, so select by label rather than index.
    cpu_sliders = [s for s in at.slider if s.label == "Dataset size (rows)"]
    assert cpu_sliders, "CPU benchmark slider not found"
    cpu_sliders[0].set_value(1000).run()
    race_buttons = [b for b in at.button if b.label == "🏁 Start Benchmark Race"]
    assert race_buttons, "CPU benchmark button not found"
    race_buttons[0].click().run()
    assert not at.exception, f"benchmark raised: {[str(e) for e in at.exception]}"
    # After running, the results include per-approach metrics.
    assert len(at.metric) >= 3


def test_parallel_page_io_race_runs():
    at = AppTest.from_file(str(ROOT / "pages/2_Parallelization.py"), default_timeout=120)
    at.run()
    assert not at.exception
    # Keep it cheap and offline-safe: a handful of requests to an unreachable host.
    # download_site swallows per-request errors, so the race must not raise.
    urls = [i for i in at.text_input if i.label == "URL to download"]
    assert urls, "I/O URL input not found"
    urls[0].set_value("http://127.0.0.1:1/").run()
    counts = [s for s in at.slider if s.label == "Number of requests"]
    assert counts, "I/O request-count slider not found"
    counts[0].set_value(10).run()
    io_buttons = [b for b in at.button if b.label == "🏁 Start Download Race"]
    assert io_buttons, "I/O race button not found"
    io_buttons[0].click().run()
    assert not at.exception, f"I/O race raised: {[str(e) for e in at.exception]}"
    # Four approaches → four metrics in the I/O results.
    assert len(at.metric) >= 4


def test_recommender_page_content_based_selectbox():
    at = AppTest.from_file(str(ROOT / "pages/3_Recommender.py"), default_timeout=180)
    at.run()
    assert not at.exception, f"recommender load raised: {[str(e) for e in at.exception]}"
    # Pick a movie in the Content-Based tab and confirm it still renders cleanly.
    movie_boxes = [s for s in at.selectbox if s.label == "Pick a movie you liked"]
    assert movie_boxes, "content-based movie selectbox not found"
    movie_boxes[0].set_value("Jumanji (1995)").run()
    assert not at.exception
    assert len(at.dataframe) >= 1  # a recommendations table is shown


def test_recommender_page_matrix_factorization_trains():
    at = AppTest.from_file(str(ROOT / "pages/3_Recommender.py"), default_timeout=180)
    at.run()
    assert not at.exception
    # Train the MF model at the cheapest setting (min epochs) and confirm it
    # completes and reports a test-RMSE metric.
    epoch_sliders = [s for s in at.slider if s.label == "Training epochs"]
    assert epoch_sliders, "MF epochs slider not found"
    epoch_sliders[0].set_value(5).run()
    train_buttons = [b for b in at.button if b.label == "🚀 Train the model"]
    assert train_buttons, "MF train button not found"
    train_buttons[0].click().run()
    assert not at.exception, f"MF training raised: {[str(e) for e in at.exception]}"
    assert any("MF test RMSE" in m.label for m in at.metric)


def test_propensity_page_fits_a_model_at_the_smallest_sample():
    at = AppTest.from_file(str(ROOT / "pages/4_Propensity.py"), default_timeout=180)
    at.run()
    assert not at.exception, f"propensity load raised: {[str(e) for e in at.exception]}"
    # Shrink the sample first so every refit below stays cheap.
    sample = [s for s in at.select_slider if s.label == "Customers to model"]
    assert sample, "sample-size control not found"
    sample[0].set_value(2000).run()
    assert not at.exception
    # The Models tab fits on load, so the metric row must already be there.
    assert any(m.label == "ROC-AUC" for m in at.metric)
    # The page states which data it is running on: the real CSVs or the fallback.
    banners = list(at.success) + list(at.warning)
    assert any("Kaggle churn dataset" in b.value or "synthetic" in b.value for b in banners)


def test_propensity_page_cross_validation_and_tuning_run():
    at = AppTest.from_file(str(ROOT / "pages/4_Propensity.py"), default_timeout=300)
    at.run()
    sample = [s for s in at.select_slider if s.label == "Customers to model"]
    sample[0].set_value(2000).run()

    folds = [s for s in at.slider if s.label == "Folds (k)"]
    assert folds, "CV fold slider not found"
    folds[0].set_value(3).run()
    cv_buttons = [b for b in at.button if b.label == "🔁 Run cross-validation"]
    assert cv_buttons, "cross-validation button not found"
    cv_buttons[0].click().run()
    assert not at.exception, f"cross-validation raised: {[str(e) for e in at.exception]}"
    assert any(m.label == "Mean ROC-AUC" for m in at.metric)

    search_buttons = [b for b in at.button if b.label == "🎛️ Search"]
    assert search_buttons, "tuning button not found"
    search_buttons[0].click().run()
    assert not at.exception, f"hyperparameter search raised: {[str(e) for e in at.exception]}"
    assert any(m.label == "Model fits spent" for m in at.metric)


def test_propensity_page_draws_learning_and_validation_curves():
    at = AppTest.from_file(str(ROOT / "pages/4_Propensity.py"), default_timeout=300)
    at.run()
    at.select_slider[0].set_value(2000).run()
    curve_buttons = [b for b in at.button if b.label == "📈 Draw both curves"]
    assert curve_buttons, "curve button not found"
    curve_buttons[0].click().run()
    assert not at.exception, f"curves raised: {[str(e) for e in at.exception]}"
    assert any(m.label == "Gap" for m in at.metric)


def test_survival_page_shows_kaplan_meier_readouts():
    at = AppTest.from_file(str(ROOT / "pages/5_Survival.py"), default_timeout=180)
    at.run()
    assert not at.exception, f"survival load raised: {[str(e) for e in at.exception]}"
    sample = [s for s in at.select_slider if s.label == "Customers to model"]
    assert sample, "sample-size control not found"
    sample[0].set_value(1000).run()
    assert not at.exception
    # S(t) readouts are rendered straight from the Kaplan-Meier curve.
    assert any(m.label.startswith("S(12") for m in at.metric)


def test_survival_page_trains_models_and_scores_them_over_time():
    at = AppTest.from_file(str(ROOT / "pages/5_Survival.py"), default_timeout=300)
    at.run()
    at.select_slider[0].set_value(1000).run()

    train_buttons = [b for b in at.button if b.label == "🏋️ Train all three models"]
    assert train_buttons, "survival train button not found"
    train_buttons[0].click().run()
    assert not at.exception, f"survival training raised: {[str(e) for e in at.exception]}"

    score_buttons = [b for b in at.button if b.label == "📏 Score all three models over time"]
    assert score_buttons, "survival scoring button not found"
    score_buttons[0].click().run()
    assert not at.exception, f"survival scoring raised: {[str(e) for e in at.exception]}"
    # The ranking-only model must explain itself rather than invent a Brier score.
    assert any("Survival SVM" in w.value for w in at.warning)


def test_survival_page_score_only_model_has_no_curves():
    at = AppTest.from_file(str(ROOT / "pages/5_Survival.py"), default_timeout=300)
    at.run()
    at.select_slider[0].set_value(1000).run()
    model_boxes = [s for s in at.selectbox if s.label == "Model"]
    assert model_boxes, "survival model selectbox not found"
    model_boxes[0].set_value("Survival SVM").run()
    assert not at.exception, f"survival SVM selection raised: {[str(e) for e in at.exception]}"
    assert any("no survival curve" in w.value for w in at.warning)


def test_segmentation_page_reacts_to_the_retention_slider():
    """The CLTV tab's headline interaction must not raise — and must not refit.

    Dragging the retention rate recomputes customer equity from the cached
    per-customer contribution only; if this ever starts reloading the 640,000-row
    transaction log, the page becomes unusable in a live demo.
    """
    at = _run("pages/6_Segmentation.py")
    assert not at.exception
    # 4 CLTV assumption sliders + the clustering k slider.
    assert len(at.slider) >= 5
    retention = at.slider[1]
    at = retention.set_value(0.85).run()
    assert not at.exception, [str(e) for e in at.exception]
    # Customer equity is reported as a metric and must survive the change.
    assert any("equity" in m.label.lower() for m in at.metric)
