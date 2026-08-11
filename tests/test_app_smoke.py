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
