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
    # The live sales table is rendered.
    assert len(at.dataframe) >= 1


def test_parallel_page_benchmark_runs():
    at = AppTest.from_file(str(ROOT / "pages/2_Parallelization.py"), default_timeout=120)
    at.run()
    assert not at.exception
    # Keep the run cheap: smallest dataset size, then click the race button.
    at.slider[0].set_value(1000).run()
    at.button[0].click().run()
    assert not at.exception, f"benchmark raised: {[str(e) for e in at.exception]}"
    # After running, the results include per-approach metrics.
    assert len(at.metric) >= 3
