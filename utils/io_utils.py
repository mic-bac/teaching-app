"""
io_utils.py
-----------
Importable, UI-free helpers for the **I/O-bound concurrency** half of the
Parallelization teaching page. This is the counterpart to ``compute_utils.py``:
that module speeds up **CPU-bound** work with more cores, this one speeds up
**I/O-bound** work (waiting on the network) by overlapping the waiting.

Refactored from ``parallelization/parallel/io_concurrency.py`` (a percent-cell
notebook), itself adapted from Real Python's "Speed Up Your Python Program With
Concurrency" (https://realpython.com/python-concurrency/).

Every download goes through ``download_site``/``download_site_async``, which
swallow per-request errors and return ``0`` bytes on failure. That is deliberate:
the app does *real* downloads live, and a flaky network or a stray URL must never
crash the demo — it should just show a slow or zero-byte result. Worker functions
for multiprocessing live at module top level so they can be pickled, exactly like
``compute_utils.expensive_row_op``.

Nothing runs at import time; the demo is guarded under ``if __name__ ==
"__main__":``.
"""

import asyncio
import concurrent.futures
import threading
from multiprocessing import Pool
from time import time

import aiohttp
import requests

# A small, lightweight default endpoint for the page's URL box. Any URL works;
# the point is the network round-trip we get to overlap.
DEFAULT_URL = "http://olympus.realpython.org/dice"

# Per-request timeout (seconds) so one slow/dead host can't hang the whole race.
REQUEST_TIMEOUT = 10


def download_site(session, url):
    """Fetch ``url`` with a ``requests`` session; return the number of bytes read.

    Wrapped in ``try/except`` so a failed request returns ``0`` instead of raising
    — graceful degradation keeps a live demo alive when the network misbehaves.
    """
    try:
        with session.get(url, timeout=REQUEST_TIMEOUT) as response:
            return len(response.content)
    except Exception:
        return 0


def run_sequential(sites):
    """Download every site one at a time over a single reused session.

    The baseline: total time is the *sum* of all the network waits, because each
    request blocks the next. Returns ``(bytes_per_site, elapsed_seconds)``.
    """
    time_start = time()
    with requests.Session() as session:
        results = [download_site(session, url) for url in sites]
    elapsed = time() - time_start
    return results, elapsed


# --- Threading ------------------------------------------------------------
# ``requests.Session`` is not thread-safe, so each worker thread gets its own
# via ``threading.local()``.
_thread_local = threading.local()


def _get_thread_session():
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


def _download_on_thread(url):
    return download_site(_get_thread_session(), url)


def run_threaded(sites, max_workers: int = 5):
    """Download across a ``ThreadPoolExecutor`` so the network waits overlap.

    Python releases the GIL while a thread waits on I/O, so several downloads are
    in flight at once. Returns ``(bytes_per_site, elapsed_seconds)``.
    """
    time_start = time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_download_on_thread, sites))
    elapsed = time() - time_start
    return results, elapsed


# --- Asyncio --------------------------------------------------------------
async def _download_site_async(session, url):
    try:
        async with session.get(url, timeout=REQUEST_TIMEOUT) as response:
            return len(await response.read())
    except Exception:
        return 0


async def _download_all_async(sites):
    async with aiohttp.ClientSession() as session:
        tasks = [_download_site_async(session, url) for url in sites]
        return await asyncio.gather(*tasks)


def run_async(sites):
    """Download via a single-threaded asyncio event loop (``aiohttp``).

    Each ``await`` hands control back to the loop while it waits, so one thread
    juggles all the in-flight requests with very little per-task overhead. Falls
    back to sequential timing if the event loop can't be created. Returns
    ``(bytes_per_site, elapsed_seconds)``.
    """
    time_start = time()
    try:
        results = asyncio.run(_download_all_async(sites))
    except Exception:
        # asyncio/aiohttp unavailable or a loop is already running: don't crash.
        return run_sequential(sites)
    elapsed = time() - time_start
    return results, elapsed


# --- Multiprocessing (the wrong tool for I/O — shown for contrast) --------
_mp_session = None


def _set_mp_session():
    global _mp_session
    if _mp_session is None:
        _mp_session = requests.Session()


def _download_on_process(url):
    return download_site(_mp_session, url)


def run_multiprocess(sites):
    """Download across a process ``Pool`` — CPU-bound's hero, here for contrast.

    It overlaps the downloads too, but process spawn + IPC overhead makes it lose
    to threads and asyncio when the real cost is waiting. Falls back to sequential
    if multiprocessing is unavailable. Returns ``(bytes_per_site, elapsed_seconds)``.
    """
    try:
        time_start = time()
        with Pool(initializer=_set_mp_session) as pool:
            results = pool.map(_download_on_process, sites)
        elapsed = time() - time_start
        return results, elapsed
    except Exception:
        return run_sequential(sites)


def run_io_benchmark(url: str = DEFAULT_URL, count: int = 80) -> dict:
    """Download ``url`` ``count`` times four ways and time each approach.

    Returns a dict mapping ``"Sequential"``, ``"Threaded"``, ``"Async"`` and
    ``"Multiprocessing"`` to their elapsed times in seconds.
    """
    sites = [url] * count
    _, sequential_time = run_sequential(sites)
    _, threaded_time = run_threaded(sites)
    _, async_time = run_async(sites)
    _, multiprocess_time = run_multiprocess(sites)
    return {
        "Sequential": float(sequential_time),
        "Threaded": float(threaded_time),
        "Async": float(async_time),
        "Multiprocessing": float(multiprocess_time),
    }


if __name__ == "__main__":
    print(run_io_benchmark(count=40))
