"""
compute_utils.py
----------------
Importable, picklable helpers for the Parallelization teaching page.

Refactored from ``parallelization/parallel/parallel.py`` (a percent-cell
notebook) into functions that can be imported and reused. ``expensive_row_op``
lives at module top level so ``multiprocessing`` can pickle it when dispatching
work to worker processes.

Nothing in this module runs at import time; the demo is guarded under
``if __name__ == "__main__":``.
"""

from time import time, sleep
import warnings
from multiprocessing import cpu_count, get_context

import numpy as np


def expensive_row_op(row):
    """Simulate a heavy per-row computation, then return the row mean.

    Must stay at module top level so it is picklable for multiprocessing.
    """
    sleep(0.00001)  # Simulate work (0.01ms per row)
    return np.mean(row)


def run_serial(data):
    """Apply ``expensive_row_op`` to each row via a list comprehension.

    Returns ``(results, elapsed_seconds)``.
    """
    time_start = time()
    results = [expensive_row_op(row) for row in data]
    elapsed = time() - time_start
    return results, elapsed


def fork_pool(processes: int | None = None, **kwargs):
    """A process pool that starts workers by **fork** — on purpose, and explicitly.

    Streamlit runs each page as Python's ``__main__`` module. The "spawn" and
    "forkserver" start methods re-execute ``__main__`` in every worker, which
    means re-running the whole *page* (data loading, model training, even a new
    pool) in each worker — the app hangs. ``fork`` copies the running process
    instead and never re-runs ``__main__``. It is requested explicitly because
    Python 3.14 makes "forkserver" the default on Linux.

    Python warns that forking a multi-threaded process (Streamlit's server is
    one) *can* deadlock if another thread holds a lock at that instant. For a
    short-lived worker pool in a single-user demo that risk is far smaller than
    the certain hang above, so that one warning is silenced here — nowhere else.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*use of fork\\(\\) may lead to deadlocks.*",
                                category=DeprecationWarning)
        return get_context("fork").Pool(processes, **kwargs)


def run_parallel(data, n_cpu: int = 4):
    """Apply ``expensive_row_op`` across a ``multiprocessing.Pool``.

    Uses ``min(n_cpu, cpu_count())`` worker processes. If multiprocessing
    fails on this platform, falls back to serial timing rather than crashing.
    Returns ``(results, elapsed_seconds)``.
    """
    n_workers = min(n_cpu, cpu_count())
    try:
        time_start = time()
        with fork_pool(n_workers) as pool:
            results = pool.map(expensive_row_op, data)
        elapsed = time() - time_start
        return results, elapsed
    except Exception:
        # Multiprocessing unavailable (e.g. restricted platform): time serially.
        return run_serial(data)


def run_vectorized(data):
    """Compute all row means at once with numpy's optimized C code.

    Returns ``(result, elapsed_seconds)``.
    """
    time_start = time()
    result = np.mean(data, axis=1)
    elapsed = time() - time_start
    return result, elapsed


def run_benchmark(rows: int, cols: int = 10) -> dict:
    """Generate random data and benchmark all three approaches.

    Returns a dict mapping ``"Serial"``, ``"Parallel"`` and ``"Vectorized"``
    to their elapsed times in seconds.
    """
    data = np.random.rand(rows, cols)
    _, serial_time = run_serial(data)
    _, parallel_time = run_parallel(data)
    _, vectorized_time = run_vectorized(data)
    return {
        "Serial": float(serial_time),
        "Parallel": float(parallel_time),
        "Vectorized": float(vectorized_time),
    }


def benchmark_reading(timings: dict, n_cpu: int = 4) -> str:
    """The race result in plain words — what the three bars mean.

    A learner sees numbers; this turns them into the two things worth saying:
    cores help, but not ``cores``-fold (starting processes and moving data costs
    time), and one library call beats both because the loop runs in compiled code.
    """
    workers = min(n_cpu, cpu_count())
    parallel = timings["Serial"] / timings["Parallel"]
    vectorized = timings["Serial"] / max(timings["Vectorized"], 1e-9)
    if parallel > 1.1:
        cores = (
            f"**Parallel** used {workers} cores and was **{parallel:.1f}× faster** than the plain loop — "
            f"not {workers}×, because starting the workers and handing them the data takes time too."
        )
    else:
        cores = (
            f"**Parallel** used {workers} cores and was **not faster** here: for this little data, "
            "starting the workers costs more than they save. Try more rows."
        )
    return (
        f"{cores}\n\n**Vectorized** was **{vectorized:,.0f}× faster** on a single core. Same result, "
        "one library call instead of a loop — so before reaching for more cores, check whether "
        "your library can already do the whole job at once."
    )


if __name__ == "__main__":
    print(run_benchmark(2000))
