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
from multiprocessing import Pool, cpu_count

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


def run_parallel(data, n_cpu: int = 4):
    """Apply ``expensive_row_op`` across a ``multiprocessing.Pool``.

    Uses ``min(n_cpu, cpu_count())`` worker processes. If multiprocessing
    fails on this platform, falls back to serial timing rather than crashing.
    Returns ``(results, elapsed_seconds)``.
    """
    n_workers = min(n_cpu, cpu_count())
    try:
        time_start = time()
        with Pool(n_workers) as pool:
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


if __name__ == "__main__":
    print(run_benchmark(2000))
