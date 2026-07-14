from pathlib import Path

import pandas as pd
import streamlit as st

from utils import io_utils
from utils.compute_utils import run_benchmark
from utils.data_generator import generate_dataset
from utils.teaching import source_of, split_script_blocks

st.set_page_config(page_title="Parallelization", layout="wide", page_icon="⚡")

ROOT = Path(__file__).resolve().parents[1]

st.title("⚡ Two Ways to Go Faster: Parallelism vs. Concurrency")

st.markdown(
    """
There are **two very different reasons** a program is slow, and each has its own
cure:

- **CPU-bound** work is slow because it's *computing* a lot. The fix is more
  **cores** working at once — *parallelism* (`multiprocessing`, vectorization).
- **I/O-bound** work is slow because it's *waiting* — on the network, a disk, a
  database. Adding cores barely helps; instead we **overlap the waiting** —
  *concurrency* (threads, `asyncio`).

Each tab below races the approaches for one kind of workload.
"""
)

cpu_tab, io_tab = st.tabs(
    ["🧮 CPU-bound — Parallelism", "🌐 I/O-bound — Concurrency"]
)

# ===========================================================================
# CPU-BOUND TAB
# ===========================================================================
with cpu_tab:
    st.header("Serial vs. Parallel vs. Vectorized")

    st.markdown(
        """
Processing a large dataset row by row can be done in very different ways, and
the speed difference can be enormous:

- **Serial** — a plain Python loop processes one row at a time. Simple, but the
  slowest: every row waits for the previous one to finish.
- **Parallel** — `multiprocessing.Pool` splits the rows across several CPU
  cores that work at the same time. Faster for genuinely heavy per-row work,
  though spawning processes and shuffling data between them adds overhead.
- **Vectorized** — NumPy applies the operation to the whole array at once in
  optimized C code, with no Python-level loop. Usually the fastest by far for
  built-in operations like `mean`.

First look at the data and the code below, then start the race at the bottom to
see how they compare.
"""
    )

    st.subheader("What the data looks like")
    st.markdown(
        "Every run works on a table of random numbers. Each approach computes the "
        "**mean of each row** — a tiny operation repeated across all the rows, which "
        "is exactly the kind of workload that parallelization speeds up."
    )
    st.dataframe(generate_dataset(5), use_container_width=True)
    st.caption(
        "First 5 rows, generated on the fly. The real `parallel_big_data.csv` (184 MB, "
        "1 M rows) has the same shape — random floats in [0, 1) across 10 columns."
    )

    st.divider()

    st.subheader("Python vs R — the same steps, side by side")
    st.markdown(
        "The same benchmark in both languages, broken into matching steps so you can "
        "compare them block by block. The R script is shown for comparison only and is "
        "never executed here."
    )

    # A short, comparison-friendly label per shared step number (see the
    # `# --- N. ... ---` markers in both scripts).
    STEP_LABELS = {
        1: "1 · Imports",
        2: "2 · Load the dataset",
        3: "3 · Define a slow row operation",
        4: "4 · Convert to array / matrix",
        5: "5 · Serial — plain loop / apply",
        6: "6 · Parallel — multiprocessing / mclapply",
        7: "7 · Vectorized — NumPy / rowMeans",
    }

    py_path = ROOT / "parallelization" / "parallel" / "parallel.py"
    r_path = ROOT / "parallelization" / "parallel" / "parallel.R"

    try:
        py_source = py_path.read_text()
        r_source = r_path.read_text()
    except OSError as exc:
        py_source = r_source = None
        st.warning(f"Could not read the source scripts: {exc}")

    py_blocks = {b["number"]: b for b in split_script_blocks(py_source or "")}
    r_blocks = {b["number"]: b for b in split_script_blocks(r_source or "")}
    shared = sorted(set(py_blocks) & set(r_blocks))

    if shared:
        head_py, head_r = st.columns(2)
        head_py.markdown("### 🐍 Python")
        head_r.markdown("### 📊 R")

        for n in shared:
            with st.container(border=True):
                st.markdown(f"**{STEP_LABELS.get(n, py_blocks[n]['title'])}**")
                code_py, code_r = st.columns(2)
                code_py.code(py_blocks[n]["code"], language="python")
                code_r.code(r_blocks[n]["code"], language="r")
    elif py_source is not None:
        # Parsing found no shared step markers — fall back to whole files.
        col_py, col_r = st.columns(2)
        col_py.markdown("**🐍 Python**")
        col_py.code(py_source, language="python")
        col_r.markdown("**📊 R**")
        col_r.code(r_source, language="r")

    with st.expander("Show the full, unedited scripts"):
        full_py, full_r = st.columns(2)
        full_py.markdown("**🐍 `parallel.py`**")
        full_py.code(py_source or "", language="python")
        full_r.markdown("**📊 `parallel.R`**")
        full_r.code(r_source or "", language="r")

    st.divider()

    st.subheader("🏁 Race them")
    st.markdown(
        "Now that you've seen the code, pick a dataset size and run all three "
        "approaches on freshly generated data."
    )

    rows = st.slider(
        "Dataset size (rows)",
        min_value=1000,
        max_value=100000,
        value=20000,
        step=1000,
    )

    if st.button("🏁 Start Benchmark Race"):
        with st.spinner(f"Racing three approaches over {rows:,} rows..."):
            timings = run_benchmark(rows)

        st.subheader("Results")

        chart_df = pd.DataFrame(
            {"Seconds": [timings["Serial"], timings["Parallel"], timings["Vectorized"]]},
            index=["Serial", "Parallel", "Vectorized"],
        )
        st.bar_chart(chart_df, sort=False)

        col_s, col_p, col_v = st.columns(3)
        col_s.metric("Serial", f"{timings['Serial']:.4f} s")
        col_p.metric("Parallel", f"{timings['Parallel']:.4f} s")
        col_v.metric("Vectorized", f"{timings['Vectorized']:.4f} s")

        st.dataframe(
            chart_df.style.format({"Seconds": "{:.6f}"}),
            use_container_width=True,
        )
    else:
        st.info("Choose a dataset size above and click **Start Benchmark Race**.")

# ===========================================================================
# I/O-BOUND TAB
# ===========================================================================
with io_tab:
    st.header("Synchronous vs. Threaded vs. Async vs. Multiprocessing")

    st.markdown(
        """
Now the bottleneck is **waiting on the network**, not computing. Adding CPU cores
barely helps here — idle cores can't make a remote server answer any faster. The
trick is to **overlap the waiting**: fire off the next request before the previous
one has come back.

Below are four ways to download the same page many times — a plain loop, threads,
asyncio, and (for contrast) multiprocessing — each explained with the real code
that runs. Scroll to the bottom to race them.

> These are **real downloads**. If the network is slow or unreachable the race
> still runs — failed requests just count as zero bytes.
"""
    )

    # -- The one primitive every approach shares -----------------------------
    with st.container(border=True):
        st.markdown("**The shared building block — one download**")
        st.markdown(
            "All four approaches fetch a URL with this one helper and count the bytes "
            "that come back. Only *how they schedule these calls* differs. It catches "
            "per-request errors and returns `0` on failure, so a flaky network just "
            "slows the race down instead of crashing the demo."
        )
        st.code(source_of(io_utils.download_site), language="python")

    st.subheader("The four approaches, step by step")

    approaches = [
        {
            "title": "1 · Sequential — one request at a time (the baseline)",
            "explanation": (
                "The simplest possible version: loop over the URLs and download them "
                "one after another over a single reused `Session`. Each `session.get()` "
                "**blocks** — the program stops and waits for the entire response before "
                "moving to the next URL — so the total time is the **sum** of every "
                "individual wait. Reusing one `Session` at least keeps the TCP "
                "connection open between requests instead of re-opening it each time. "
                "This is the baseline every other approach tries to beat."
            ),
            "funcs": [io_utils.run_sequential],
        },
        {
            "title": "2 · Threaded — overlap the waiting with a pool of threads",
            "explanation": (
                "A `ThreadPoolExecutor` hands the downloads to a small pool of worker "
                "threads (here `max_workers=5`), so several requests are in flight at "
                "once. The key insight: while a thread is blocked waiting on the network, "
                "Python **releases the GIL**, letting the other threads run. So the "
                "*waiting* overlaps — five requests wait together instead of one after "
                "another — and the wall-clock time drops roughly by the number of "
                "workers. One catch: a `requests.Session` isn't thread-safe, so each "
                "thread gets its own via `threading.local()` (the `_get_thread_session` "
                "helper below)."
            ),
            "funcs": [
                io_utils._get_thread_session,
                io_utils._download_on_thread,
                io_utils.run_threaded,
            ],
        },
        {
            "title": "3 · Async — one thread, hundreds of requests in flight",
            "explanation": (
                "Instead of many threads, asyncio uses **one** thread running an event "
                "loop. Each `await session.get(...)` voluntarily hands control back to "
                "the loop while it waits, so that single thread can keep hundreds of "
                "requests in flight, switching to whichever one is ready. "
                "`asyncio.gather` launches them all and waits for the whole batch. This "
                "needs an async-aware HTTP library — plain `requests` can't be "
                "`await`ed, so we use `aiohttp`. Because a paused coroutine costs far "
                "less than a parked thread, this scales the best of the four and is "
                "usually the fastest."
            ),
            "funcs": [
                io_utils._download_site_async,
                io_utils._download_all_async,
                io_utils.run_async,
            ],
        },
        {
            "title": "4 · Multiprocessing — the wrong tool here (on purpose)",
            "explanation": (
                "This is the champion of the CPU tab, shown here as a deliberate "
                "*counter-example*. A process `Pool` gives each worker its own Python "
                "interpreter and its own `Session`. It does overlap the downloads, but "
                "spinning up processes and shipping results back between them is pure "
                "overhead — and that overhead buys you nothing when the bottleneck is "
                "network waiting, not CPU work. So for I/O it typically lands **behind** "
                "threading and asyncio. The takeaway: match the tool to the bottleneck — "
                "cores for *computing*, overlapped waits for *waiting*."
            ),
            "funcs": [
                io_utils._set_mp_session,
                io_utils._download_on_process,
                io_utils.run_multiprocess,
            ],
        },
    ]

    for approach in approaches:
        with st.container(border=True):
            st.markdown(f"**{approach['title']}**")
            st.markdown(approach["explanation"])
            code = "\n\n".join(source_of(fn) for fn in approach["funcs"])
            st.code(code, language="python")

    with st.expander("Show the full reference notebook (`io_concurrency.py`)"):
        st.markdown(
            "The complete, article-faithful version of all four approaches, kept in the "
            "parallelization repo as the single source of truth and shown here verbatim."
        )
        io_path = ROOT / "parallelization" / "parallel" / "io_concurrency.py"
        try:
            st.code(io_path.read_text(), language="python")
        except OSError:
            st.warning(f"Could not read the reference file at `{io_path}`.")

    st.divider()

    st.subheader("🏁 Race them")
    st.markdown(
        "Now that you've seen the code, pick a URL and how many times to download it, "
        "then run all four approaches back to back."
    )

    url = st.text_input("URL to download", value=io_utils.DEFAULT_URL)
    count = st.slider(
        "Number of requests",
        min_value=10,
        max_value=200,
        value=80,
        step=10,
    )

    if st.button("🏁 Start Download Race"):
        with st.spinner(f"Downloading {url} {count}× four ways..."):
            timings = io_utils.run_io_benchmark(url, count)

        st.subheader("Results")

        order = ["Sequential", "Threaded", "Async", "Multiprocessing"]
        chart_df = pd.DataFrame(
            {"Seconds": [timings[k] for k in order]},
            index=order,
        )
        st.bar_chart(chart_df, sort=False)

        cols = st.columns(4)
        for col, name in zip(cols, order):
            col.metric(name, f"{timings[name]:.3f} s")

        st.dataframe(
            chart_df.style.format({"Seconds": "{:.6f}"}),
            use_container_width=True,
        )
    else:
        st.info("Set a URL and request count above, then click **Start Download Race**.")
