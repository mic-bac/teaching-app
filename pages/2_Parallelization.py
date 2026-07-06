from pathlib import Path

import pandas as pd
import streamlit as st

from utils.compute_utils import run_benchmark

st.set_page_config(page_title="Parallelization", layout="wide", page_icon="⚡")

st.title("⚡ Parallelization: Serial vs. Parallel vs. Vectorized")

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

Pick a dataset size and start the race to see how they compare.
"""
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
    st.bar_chart(chart_df)

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

st.divider()

st.header("Python vs R — side by side")
st.markdown(
    "The same benchmark, written in Python and in R. The R script is shown for "
    "comparison only and is never executed here."
)

ROOT = Path(__file__).resolve().parents[1]
py_path = ROOT / "parallelization" / "parallel" / "parallel.py"
r_path = ROOT / "parallelization" / "parallel" / "parallel.R"

col_py, col_r = st.columns(2)

with col_py:
    st.subheader("Python")
    try:
        py_source = py_path.read_text()
        st.code(py_source, language="python")
    except OSError:
        st.warning(f"Could not read the Python source file at `{py_path}`.")

with col_r:
    st.subheader("R")
    try:
        r_source = r_path.read_text()
        st.code(r_source, language="r")
    except OSError:
        st.warning(f"Could not read the R source file at `{r_path}`.")
