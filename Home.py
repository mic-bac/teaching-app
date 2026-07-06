"""
Home.py
-------
Landing page for the teaching app. Run with:

    streamlit run Home.py
"""

import streamlit as st

from utils.data_generator import generate_dataset

st.set_page_config(page_title="Teaching App", layout="wide", page_icon="🎓")

st.title("🎓 Welcome to the Teaching App")
st.markdown(
    """
    This is a hands-on **educational app** that brings two lessons together in
    one place. Explore each topic interactively, run the examples, and see the
    concepts come to life:

    - 🗄️ **Database Basics** — working with PostgreSQL running in Docker.
    - ⚡ **Parallelization** — comparing Serial vs Parallel vs Vectorized computation.

    Pick a lesson below to get started.
    """
)

st.divider()

st.subheader("Choose a lesson")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("### 🗄️ Database Basics")
        st.write(
            "Learn the fundamentals of relational databases using PostgreSQL "
            "running inside Docker. Connect, query, and explore real data."
        )
        st.page_link(
            "pages/1_Database_Basics.py",
            label="Open Database Basics",
            icon="🗄️",
        )

with col2:
    with st.container(border=True):
        st.markdown("### ⚡ Parallelization")
        st.write(
            "See how the same computation performs when run Serially, in "
            "Parallel, and Vectorized — and understand why it matters."
        )
        st.page_link(
            "pages/2_Parallelization.py",
            label="Open Parallelization",
            icon="⚡",
        )

st.divider()

with st.expander("📊 About this dataset"):
    st.write(
        "Several lessons use a randomly generated dataset. Below is a small "
        "1,000-row sample produced by the cached `generate_dataset` helper."
    )
    sample = generate_dataset(1000)
    st.dataframe(sample.head(), use_container_width=True)
    st.caption(f"Sample shape: {sample.shape[0]} rows × {sample.shape[1]} columns")
