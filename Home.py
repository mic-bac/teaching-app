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
    The tools behind *"customers also bought"*, churn warnings, customer segments and sales
    forecasts are not magic — and you can use them yourself. Every lesson here runs **real
    code on real data**: move a slider, press a button, and watch what happens. No maths
    degree required; curiosity is enough.

    - 🗄️ **Database Basics** — working with PostgreSQL running in Docker.
    - ⚡ **Parallelization** — comparing Serial vs Parallel vs Vectorized computation.
    - 🎯 **Recommender Systems** — content-based, collaborative, and association-rule recommenders.
    - 🔮 **Propensity Models** — look-alike modeling: how likely is a customer to churn?
    - ⏳ **Survival Analysis** — time-to-event: *when* will they churn, not just whether.
    - 🧩 **Customer Segmentation** — RFM, customer lifetime value and clustering.
    - 📈 **Time Series** — is a series forecastable at all, and what classical models make of it.
    - 🔮 **Forecasting** — Prophet and XGBoost: when they are worth it, and when a simple forecast wins.

    Pick a lesson below to get started — each one ends with **🎯 What to remember**.
    """
)

st.divider()

st.subheader("Choose a lesson")

col1, col2, col3 = st.columns(3)

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

with col3:
    with st.container(border=True):
        st.markdown("### 🎯 Recommender Systems")
        st.write(
            "Explore how Netflix- and Amazon-style recommenders work: content-based "
            "filtering, collaborative filtering, and market-basket association rules."
        )
        st.page_link(
            "pages/3_Recommender.py",
            label="Open Recommender Systems",
            icon="🎯",
        )

col4, col5, col6 = st.columns(3)

with col4:
    with st.container(border=True):
        st.markdown("### 🔮 Propensity Models")
        st.write(
            "Look-alike modeling: score every customer's probability of churning, "
            "compare three model families, tune them, and cut the scores into risk segments."
        )
        st.page_link(
            "pages/4_Propensity.py",
            label="Open Propensity Models",
            icon="🔮",
        )

with col5:
    with st.container(border=True):
        st.markdown("### ⏳ Survival Analysis")
        st.write(
            "Time-to-event modeling: Kaplan-Meier curves, censored data, Cox and "
            "survival forests — predicting *when* a customer leaves, not just whether."
        )
        st.page_link(
            "pages/5_Survival.py",
            label="Open Survival Analysis",
            icon="⏳",
        )

with col6:
    with st.container(border=True):
        st.markdown("### 🧩 Customer Segmentation")
        st.write(
            "RFM scoring, customer lifetime value from measured cohort retention, and "
            "clustering — who your customers are, and what each of them is worth."
        )
        st.page_link(
            "pages/6_Segmentation.py",
            label="Open Customer Segmentation",
            icon="🧩",
        )

col7, col8, col9 = st.columns(3)

with col7:
    with st.container(border=True):
        st.markdown("### 📈 Time Series")
        st.write(
            "Is this series forecastable at all? Decomposition, stationarity, the four error "
            "measures, baselines, Holt-Winters and SARIMA — and why Christmas cannot be tested."
        )
        st.page_link(
            "pages/7_Time_Series.py",
            label="Open Time Series",
            icon="📈",
        )

with col8:
    with st.container(border=True):
        st.markdown("### 🔮 Forecasting")
        st.write(
            "Prophet's explainable forecasts and XGBoost's forecasting at scale — and why the "
            "winner depends on the level of detail you forecast at."
        )
        st.page_link(
            "pages/8_Forecasting.py",
            label="Open Forecasting",
            icon="🔮",
        )

st.divider()

with st.expander("📊 About this dataset"):
    st.write(
        "Several lessons use a randomly generated dataset. Below is a small "
        "1,000-row sample produced by the cached `generate_dataset` helper."
    )
    sample = generate_dataset(1000)
    st.dataframe(sample.head(), width="stretch")
    st.caption(f"Sample shape: {sample.shape[0]} rows × {sample.shape[1]} columns")
