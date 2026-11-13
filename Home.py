"""
Home.py
-------
Landing page for the teaching app. Run with:

    streamlit run Home.py
"""

import streamlit as st


st.set_page_config(page_title="Teaching App", layout="wide", page_icon="🎓")

st.title("🎓 Welcome to the Teaching App")
st.markdown(
    """
    The tools behind *"customers also bought"*, churn warnings, customer segments, sales
    forecasts and photo recognition are not magic — and you can use them yourself. Every lesson here runs **real
    code on real data**: move a slider, press a button, and watch what happens. No maths
    degree required; curiosity is enough.

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
            "Where does a company's data live, and how do you get your own numbers out of it? Ask a real database a question in a few lines of SQL."
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
            "Why does this report take an hour? Race the same job three ways and see which trick makes it fast — and which one only looks clever."
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
            "How do Netflix and Amazon know what to suggest? Three approaches on real ratings and receipts: similar items, similar people, bought together."
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
            "Which customers are about to leave? Give every customer a churn risk score, compare three models, and decide who gets the retention offer."
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
            "Not just *whether* a customer leaves, but *when*. Read retention curves and see what speeds up or delays the goodbye."
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
            "Who are your best customers, and what is each one worth? Sort them by buying behaviour, estimate lifetime value, and find natural groups."
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
            "Can next quarter's sales be forecast at all? Check the data first, then try the classic methods — and learn why Christmas is hard to test."
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
            "Are the famous tools (Prophet, XGBoost) worth it? Sometimes — see when they win, and when \"same week last year\" is hard to beat."
        )
        st.page_link(
            "pages/8_Forecasting.py",
            label="Open Forecasting",
            icon="🔮",
        )

with col9:
    with st.container(border=True):
        st.markdown("### 🍝 Object Detection")
        st.write(
            "How does a computer find things in a photo? See what a free model detects, and how 50 labelled pictures teach it something new."
        )
        st.page_link(
            "pages/9_Object_Detection.py",
            label="Open Object Detection",
            icon="🍝",
        )
