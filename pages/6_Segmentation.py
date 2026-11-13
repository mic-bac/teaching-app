"""
6_Segmentation.py
-----------------
Customer Segmentation lesson page (RFM · CLTV · Clustering).

Surfaces the sibling ``segmentation/`` repo as an interactive lesson: aggregating
a transaction log into RFM, measuring a retention rate from cohorts and turning
it into customer lifetime value, and clustering customers on demographics plus
behaviour. All logic lives in ``utils/segmentation_utils`` (UI-free), which in
turn imports the sibling's own ``src/`` modules — so what runs here is literally
the code the lesson scripts run, not a copy of it.

Two datasets on purpose: RFM and CLTV need dated transactions, clustering needs
customer attributes. The Overview tab makes that the first thing a learner sees.

Graceful behaviour: sibling repos are git-ignored, so this page checks that
``segmentation/`` is present and explains itself if not. When the repo is there
but its data files are not, the sibling loaders fall back to synthetic data and
the page keeps working (a neutral caption describes the data).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import segmentation_utils as S
from utils.teaching import source_of

st.set_page_config(page_title="Customer Segmentation", layout="wide", page_icon="🧩")

ROOT = Path(__file__).resolve().parents[1]
SIBLING = ROOT / "segmentation"


def show_source(obj, label: str) -> None:
    """Reveal the real function that runs, so code can't drift from behaviour."""
    with st.expander(label):
        st.code(source_of(obj), language="python")


def eur(value: float) -> str:
    """Euros with a proper minus sign: −€8, not €-8."""
    return f"−€{-value:,.0f}" if value < 0 else f"€{value:,.0f}"


def show_sibling_script(filename: str, label: str) -> None:
    """Show a sibling teaching script verbatim (single source of truth)."""
    with st.expander(label):
        path = SIBLING / filename
        try:
            st.code(path.read_text(), language="python")
        except OSError:
            st.warning(f"Could not read the reference script at `{path}`.")


# ===========================================================================
# The sibling repo has to be present — it is git-ignored, so it may not be
# ===========================================================================
if not S.SIBLING_AVAILABLE:
    st.title("🧩 Customer Segmentation")
    st.error(
        "This lesson reads its code and data from the sibling `segmentation/` repo, which "
        f"is not importable from `{SIBLING}`.\n\n"
        f"Import error: `{S.SIBLING_ERROR}`"
    )
    st.info(
        "Sibling repos are git-ignored by the app, so they have to be cloned alongside it. "
        "Once `segmentation/` is in place, this page works with no further setup — the "
        "lesson falls back to synthetic data if the datasets themselves are missing."
    )
    st.stop()


# ===========================================================================
# Cached data / model builders (keep utils Streamlit-free; cache here)
# ===========================================================================
@st.cache_resource(show_spinner="Loading the transaction log…")
def cached_transactions(b2c_only: bool = True):
    """The cleaned transaction log.

    ``cache_resource`` rather than ``cache_data`` on purpose: this is ~640,000
    rows, and ``cache_data`` hands back a deep copy on *every* rerun — which on a
    frame this size is a visible pause every time a widget moves. Nothing here
    mutates it (the sibling helpers all copy internally), so sharing one instance
    is both safe and much faster.
    """
    return S.load_transactions(b2c_only=b2c_only, verbose=False)


@st.cache_data(show_spinner=False)
def cached_customers():
    transactions, _ = cached_transactions()
    return S.customer_table(transactions)


@st.cache_data(show_spinner=False)
def cached_rfm():
    transactions, _ = cached_transactions()
    return S.segment_rfm(transactions)


@st.cache_data(show_spinner="Building cohorts…")
def cached_cohorts():
    transactions, _ = cached_transactions()
    return (
        S.cohort_retention(transactions),
        S.annual_retention(transactions),
        S.retention_by_recency_band(transactions),
    )


@st.cache_data(show_spinner="Comparing consumers and businesses…")
def cached_wholesale_comparison():
    """Consumers vs businesses, computed on demand.

    Deliberately loads its own unfiltered log rather than reusing the cached one:
    the unfiltered log is ~800,000 rows, and pinning it in the resource cache
    alongside the B2C log would keep 1.4 million rows resident for a table that
    is read once. Here the big frame is local, so it is freed as soon as the
    (two-row) comparison is built.
    """
    unfiltered, _ = S.load_transactions(b2c_only=False, verbose=False)
    return S.wholesale_comparison(unfiltered)


@st.cache_data(show_spinner="Loading customer attributes…")
def cached_personality():
    return S.load_personality(verbose=False)


@st.cache_resource(show_spinner=False)
def cached_feature_matrix():
    customers, _ = cached_personality()
    X, _scaler = S.scale_features(customers, S.CLUSTERING_FEATURES)
    return X


@st.cache_data(show_spinner="Sweeping k…")
def cached_k_sweep():
    return S.sweep_k(cached_feature_matrix(), range(2, 11))


@st.cache_resource(show_spinner=False)
def cached_eps():
    return S.choose_eps(cached_feature_matrix())


@st.cache_resource(show_spinner=False)
def cached_bandwidth():
    return S.estimate_meanshift_bandwidth(cached_feature_matrix())


@st.cache_resource(show_spinner="Fitting clustering algorithms…")
def cached_labels(k: int, algorithm_names: tuple[str, ...]):
    """Fit the selected algorithms at this k. Keyed by both, so changing k refits
    only once and switching tabs refits nothing."""
    X = cached_feature_matrix()
    eps, _curve = cached_eps()
    algorithms = S.build_algorithms(k, eps, cached_bandwidth())
    return {name: algorithms[name]["build"]().fit_predict(X) for name in algorithm_names}


@st.cache_resource(show_spinner=False)
def cached_pca():
    return S.pca_projection(cached_feature_matrix())


@st.cache_resource(show_spinner="Resampling — this one is deliberately slow…")
def cached_stability(k: int, algorithm_names: tuple[str, ...]):
    X = cached_feature_matrix()
    eps, _curve = cached_eps()
    algorithms = S.build_algorithms(k, eps, cached_bandwidth())
    selected = {name: algorithms[name] for name in algorithm_names}
    return S.stability_frame(selected, X, cached_labels(k, algorithm_names))


# ===========================================================================
# Header
# ===========================================================================
st.title("🧩 Customer Segmentation — who are our customers, and what are they worth?")
st.markdown(
    """
    *"We know what our customers buy. We do not know how best to talk to them, or which of
    them are worth investing in."*

    Three answers, in increasing order of ambition: **RFM** ranks customers on what they
    just did, **CLTV** puts a euro value on what they will do next, and **clustering** finally
    tells you *who they are*.

    Every tab shows the **real code** that produced the result — imported from the sibling
    `segmentation/` repo, not copied from it.
    """
)
with st.container(border=True):
    st.markdown(
        """
        💼 **Where you'll use this** — deciding who gets the loyalty programme and who gets a win-back
        mail, how much you can afford to spend to win a new customer, and writing campaign briefs for
        groups of real people (*"young families, price-conscious, mostly online"*) instead of averages.
        """
    )

transactions, _tx_source = cached_transactions()
personality, _personality_source = cached_personality()

st.caption(
    f"**The data:** an online gift retailer's transaction log ({len(transactions):,} order lines from "
    f"{transactions['Customer ID'].nunique():,} customers) for RFM and CLTV, and a table of "
    f"{len(personality):,} customers with demographics and behaviour for clustering."
)

tab_overview, tab_rfm, tab_cltv, tab_cluster, tab_compare = st.tabs(
    ["🗺️ Overview", "📊 RFM", "💶 CLTV", "🔬 Clustering", "🎯 Put together"]
)


# ===========================================================================
# Overview
# ===========================================================================
with tab_overview:
    st.subheader("Different questions need different data")
    st.markdown(
        """
        The single most common mistake in customer analytics is reaching for a method the
        data cannot support. These three methods want genuinely different things, and no
        amount of cleverness substitutes for the column you do not have.
        """
    )
    st.dataframe(S.dataset_roles(), width="stretch", hide_index=True)
    st.caption(
        "Why two datasets? **Our** transaction log has no age or income column, and **our** "
        "attribute table has no purchase dates — the two public datasets are not linked. In a real "
        "company they often are: a **loyalty card** or online account ties every receipt to a person "
        "(and their age, address, household), so one joined table can feed all three methods."
    )

    st.divider()
    st.subheader("How the lesson fits together")
    st.graphviz_chart(S.segmentation_architecture_dot(), width="stretch")
    st.caption(
        "Green: the two datasets this lesson uses. Dashed: the link a loyalty card or CRM gives a real "
        "retailer — with it, the clustering could use purchase history *and* demographics together."
    )

    st.divider()
    st.subheader("Preparing the transaction log: keep consumers, drop resellers")
    st.markdown(
        """
        A data-preparation step for the **RFM and CLTV tabs**. The online retailer in our transaction
        log sells to the public *and* to shops that resell its stock, and the two behave nothing
        alike — a reseller orders 96 of one item every fortnight. Since this lesson is about
        consumers (B2C), the resellers are removed first. Nothing in the data says which is which,
        so it is inferred from buying behaviour: **median line quantity ≥ 24**, **average order
        value ≥ 1,000**, or **50+ orders** in the two years.
        """
    )
    show_source(S.is_wholesale, "Show the code: how a business buyer is identified")

    # Loading the *unfiltered* log costs another 800,000 rows, so it happens only
    # when asked for — the rest of the page never needs it.
    if st.button("Show me the two populations side by side"):
        comparison = cached_wholesale_comparison()
        st.dataframe(
            comparison.style.format(
                {
                    "Customers": "{:,.0f}",
                    "Line items": "{:,.0f}",
                    "Revenue": "{:,.0f}",
                    "Median order value": "{:,.0f}",
                    "Median line quantity": "{:,.0f}",
                    "Recency vs Frequency": "{:+.2f}",
                    "Share of revenue": "{:.1%}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        consumer_row, business_row = comparison.iloc[0], comparison.iloc[1]
        st.markdown(
            f"""
            Dropping the businesses costs **{business_row['Share of revenue']:.0%} of
            revenue** — a lot, and shown rather than hidden, because a lesson that quietly
            deleted half a company's turnover would be teaching the wrong habit.

            It also *sharpens* the signal. Recency against frequency correlates
            **{business_row['Recency vs Frequency']:+.2f}** among businesses (they restock on
            a schedule, so recency says little about them) versus
            **{consumer_row['Recency vs Frequency']:+.2f}** among consumers. Mixed together,
            the businesses were diluting the very relationship RFM depends on.
            """
        )


# ===========================================================================
# RFM
# ===========================================================================
with tab_rfm:
    st.subheader("From a transaction log to three numbers per customer")
    st.markdown(
        """
        RFM describes each customer with three numbers from their purchase history:

        - **R — Recency:** how many days since their last order? (fewer = better)
        - **F — Frequency:** how many orders have they placed?
        - **M — Monetary:** how much have they spent in total?
        """
    )

    scored = cached_rfm()
    example = S.example_customer(scored)
    orders = S.customer_orders(transactions, example["Id"])
    snapshot = transactions["InvoiceDate"].max() + pd.Timedelta(days=1)

    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"**Example: customer {example['Id']:.0f}'s orders**")
        st.dataframe(
            orders.style.format({"Amount": "€{:,.2f}"}), width="stretch", hide_index=True,
        )
    with right:
        st.markdown("**…become three numbers**")
        st.markdown(
            f"""
            - **R = {example['Recency']:.0f} days** — last order on {orders['Date'].iloc[-1]},
              counted to "today" (the day after the data ends, {snapshot.date()}).
            - **F = {example['Frequency']:.0f} orders** — one per row on the left. An order
              with {orders['Products'].max()} products still counts **once**.
            - **M = €{example['Monetary']:,.0f}** — the sum of the Amount column.
            """
        )
    show_source(S.build_rfm, "Show the code: the whole of RFM's data work")

    st.divider()
    st.subheader("Scoring, 1 (poor) to 5 (excellent)")
    st.markdown(
        "Euros and days cannot be compared directly, so each dimension is turned into a **rank**: "
        "sort all customers, cut the list into five equal groups, and give the best group a 5. "
        "Recency runs **backwards** — bought yesterday is *good*, so it scores 5."
    )
    st.markdown(f"For customer {example['Id']:.0f}:")
    for line in S.score_reading(scored, example):
        st.markdown(f"- {line}")
    st.markdown(
        f"Together: cell **`{example['RFM_Cell']}`** (the three digits side by side) and a sum of "
        f"**{example['RFM_Sum']:.0f}** out of 15."
    )
    st.dataframe(
        scored[["Id", "Recency", "Frequency", "Monetary", "R_Score", "F_Score", "M_Score",
                "RFM_Cell", "RFM_Sum"]].head(10),
        width="stretch", hide_index=True,
    )

    busiest_sum = int(scored["RFM_Sum"].value_counts().idxmax())
    tied = scored[scored["RFM_Sum"] == busiest_sum]
    most_recent = tied.loc[tied["R_Score"].idxmax()]
    least_recent = tied.loc[tied["R_Score"].idxmin()]
    st.info(
        f"**Why keep three digits instead of adding them up?** "
        f"{len(tied):,} customers score a total of {busiest_sum}. Among them, cell "
        f"`{most_recent['RFM_Cell']}` last bought {most_recent['Recency']:.0f} days ago "
        f"(€{most_recent['Monetary']:,.0f}), while cell `{least_recent['RFM_Cell']}` last "
        f"bought {least_recent['Recency']:.0f} days ago (€{least_recent['Monetary']:,.0f}). "
        "Same sum, opposite customers, opposite next actions — one needs a thank-you, the "
        "other a win-back."
    )

    st.divider()
    st.subheader("Two ways to turn scores into segments")
    cube_col, ladder_col = st.columns(2)

    with cube_col:
        st.markdown("**The RFM cube** — split each dimension at the median into high/low, "
                    "get 2 × 2 × 2 = eight named corners")
        cube_table = (
            scored.groupby("Cube")
            .agg(Customers=("Id", "count"), Recency=("Recency", "mean"),
                 Frequency=("Frequency", "mean"), Monetary=("Monetary", "mean"))
            .round(0)
            .sort_values("Customers", ascending=False)
        )
        st.dataframe(cube_table, width="stretch")

    with ladder_col:
        st.markdown("**The business ladder** — rules checked top to bottom, first match wins "
                    "(e.g. Champions = R, F and M all 4 or 5)")
        ladder_table = (
            scored.groupby("Segment")
            .agg(Customers=("Id", "count"), Recency=("Recency", "mean"),
                 Frequency=("Frequency", "mean"), Monetary=("Monetary", "mean"))
            .round(0)
            .sort_values("Customers", ascending=False)
        )
        st.dataframe(ladder_table, width="stretch")

    champions, lost = ladder_table.loc["Champions"], ladder_table.loc["Lost"]
    st.markdown(
        f"""
        **How to read the tables** (averages per segment): the {champions['Customers']:,.0f}
        **Champions** last ordered {champions['Recency']:.0f} days ago on average, placed
        {champions['Frequency']:.0f} orders and spent €{champions['Monetary']:,.0f} — a
        thank-you and early access, not a discount. The {lost['Customers']:,.0f} **Lost** customers
        have been quiet for {lost['Recency']:.0f} days after about {lost['Frequency']:.0f}
        order{'' if round(lost['Frequency']) == 1 else 's'} worth €{lost['Monetary']:,.0f} — not worth an expensive campaign. The cube uses
        neutral names for the same idea; the ladder's names are written for a marketing team.
        """
    )

    correlations = scored[["Recency", "Frequency", "Monetary"]].corr()
    st.caption(
        f"The eight corners of the cube are far from equally filled, and the correlation says why: "
        f"Frequency and Monetary correlate {correlations.loc['Frequency', 'Monetary']:+.2f} — "
        f"people who order often also spend more — so 'frequent but cheap' barely exists. "
        f"Recency is the more independent one "
        f"({correlations.loc['Recency', 'Monetary']:+.2f} against Monetary), which is exactly "
        f"why an 'At Risk' segment (spent well, but gone quiet) is worth defining."
    )

    show_source(S.rfm_ladder, "Show the code: the segment rules, in priority order")

    st.divider()
    st.subheader("The segments in 3D")
    view = st.radio(
        "Axes", ["Scores 1–5 (the cube)", "Raw values (log scale)"], horizontal=True,
        help="Scores show how the rules carve up the cube; raw values show the real days, "
             "orders and euros.",
    )
    highlight = st.multiselect(
        "Highlight segments",
        options=sorted(scored["Segment"].unique()),
        default=["Champions", "At Risk", "Lost"],
        help="Rotate the plot with the mouse — each segment occupies its own corner.",
    )
    if view.startswith("Scores"):
        cloud = S.rfm_score_cloud(scored)
        plotted = cloud[cloud["Segment"].isin(highlight)] if highlight else cloud
        fig_rfm = px.scatter_3d(
            plotted, x="R", y="F", z="M", color="Segment", opacity=0.6,
            color_discrete_sequence=px.colors.qualitative.Set2,
            hover_data={"Id": True, "Recency": True, "Frequency": True, "Monetary": ":,.0f",
                        "R": False, "F": False, "M": False},
            labels={"R": "R score (5 = recent)", "F": "F score (5 = frequent)",
                    "M": "M score (5 = big spender)"},
        )
    else:
        plotted = scored[scored["Segment"].isin(highlight)] if highlight else scored
        fig_rfm = px.scatter_3d(
            plotted, x="Recency", y="Frequency", z="Monetary", color="Segment",
            opacity=0.6, color_discrete_sequence=px.colors.qualitative.Set2,
            log_x=True, log_y=True, log_z=True,
            labels={"Recency": "Recency (days)", "Frequency": "Orders", "Monetary": "Spend (€)"},
        )
    fig_rfm.update_traces(marker=dict(size=3, line=dict(width=0)))
    fig_rfm.update_layout(height=620, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_rfm, width="stretch")
    st.caption(
        "Each dot is a customer (a small random wobble is added in the score view, otherwise "
        "thousands would sit on the same 125 grid points). **Champions** fill the corner where all "
        "three scores are 4–5; **Lost** the opposite corner; **At Risk** sits at medium-to-high F "
        "but low R — they used to order regularly and have stopped. Hover over a dot to see the real numbers."
    )

    show_sibling_script("rfm.py", "📄 The full lesson script: segmentation/rfm.py")


# ===========================================================================
# CLTV
# ===========================================================================
with tab_cltv:
    customers = cached_customers()
    retention_matrix, annual, band_retention = cached_cohorts()
    measured_rr = annual["retention_rate"]

    st.subheader("What is a customer worth over their whole lifetime?")
    st.markdown(
        r"""
        In plain words: **what a customer earns us per year × how many years they are likely to
        stay (in today's money) − what it cost to win them.** The lecture's formula:

        $$LTV_i = GC_i \times \frac{Rr}{1 - Rr + d} - AC_i$$

        | Symbol | Meaning | In this lesson |
        |---|---|---|
        | $GC_i$ | **Gross contribution** — the *profit* (not revenue) customer *i* brings per year | revenue per year × margin |
        | $Rr$ | **Retention rate** — chance a customer is still buying a year later | measured below |
        | $d$ | **Discount rate** — a euro next year is worth less than a euro today | assumption |
        | $AC_i$ | **Acquisition cost** — marketing spent to win the customer | assumption |
        | $\frac{Rr}{1-Rr+d}$ | **Lifetime multiplier** — "how many years of profit, in today's money" | follows from Rr and d |
        """
    )

    example_gc = 300.0
    example_multiplier = S.clv_multiplier(measured_rr, S.DISCOUNT_RATE)
    example_ltv = example_gc * example_multiplier - S.ACQUISITION_COST
    with st.container(border=True):
        st.markdown(
            f"""
            **Worked example.** A customer spends €1,000 a year. At a {S.GROSS_MARGIN:.0%} margin
            that is **GC = €{example_gc:,.0f}** of profit a year. With the measured retention
            **Rr = {measured_rr:.0%}** and **d = {S.DISCOUNT_RATE:.0%}**, the multiplier is
            {measured_rr:.2f} / (1 − {measured_rr:.2f} + {S.DISCOUNT_RATE:.2f}) =
            **{example_multiplier:.2f}**. So the customer is worth
            €{example_gc:,.0f} × {example_multiplier:.2f} − €{S.ACQUISITION_COST:,.0f} =
            **€{example_ltv:,.0f}**: the €{S.ACQUISITION_COST:,.0f} spent to win them pays back
            {(example_gc * example_multiplier) / S.ACQUISITION_COST:.1f} times.
            """
        )
    with st.expander("🔎 The maths, if you're curious — where the multiplier comes from"):
        st.markdown(
            "Each future year is worth GC × (chance they are still here) × (what a euro then is "
            "worth today). Here is the example customer, year by year:"
        )
        st.dataframe(
            S.clv_year_by_year(example_gc, measured_rr, S.DISCOUNT_RATE).style.format(
                {"Chance still a customer": "{:.0%}", "€1 then is worth today": "€{:.2f}",
                 "Expected value today (€)": "€{:,.0f}"}
            ),
            width="stretch", hide_index=True,
        )
        st.markdown(
            r"""
            The values shrink every year, and adding up *all* years (a geometric series) gives
            exactly the closed form:

            $$\sum_{t=1}^{\infty} GC \cdot \frac{Rr^t}{(1+d)^t} = GC \cdot \frac{Rr}{1 - Rr + d}$$

            **GC** itself is annualised: revenue ÷ years observed × margin, with at least one year
            assumed — otherwise a customer who bought once (zero days between first and last order)
            would be divided by zero.
            """
        )
        show_source(S.annual_contribution, "Show the code: GC per customer")

    st.divider()
    st.subheader("Measure the retention rate — do not guess it")
    st.markdown(
        "Retention cannot be read off a customer table — it needs **cohorts**: take everyone who "
        "first bought in a given month, and count how many come back in each later month."
    )
    heat = S.cohort_matrix_frame(retention_matrix)
    fig_cohort = px.imshow(
        heat, text_auto=".0f", aspect="auto", color_continuous_scale="Blues",
        labels=dict(x="Months since first purchase", y="Cohort (month of first order)",
                    color="% ordering"),
    )
    fig_cohort.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_cohort, width="stretch")
    st.markdown(S.cohort_reading(retention_matrix))
    st.caption(
        "Column 0 is 100% by definition (it is the month they joined). The empty bottom-right is "
        "**not** zero retention — a cohort that joined three months before the data ends simply "
        "has no twelve-month figure yet."
    )

    curve_frame = S.retention_curve_frame(retention_matrix)
    fig_curve = px.line(
        curve_frame, x="Period", y="Retention", color="Cohort",
        labels={"Period": "Months since first purchase", "Retention": "% ordering that month"},
    )
    fig_curve.update_traces(opacity=0.35)
    fig_curve.update_traces(selector=dict(name="average"), opacity=1.0,
                            line=dict(width=4, color="black"))
    fig_curve.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_curve, width="stretch")
    average_curve = curve_frame[curve_frame["Cohort"] == "average"].set_index("Period")["Retention"]
    st.caption(
        f"Same numbers as lines; the black line is the average cohort. It drops to about "
        f"{average_curve.iloc[0]:.0f}% in month 1 and then **flattens** around "
        f"{average_curve.iloc[-6:].mean():.0f}% instead of falling to zero — an early drop-off, "
        "then a loyal core. Monthly numbers look low because most people do not buy gifts every "
        "month; that is why the CLTV formula uses a **yearly** rate instead:"
    )
    st.success(
        f"**Measured annual retention:** of the {annual['cohort_size']:,} customers who first "
        f"bought in the first year, {annual['returned']:,} bought again in the second year — "
        f"**Rr = {measured_rr:.1%}**."
    )
    show_source(S.annual_retention, "Show the code: measuring Rr from cohorts")

    st.divider()
    st.subheader("Retention is not the same for everybody")
    st.markdown(
        "One rate for the whole base is an average. Someone who ordered last week is far more "
        "likely to come back than someone silent for half a year. We can measure this: stand at "
        "the one-year mark, group customers by how long they had been quiet, and count who "
        "ordered again in the next twelve months."
    )
    band_plot = band_retention.reset_index()
    band_plot["Band"] = band_plot["Band"].astype(str)
    fig_band = px.bar(
        band_plot, x="Band", y="RetentionRate", text_auto=".0%",
        labels={"Band": "Days since last order (at the one-year mark)",
                "RetentionRate": "Ordered again within a year"},
    )
    fig_band.add_hline(y=measured_rr, line_dash="dash", line_color="green",
                       annotation_text=f"one rate for all: {measured_rr:.0%}")
    fig_band.update_layout(height=340, yaxis_tickformat=".0%", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_band, width="stretch")

    rr_each = S.behavioural_retention(customers["Recency"], band_retention)
    gc_default = S.annual_contribution(customers)
    first_band, last_band = band_retention.iloc[0], band_retention.iloc[-1]
    same_gc = 300.0
    st.markdown(
        f"""
        **Reading:** customers who had ordered in the last
        {int(band_retention.index[0].split('-')[1].rstrip('d')) + 1} days came back
        {first_band['RetentionRate']:.0%} of the time; those quiet for
        {band_retention.index[-1].replace('-', '–').replace('d', ' days')} only
        {last_band['RetentionRate']:.0%}.

        **What it does to the value:** two customers who each bring GC = €{same_gc:,.0f} a year.
        With one rate for all, both are worth
        {eur(same_gc * S.clv_multiplier(measured_rr, S.DISCOUNT_RATE) - S.ACQUISITION_COST)}.
        With their own rate, the recent one is worth
        **{eur(same_gc * S.clv_multiplier(first_band['RetentionRate'], S.DISCOUNT_RATE) - S.ACQUISITION_COST)}**
        and the quiet one **{eur(same_gc * S.clv_multiplier(last_band['RetentionRate'], S.DISCOUNT_RATE) - S.ACQUISITION_COST)}**.
        Across the whole base, total value moves
        **{S.predictive_clv(gc_default, rr_each, S.DISCOUNT_RATE, S.ACQUISITION_COST).sum() / S.predictive_clv(gc_default, measured_rr, S.DISCOUNT_RATE, S.ACQUISITION_COST).sum() - 1:+.0%}**.
        Recency is just the factor this data lets us measure; with more data you would also split
        by channel, product category or acquisition campaign.
        """
    )
    show_source(S.retention_by_recency_band, "Show the code: retention per recency band")

    st.divider()
    st.subheader("Now turn it into money — and watch how much the assumptions matter")
    st.markdown(
        "`Rr` is measured; the other three are **business assumptions**. Drag them and watch the "
        "value of the whole customer base move without a single number in the data changing."
    )

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        gross_margin = st.slider("Gross margin", 0.05, 0.60, S.GROSS_MARGIN, 0.05, format="%.2f",
                                 help="Profit per euro of revenue.")
    with s2:
        retention_rate = st.slider("Retention rate Rr", 0.30, 0.95, float(round(measured_rr, 2)),
                                   0.01, help=f"Measured from cohorts: {measured_rr:.1%}")
    with s3:
        discount_rate = st.slider("Discount rate d", 0.00, 0.30, S.DISCOUNT_RATE, 0.01)
    with s4:
        acquisition_cost = st.slider("Acquisition cost €", 0, 600, int(S.ACQUISITION_COST), 25)

    gross_contribution = S.annual_contribution(customers, gross_margin)
    clv = S.predictive_clv(gross_contribution, retention_rate, discount_rate, acquisition_cost)
    multiplier = S.clv_multiplier(retention_rate, discount_rate)
    ceiling = gross_contribution * multiplier

    unprofitable = ceiling < acquisition_cost
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Lifetime multiplier", f"{multiplier:.2f}×",
              help="Years of contribution each customer is worth, in today's money.")
    m2.metric("Customer equity", f"€{clv.sum():,.0f}",
              help="The value of the entire customer base — the slides' Customer Equity.")
    m3.metric("Median CLTV", f"€{clv.median():,.0f}")
    m4.metric("Unprofitable to acquire", f"{(ceiling < acquisition_cost).mean():.0%}",
              help="Customers whose lifetime value never repays the acquisition cost.")
    st.caption(
        f"**Reading:** each customer is worth {multiplier:.2f} years of their yearly profit. Added "
        f"up over all {len(clv):,} customers that is the **customer equity** — roughly what the "
        f"customer base is worth to the company. The typical (median) customer is worth "
        f"{eur(clv.median())} after acquisition cost, and for {(ceiling < acquisition_cost).mean():.0%} "
        f"of customers the €{acquisition_cost} spent to win them never pays back"
        + (f" — {(customers.loc[unprofitable, 'Orders'] == 1).mean():.0%} of those ordered only once."
           if unprofitable.any() else ".")
    )

    if retention_rate != round(measured_rr, 2):
        equity_at_measured = S.predictive_clv(
            gross_contribution, measured_rr, discount_rate, acquisition_cost
        ).sum()
        delta = clv.sum() / equity_at_measured - 1
        st.warning(
            f"At Rr = {retention_rate:.0%} the customer base is worth **{delta:+.0%}** "
            f"compared with the measured {measured_rr:.1%}. None of the values on this slider "
            "is unreasonable — they are just wrong, and each one would set a different "
            "acquisition budget for the whole business."
        )

    grid = np.round(np.arange(0.35, 0.96, 0.01), 2)
    equity = S.equity_curve(gross_contribution, grid, discount_rate, acquisition_cost)
    fig_equity = px.line(equity, x="RetentionRate", y="CustomerEquity",
                         labels={"RetentionRate": "Retention rate Rr",
                                 "CustomerEquity": "Customer equity (€)"})
    fig_equity.add_vline(x=measured_rr, line_dash="dash", line_color="green",
                         annotation_text=f"measured {measured_rr:.1%}")
    fig_equity.add_vline(x=retention_rate, line_dash="dot", line_color="red",
                         annotation_text="your setting")
    fig_equity.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_equity, width="stretch")
    eq = equity.set_index("RetentionRate")["CustomerEquity"]
    st.caption(
        f"**Reading:** each point is the value of the whole customer base if retention were that "
        f"rate. The curve bends **upwards**: going from 50% to 60% adds €{eq.loc[0.6] - eq.loc[0.5]:,.0f}, "
        f"going from 80% to 90% adds €{eq.loc[0.9] - eq.loc[0.8]:,.0f}. Keeping customers is a "
        "powerful lever — often stronger than any discount."
    )

    with st.expander("The sensitivity table, in full"):
        st.dataframe(
            S.sensitivity_frame([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
                                [0.05, 0.10, 0.15, 0.20]),
            width="stretch",
        )
        st.caption(
            "Each cell is the lifetime multiplier. Example: at Rr = 60% and d = 10% a customer is "
            "worth 1.20 years of profit; at Rr = 90% it is 4.50 years — ~3.8× more. Moving the "
            "discount rate across its whole plausible range (5% → 20%) does far less."
        )

    st.divider()
    st.subheader("Value tiers")
    tiers = S.value_tiers(clv)
    tier_table = (
        pd.DataFrame({"CLTV": clv, "Tier": tiers, "Revenue": customers["Revenue"],
                      "Orders": customers["Orders"], "Recency": customers["Recency"]})
        .groupby("Tier", observed=True)
        .agg(Customers=("CLTV", "size"), Mean_CLTV=("CLTV", "mean"),
             Mean_Revenue=("Revenue", "mean"), Mean_Orders=("Orders", "mean"),
             Mean_Recency=("Recency", "mean"))
        .round(1)
    )
    tier_table["Share of equity"] = (
        pd.DataFrame({"CLTV": clv, "Tier": tiers})
        .groupby("Tier", observed=True)["CLTV"].sum() / clv.sum()
    )
    st.dataframe(
        tier_table.style.format({"Share of equity": "{:.1%}", "Mean_CLTV": "€{:,.0f}",
                                 "Mean_Revenue": "€{:,.0f}"}),
        width="stretch",
    )
    premium, low = tier_table.iloc[-1], tier_table.iloc[0]
    st.caption(
        f"**Reading:** customers are sorted by CLTV and cut into four equal groups. The top quarter "
        f"(**Premium**, average €{premium['Mean_CLTV']:,.0f}, {premium['Mean_Orders']:.0f} orders) holds "
        f"**{premium['Share of equity']:.0%}** of the value of the whole base. The bottom quarter "
        f"(**Low Value**, average {eur(low['Mean_CLTV'])}) "
        + ("is worth *less than nothing*: their profit does not repay what it cost to win them. "
           if low["Mean_CLTV"] < 0 else "adds little. ")
        + "That is the case for spending retention budget unevenly."
    )

    show_sibling_script("cltv.py", "📄 The full lesson script: segmentation/cltv.py")


# ===========================================================================
# Clustering
# ===========================================================================
with tab_cluster:
    st.subheader("Letting the data find the segments")
    st.markdown(
        f"""
        RFM and CLTV rank customers on what they bought. Neither can say *who* a segment is.
        Clustering can: it groups customers who are **similar on several attributes at once** —
        here {", ".join(S.FEATURE_LABELS.get(f, (f,))[0] for f in S.CLUSTERING_FEATURES)}.

        It is **unsupervised**: there is no right answer to check against and no accuracy to
        report. So evaluating the result takes as much work as producing it.
        """
    )
    st.dataframe(S.FAMILIES, width="stretch", hide_index=True)
    st.caption(
        "Algorithms differ in what they *call* a cluster. Example: K-Means puts every customer with "
        "the nearest of k centres; DBSCAN only groups customers in crowded areas and calls loners "
        "'noise'. Same customers, different groups — as the comparison below shows."
    )

    st.divider()
    st.subheader("How many clusters?")
    st.markdown(
        """
        For each k from 2 to 10 the customers are clustered with K-Means and the result is scored
        four ways. All four ask the same question — *are the groups tight inside and far apart
        from each other?* — with different formulas:

        - **Silhouette** (−1 to 1): for each customer, is it closer to its own group than to the next
          one? 0.5 = clearly separated, 0.25 = overlapping, 0 = no structure.
        - **Calinski-Harabasz**: spread *between* groups ÷ spread *within* groups (higher = better).
        - **Davies-Bouldin**: how much each group overlaps its most similar neighbour (lower = better).
        - **Inertia**: total distance of customers to their group centre. It always falls as k grows,
          so look for the "elbow" where adding groups stops helping much.
        """
    )
    sweep = cached_k_sweep()
    metric_choice = S.choose_k(sweep)

    normalised = S.k_sweep_frame(sweep)
    fig_k = go.Figure()
    for column, label in [
        ("silhouette", "Silhouette"), ("calinski_harabasz", "Calinski-Harabasz"),
        ("davies_bouldin", "Davies-Bouldin (flipped)"), ("inertia", "Inertia (flipped)"),
    ]:
        fig_k.add_trace(go.Scatter(x=normalised["k"], y=normalised[column], mode="lines+markers",
                                   name=label))
    fig_k.update_layout(height=380, yaxis_title="Rescaled 0-1 — up is always better",
                        xaxis_title="Number of clusters (k)", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_k, width="stretch")
    st.caption(
        "The four scores are rescaled to 0–1 and flipped where needed, so **up is always better** "
        "and they fit on one chart. The raw values are in the table below."
    )

    k = st.slider("Clusters to fit (k)", 2, 8, 4,
                  help="The metrics have a favourite. It is not the one we ship — see below.")
    silhouette_by_k = sweep.set_index("k")["silhouette"]
    st.info(
        f"**Every quality metric prefers k = {metric_choice}.** We use **k = 4** anyway: k = 2 just "
        f"splits the base into 'spends a lot' and 'spends little', which RFM already gave us for "
        f"free. Between k = 4, 5 and 6 the silhouette differs by only "
        f"{silhouette_by_k.loc[[4, 5, 6]].max() - silhouette_by_k.loc[[4, 5, 6]].min():.3f} — "
        "the metric cannot tell them apart, so four groups (easy to brief a team on) wins. "
        "**The metrics narrow the field; a human picks.**"
    )
    st.dataframe(sweep.round(3), width="stretch", hide_index=True)
    show_source(S.sweep_k, "Show the code: scoring each k four ways")

    st.divider()
    st.subheader("Four algorithms, four ideas of what a cluster is")
    algorithm_names = st.multiselect(
        "Algorithms to fit",
        options=["K-Means", "Hierarchical", "DBSCAN", "MeanShift"],
        default=["K-Means", "Hierarchical", "DBSCAN", "MeanShift"],
        help="K-Means and Hierarchical are told k; DBSCAN and MeanShift decide the number "
             "of clusters themselves.",
    )
    if not algorithm_names:
        st.info("Select at least one algorithm to fit.")
    else:
        labels = cached_labels(k, tuple(algorithm_names))
        X = cached_feature_matrix()

        st.markdown("**How big are the groups?**")
        sizes = S.cluster_size_frame(labels)
        sizes["Group"] = [
            "Noise" if c == "Noise" else ("largest" if r == 0 else f"#{r + 1}")
            for c, r in zip(sizes["Cluster"], sizes["Rank"])
        ]
        groups = [g for g in ["largest", *[f"#{i}" for i in range(2, 30)]] if g in set(sizes["Group"])]
        palette = px.colors.sample_colorscale("Teal", [1 - i / max(len(groups), 2) for i in range(len(groups))])
        fig_sizes = px.bar(
            sizes, y="Algorithm", x="Share", color="Group", orientation="h", text="Customers",
            category_orders={"Group": [*groups, "Noise"], "Algorithm": algorithm_names},
            color_discrete_map={**dict(zip(groups, palette)), "Noise": "lightgrey"},
            hover_data={"Cluster": True, "Customers": True, "Share": ":.1%", "Group": False},
            labels={"Share": "Share of customers", "Group": "Group (by size)"},
        )
        fig_sizes.update_traces(textposition="inside", insidetextanchor="middle",
                                marker_line_color="white", marker_line_width=1)
        fig_sizes.update_layout(height=120 + 55 * len(algorithm_names), xaxis_tickformat=".0%",
                                barmode="stack", legend=dict(orientation="h", y=-0.25),
                                margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_sizes, width="stretch")
        st.caption("Each bar is 100% of customers, cut into that algorithm's groups (largest first, "
                   "numbers = customers; grey = noise, not placed in any group).")
        for line in S.cluster_size_reading(sizes):
            st.markdown(f"- {line}")
        st.markdown(
            "K-Means tends towards **balanced** groups, because every customer must join the "
            "nearest centre. The density-based methods (DBSCAN, MeanShift) find a few "
            "big crowds plus tiny pockets — **extreme** sizes. For a marketing campaign, a cluster of 9 customers is no segment."
        )

        st.markdown("**The scores per algorithm**")
        scores = pd.DataFrame(
            {name: S.score_clustering(X, lab) for name, lab in labels.items()}
        ).T
        st.dataframe(scores.round(3), width="stretch")
        best = scores["silhouette"].astype(float).idxmax()
        st.caption(
            f"`clusters` = number of groups found, `noise` = share left unassigned, "
            f"`smallest_cluster` = customers in the smallest group; then the three quality scores "
            f"from above. Read them as a group, not a leaderboard: here **{best}** separates best "
            f"(silhouette {scores.loc[best, 'silhouette']:.2f}), but always check the sizes first — "
            "a high score from a 5-customer cluster is worthless."
        )
        show_source(S.score_clustering, "Show the code: scoring without labels")

        coords, variance = cached_pca()
        st.markdown(
            f"**Visual inspection** — customers have {len(S.CLUSTERING_FEATURES)} attributes, so we "
            f"squeeze them onto a 2D map (PCA) that keeps {variance.sum():.0%} of the differences "
            "between them: good enough to see the broad picture, not to judge exact borders."
        )
        projection_cols = st.columns(min(len(algorithm_names), 4))
        for column, name in zip(projection_cols, algorithm_names):
            with column:
                frame = S.pca_frame(coords, labels[name])
                fig_pca = px.scatter(frame, x="PC1", y="PC2", color="Cluster", opacity=0.6,
                                     title=name,
                                     color_discrete_sequence=px.colors.qualitative.Set2)
                fig_pca.update_traces(marker=dict(size=4, line=dict(width=0)))
                fig_pca.update_layout(height=360, showlegend=False,
                                      margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig_pca, width="stretch")
        st.caption(
            "Each dot is a customer; nearby dots are similar customers. Colours are the groups. "
            "Good clustering shows colours in **separate regions**; K-Means typically cuts the cloud "
            "into neat slices, while a single dominant colour (as MeanShift tends to produce) means "
            "the algorithm saw one big crowd."
        )

        st.divider()
        st.subheader("Are the clusters real? The stability check")
        st.markdown(
            "Every algorithm returns *something* on any input, and none of them warn you when "
            "the structure they found is an accident of this sample. Refit on random 80% "
            "subsamples and measure whether the same customers stay together."
        )
        if st.button("Run the stability check", help="Refits every selected algorithm five times."):
            st.dataframe(
                cached_stability(k, tuple(algorithm_names)).round(3),
                width="stretch", hide_index=True,
            )
            st.caption(
                "**ARI** (agreement): 1.0 = the same customers end up together every time, 0 = random. "
                "Example: an ARI of 0.9 means if you re-ran the analysis next quarter, almost every "
                "customer pair would be grouped the same way. For campaigns you want well above 0.75. "
                "But stable is not enough: DBSCAN finds the *same meaningless* tiny groups every time. "
                "So the verdict also checks the **silhouette**: below 0.1 means there is no real "
                "structure, however stable it looks."
            )
        show_source(S.stability, "Show the code: resampling and measuring agreement")
        show_source(S.stability_verdict, "Show the code: the verdict reads stability AND separation")

        st.divider()
        st.subheader("Who are these people?")
        primary = st.selectbox(
            "Algorithm to profile", algorithm_names,
            index=algorithm_names.index("K-Means") if "K-Means" in algorithm_names else 0,
        )
        st.markdown(
            f"Profiles of the **{primary}** clusters (k = {k}"
            + (", as set above" if primary in ("K-Means", "Hierarchical")
               else "; this algorithm chose the number itself") + ")."
        )
        customers_attr, _ = cached_personality()
        profiles = S.profile_clusters(customers_attr, labels[primary], S.CLUSTERING_FEATURES)
        fingerprints = S.cluster_fingerprints(customers_attr, profiles, S.CLUSTERING_FEATURES)

        st.dataframe(profiles, width="stretch")
        st.caption("Average of each attribute per cluster. Hard to compare across columns — income "
                   "is in tens of thousands, children in single digits. Hence the heatmap:")
        fig_fingerprint = px.imshow(
            fingerprints.T, text_auto=True, color_continuous_scale="RdBu_r",
            zmin=-2, zmax=2, aspect="auto",
            labels=dict(x="Cluster", y="Feature", color="z-score"),
        )
        fig_fingerprint.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_fingerprint, width="stretch")
        st.caption(
            "Each cell: how far the cluster's average is from the average customer, in standard "
            "deviations (a standard yardstick, so income and children become comparable). **Red = "
            "above average, blue = below**; ±1 or more is a strong difference, near 0 = typical. "
            "Example: +1.7 on spending means this group spends far more than the typical customer."
        )
        st.markdown(f"**The {primary} clusters in plain words:**")
        for cluster in sorted(fingerprints.index):
            name = "Noise (not in any cluster)" if cluster == -1 else f"Cluster {cluster}"
            count = profiles.loc[cluster, "Customers"]
            st.markdown(
                f"- **{name}** — {count:,} customers ({count / profiles['Customers'].sum():.0%}): "
                f"{S.describe_cluster(profiles.loc[cluster], fingerprints.loc[cluster], top=4)}."
            )
        st.caption(
            "*This* is what clustering adds. A spending quartile tells you a customer is in the top "
            "25%; a profile like 'high income, no children at home, accepts campaigns' is a brief you "
            "can hand to a copywriter — e.g. premium products and new-collection mailings for that "
            "group, family bundles and value offers for the high-children, low-spend one."
        )

    show_sibling_script("clustering.py", "📄 The full lesson script: segmentation/clustering.py")


# ===========================================================================
# Put together
# ===========================================================================
with tab_compare:
    st.subheader("RFM tells you who is slipping away. CLTV tells you what that costs.")
    st.markdown(
        "Cross the two and you stop having rankings and start having a to-do list. "
        "The tiers below use the measured retention rate and the lesson's default assumptions "
        "(`segmentation/src/cltv_core.py`) — not the CLTV-tab sliders — so they match `cltv.py`."
    )

    scored = cached_rfm()
    customers = cached_customers()
    _matrix, annual, _bands = cached_cohorts()
    gross_contribution = S.annual_contribution(customers, S.GROSS_MARGIN)
    clv = S.predictive_clv(gross_contribution, annual["retention_rate"], S.DISCOUNT_RATE, S.ACQUISITION_COST)

    multiplier_default = S.clv_multiplier(annual["retention_rate"], S.DISCOUNT_RATE)
    combined = pd.DataFrame(
        {
            "Customer ID": customers["Customer ID"],
            "Tier": S.value_tiers(clv).astype(str),
            "FutureValue": gross_contribution * multiplier_default,
        }
    ).merge(
        scored[["Id", "Segment"]].rename(columns={"Id": "Customer ID"}), on="Customer ID"
    )
    cross_tab = pd.crosstab(combined["Segment"], combined["Tier"])
    order = [t for t in ["Low Value", "Medium Value", "High Value", "Premium"]
             if t in cross_tab.columns]
    cross_tab = cross_tab[order]

    fig_cross = px.imshow(
        cross_tab, text_auto=True, aspect="auto", color_continuous_scale="YlOrRd",
        labels=dict(x="CLTV tier", y="RFM segment", color="Customers"),
    )
    fig_cross.update_layout(height=480, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_cross, width="stretch")

    champions_premium = int(cross_tab.loc["Champions", "Premium"]) if (
        "Champions" in cross_tab.index and "Premium" in cross_tab.columns) else 0
    lost_low = int(cross_tab.loc["Lost", "Low Value"]) if (
        "Lost" in cross_tab.index and "Low Value" in cross_tab.columns) else 0
    st.caption(
        f"**Reading:** each cell counts customers with that RFM segment (row) and CLTV tier "
        f"(column). Example: {champions_premium:,} customers are both Champions *and* Premium — the "
        f"core of the business. {lost_low:,} are Lost *and* Low Value — no budget needed. The "
        "interesting cells are the mismatches: high value but slipping away."
    )

    st.markdown("**The same table, read as decisions**")
    st.dataframe(S.segment_action_table(cross_tab), width="stretch", hide_index=True)

    st.divider()
    st.subheader("From table to action: is a win-back campaign worth it?")
    at_risk_valuable = combined[
        combined["Segment"].isin(["Can't Lose Them", "At Risk"])
        & combined["Tier"].isin(["High Value", "Premium"])
    ]
    st.markdown(
        f"""
        Take the urgent cells: **{len(at_risk_valuable):,} customers** who are *At Risk* or
        *Can't Lose Them* **and** in the top two value tiers. They used to buy a lot and have gone
        quiet. Together they are expected to bring **€{at_risk_valuable['FutureValue'].sum():,.0f}**
        of future profit (GC × lifetime multiplier) — *if* they stay. Say we send each a personal
        offer (a voucher plus a call from customer service). Is it worth the money?
        """
    )
    c1, c2 = st.columns(2)
    with c1:
        cost_per_customer = st.slider("Campaign cost per customer (€)", 5, 100, 25, 5,
                                      help="Voucher value actually redeemed + contact cost.")
    with c2:
        success_rate = st.slider("Share of them the campaign keeps", 0.01, 0.50, 0.10, 0.01,
                                 format="%.2f",
                                 help="Customers who would have left but stay because of the campaign.")
    case = S.campaign_value(at_risk_valuable["FutureValue"], cost_per_customer, success_rate)
    lost_cheap = combined[(combined["Segment"] == "Lost") & (combined["Tier"] == "Low Value")]
    lost_case = S.campaign_value(lost_cheap["FutureValue"], cost_per_customer, success_rate)
    k1, k2, k3 = st.columns(3)
    k1.metric("Campaign cost", f"€{case['cost']:,.0f}")
    k2.metric("Profit kept", f"€{case['saved']:,.0f}")
    k3.metric("Net effect", f"€{case['net']:,.0f}", delta=f"{case['return_per_euro']:.1f}€ back per € spent",
              delta_color="normal" if case["net"] >= 0 else "inverse")
    st.markdown(
        f"""
        **Reading:** {case['customers']:,} customers × €{cost_per_customer} = €{case['cost']:,.0f}.
        If the campaign keeps {success_rate:.0%} of them, it saves {success_rate:.0%} ×
        €{at_risk_valuable['FutureValue'].sum():,.0f} = €{case['saved']:,.0f}. It breaks even if it keeps
        just **{case['break_even_rate']:.1%}** of them. The same €{cost_per_customer} per head spent
        on the {len(lost_cheap):,} *Lost × Low Value* customers would need to keep
        **{lost_case['break_even_rate']:.0%}** of them just to break even — that is the business
        value of crossing RFM with CLTV: **the budget goes where the money is.**

        **And clustering?** It tells you *how* to talk to them. In a company where a loyalty card
        links purchases to customer attributes, you would look up which clusters these customers
        belong to and write the offer accordingly — premium products for the affluent big spenders,
        family bundles for the households with children.
        """
    )
    show_source(S.campaign_value, "Show the code: the campaign maths")

    st.divider()
    st.subheader("What each method could not see")
    st.markdown(
        """
        - **RFM** measured outcomes only. It knows a customer went quiet; it does not know
          whether they are worth chasing, and its scores are *relative* — a customer can
          change segment next quarter without changing behaviour at all.
        - **CLTV** put a euro value on that, with a retention rate measured from cohorts
          rather than guessed. But it still saw nothing except purchases, and its answer moves
          more with the assumptions than with the data.
        - **Clustering** finally described the customers — and needed a different dataset to do
          it, because our transaction log has no age, income or household. (With a loyalty card
          linking the two, one dataset would do.)

        Three methods, three questions, three data requirements. That is the lesson.
        """
    )
    with st.container(border=True):
        st.markdown(
            """
🎯 **What to remember**
- **RFM is quick and cheap** — three numbers from your sales data already sort customers usefully.
- **CLTV turns segments into euros** — and retention is the number that moves it most, so measure it.
- **Clustering tells you *who* people are** — but only with the right data, and only if the groups
  are stable *and* clearly separated.
"""
        )
