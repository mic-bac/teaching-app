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
the page says so in the header.
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

transactions, tx_source = cached_transactions()
personality, personality_source = cached_personality()

left, right = st.columns(2)
with left:
    if tx_source == "retail":
        st.success(
            f"**Transactions:** Online Retail II — {len(transactions):,} B2C line items, "
            f"{transactions['Customer ID'].nunique():,} customers."
        )
    else:
        st.warning(
            "**Transactions:** the Online Retail II CSV was not found, so RFM and CLTV are "
            "running on a **synthetic** transaction log with the same structure. Everything "
            "works; the numbers are simulated. Run `uv run python fetch_data.py` in "
            "`segmentation/` for the real thing."
        )
with right:
    if personality_source == "kaggle":
        st.success(
            f"**Customer attributes:** Customer Personality — {len(personality):,} customers "
            f"with demographics and behaviour."
        )
    else:
        st.warning(
            "**Customer attributes:** the Kaggle CSV was not found, so clustering is running "
            "on a **synthetic** customer base with the same columns."
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
    st.dataframe(S.dataset_roles(), use_container_width=True, hide_index=True)
    st.caption(
        "This is why the lesson uses two datasets. A till receipt records a price and a "
        "product code — never an age — so clustering by demographics is impossible on the "
        "transaction log, and measuring a retention rate is impossible on the attribute file."
    )

    st.divider()
    st.subheader("How the lesson fits together")
    st.graphviz_chart(
        S.segmentation_architecture_dot(tx_source, personality_source), use_container_width=True
    )
    st.caption(
        "Green boxes are real datasets; orange means a synthetic fallback is standing in. "
        "The diagram is generated from the live data source, so it cannot lie about which "
        "one you are looking at."
    )

    st.divider()
    st.subheader("A judgement call, made in the open: B2C only")
    st.markdown(
        """
        This retailer sells to the public *and* to shops that resell its stock, and the two
        behave nothing alike — a reseller orders 96 of one item every fortnight. Nothing in
        the data says which is which, so it is inferred from buying behaviour: **median line
        quantity ≥ 24**, **average order value ≥ 1,000**, or **50+ orders** in the window.
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
            use_container_width=True,
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
        RFM is the cheapest useful segmentation there is, and almost all of its difficulty is
        in this first step. Everything afterwards is arithmetic on three columns.
        """
    )

    scored = cached_rfm()
    customers = cached_customers()

    a, b, c = st.columns(3)
    a.metric("Line items in", f"{len(transactions):,}")
    b.metric("Customers out", f"{len(scored):,}")
    c.metric("Ordered exactly once", f"{(scored['Frequency'] == 1).mean():.0%}")

    show_source(S.build_rfm, "Show the code: the whole of RFM's data work")

    st.markdown("**One customer, worked through by hand**")
    busy = scored.nlargest(20, "Frequency").iloc[10]
    customer_rows = transactions[transactions["Customer ID"] == busy["Id"]]
    lines, orders = len(customer_rows), customer_rows["Invoice"].nunique()
    st.markdown(
        f"""
        Customer **{busy['Id']:.0f}** has **{lines:,} line items** in the log across
        **{orders} distinct invoices** — so Frequency is **{busy['Frequency']:.0f}**, not
        {lines:,}. Counting rows instead of invoices would have made this customer look
        **{lines / orders:.0f}× more loyal** than they are, which is the most common way to
        get RFM quietly wrong.
        """
    )

    st.divider()
    st.subheader("Scoring, 1 (poor) to 5 (excellent)")
    st.markdown(
        "Recency runs **backwards** — bought yesterday is *good* — and all three dimensions "
        "are rank-transformed before binning so that ties do not collapse the quintiles."
    )
    st.dataframe(
        scored[["Id", "Recency", "Frequency", "Monetary", "R_Score", "F_Score", "M_Score",
                "RFM_Cell", "RFM_Sum"]].head(10),
        use_container_width=True, hide_index=True,
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
    st.subheader("Two ways to read the scores")
    cube_col, ladder_col = st.columns(2)

    with cube_col:
        st.markdown("**The RFM cube** — split each dimension in two, get eight named corners")
        cube_table = (
            scored.groupby("Cube")
            .agg(Customers=("Id", "count"), Recency=("Recency", "mean"),
                 Monetary=("Monetary", "mean"))
            .round(0)
            .sort_values("Customers", ascending=False)
        )
        st.dataframe(cube_table, use_container_width=True)

    with ladder_col:
        st.markdown("**The business ladder** — ordered rules, first match wins")
        ladder_table = (
            scored.groupby("Segment")
            .agg(Customers=("Id", "count"), Recency=("Recency", "mean"),
                 Monetary=("Monetary", "mean"))
            .round(0)
            .sort_values("Customers", ascending=False)
        )
        st.dataframe(ladder_table, use_container_width=True)

    correlations = scored[["Recency", "Frequency", "Monetary"]].corr()
    st.caption(
        f"The eight octants are far from equally filled, and the correlation matrix says why: "
        f"Frequency and Monetary correlate {correlations.loc['Frequency', 'Monetary']:+.2f}, so "
        f"'frequent but cheap' barely exists. The cube assumes three independent dimensions; "
        f"this data has closer to two. Recency is the independent one "
        f"({correlations.loc['Recency', 'Monetary']:+.2f} against Monetary) — which is exactly "
        f"why an 'At Risk' segment is worth defining."
    )

    show_source(S.rfm_ladder, "Show the code: the segment rules, in priority order")

    st.divider()
    st.subheader("The customer base in RFM space")
    highlight = st.multiselect(
        "Highlight segments",
        options=sorted(scored["Segment"].unique()),
        default=["Champions", "At Risk", "Lost"],
        help="Rotate the plot — the segments occupy visibly different corners.",
    )
    plotted = scored[scored["Segment"].isin(highlight)] if highlight else scored
    fig_rfm = px.scatter_3d(
        plotted, x="Recency", y="Frequency", z="Monetary", color="Segment",
        opacity=0.7, color_discrete_sequence=px.colors.qualitative.Set2,
        labels={"Recency": "Recency (days)", "Frequency": "Orders", "Monetary": "Spend (€)"},
    )
    fig_rfm.update_traces(marker=dict(size=3, line=dict(width=0)))
    fig_rfm.update_layout(height=620, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_rfm, use_container_width=True)

    show_sibling_script("rfm.py", "📄 The full lesson script: segmentation/rfm.py")


# ===========================================================================
# CLTV
# ===========================================================================
with tab_cltv:
    st.subheader("First, measure the retention rate — do not guess it")
    st.markdown(
        r"""
        The lecture's formula is

        $$LTV_i = GC_i \left(\frac{Rr}{1 - Rr + d}\right) - AC_i$$

        and everything turns on $Rr$, the share of customers who are still here a year from
        now. It cannot be read off a customer table — it needs **cohorts**: take everyone who
        first bought in a given month, and count how many come back in each later month.
        """
    )

    retention_matrix, annual, band_retention = cached_cohorts()

    heat = S.cohort_matrix_frame(retention_matrix)
    fig_cohort = px.imshow(
        heat, text_auto=".0f", aspect="auto", color_continuous_scale="Blues",
        labels=dict(x="Months since first purchase", y="Cohort", color="% retained"),
    )
    fig_cohort.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_cohort, use_container_width=True)
    st.caption(
        "Column 0 is 100% by construction, not as a finding. The empty bottom-left is **not** "
        "zero retention — a cohort acquired three months before the data ends simply cannot "
        "have a twelve-month figure yet. Reading those blanks as zeros is a classic way to "
        "invent a collapse that never happened."
    )

    curve_frame = S.retention_curve_frame(retention_matrix)
    fig_curve = px.line(
        curve_frame, x="Period", y="Retention", color="Cohort",
        labels={"Period": "Months since first purchase", "Retention": "% still buying"},
    )
    fig_curve.update_traces(opacity=0.35)
    fig_curve.update_traces(selector=dict(name="average"), opacity=1.0,
                            line=dict(width=4, color="black"))
    fig_curve.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_curve, use_container_width=True)
    st.caption(
        "Monthly retention looks alarming and is not: most people do not buy gifts every "
        "month. The curve **flattens** rather than falling to zero — an early drop-off, then "
        "a loyal core. That shape is the signature of a real customer base."
    )

    measured_rr = annual["retention_rate"]
    st.success(
        f"**Measured annual retention:** of the {annual['cohort_size']:,} customers acquired "
        f"in the first year, {annual['returned']:,} bought again in the second — "
        f"**Rr = {measured_rr:.1%}**. Measured, not assumed."
    )

    with st.expander("Retention is not the same for everybody"):
        st.markdown(
            "Stand at the one-year mark, sort everyone by how long they have been quiet, then "
            "look *forward* a year and count who came back. The recency half is measured "
            "strictly before the return half, so nothing here peeks at the future it predicts."
        )
        st.dataframe(
            band_retention.assign(
                RetentionRate=lambda t: (t["RetentionRate"] * 100).round(1)
            ).rename(columns={"RetentionRate": "Retention %"}),
            use_container_width=True,
        )
    show_source(S.annual_retention, "Show the code: measuring Rr from cohorts")

    st.divider()
    st.subheader("Now turn it into money — and watch how much the assumptions matter")
    st.markdown(
        "`Rr` above is measured. The other three are **business assumptions**. Drag them and "
        "watch the value of the entire customer base move without a single number in the data "
        "changing — this is the point of the whole lesson."
    )

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        gross_margin = st.slider("Gross margin", 0.05, 0.60, 0.30, 0.05, format="%.2f",
                                 help="Contribution per euro of revenue.")
    with s2:
        retention_rate = st.slider("Retention rate Rr", 0.30, 0.95, float(round(measured_rr, 2)),
                                   0.01, help=f"Measured from cohorts: {measured_rr:.1%}")
    with s3:
        discount_rate = st.slider("Discount rate d", 0.00, 0.30, 0.10, 0.01)
    with s4:
        acquisition_cost = st.slider("Acquisition cost €", 0, 600, 150, 25)

    customers = cached_customers()
    gross_contribution = S.annual_contribution(customers, gross_margin)
    clv = S.predictive_clv(gross_contribution, retention_rate, discount_rate, acquisition_cost)
    multiplier = S.clv_multiplier(retention_rate, discount_rate)
    ceiling = gross_contribution * multiplier

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Lifetime multiplier", f"{multiplier:.2f}×",
              help="Years of contribution each customer is worth, in today's money.")
    m2.metric("Customer equity", f"€{clv.sum():,.0f}",
              help="The value of the entire customer base — the slides' Customer Equity.")
    m3.metric("Median CLTV", f"€{clv.median():,.0f}")
    m4.metric("Unprofitable to acquire", f"{(ceiling < acquisition_cost).mean():.0%}",
              help="Customers whose lifetime value never repays the acquisition cost.")

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
    st.plotly_chart(fig_equity, use_container_width=True)
    st.caption(
        "The curve is **convex**: the same one-point improvement in retention is worth far "
        "more at 90% than at 50%. That is why retention beats discounting as a lever, and why "
        "averaging retention across a mixed customer base understates the total."
    )

    with st.expander("The sensitivity table, in full"):
        st.dataframe(
            S.sensitivity_frame([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
                                [0.05, 0.10, 0.15, 0.20]),
            use_container_width=True,
        )
        st.caption(
            "Moving retention from 60% to 90% multiplies every customer's value ~3.8×. "
            "Moving the discount rate across its whole plausible range does far less."
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
        use_container_width=True,
    )

    show_sibling_script("cltv.py", "📄 The full lesson script: segmentation/cltv.py")


# ===========================================================================
# Clustering
# ===========================================================================
with tab_cluster:
    st.subheader("Letting the data find the segments")
    st.markdown(
        """
        RFM and CLTV both rank customers on purchase outcomes. Neither can say *who* a segment
        is. Clustering can — but it is **unsupervised**: there is no correct answer to check
        against and no accuracy to report, so evaluating the result takes as much work as
        producing it.
        """
    )
    st.dataframe(S.FAMILIES, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("How many clusters?")
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
    st.plotly_chart(fig_k, use_container_width=True)

    k = st.slider("Clusters to fit (k)", 2, 8, 4,
                  help="The metrics have a favourite. It is not the one we ship — see below.")
    silhouette_by_k = sweep.set_index("k")["silhouette"]
    st.info(
        f"**Every quality metric prefers k = {metric_choice}.** We ship **k = 4** anyway, and "
        f"it is worth being explicit about why: k=2 splits the base into 'spends a lot' and "
        f"'spends little', which RFM already gave us for free; and between k=4, 5 and 6 the "
        f"silhouette varies by only "
        f"{silhouette_by_k.loc[[4, 5, 6]].max() - silhouette_by_k.loc[[4, 5, 6]].min():.3f} — "
        "the metric cannot separate them, so it is not the thing making this decision. "
        "**The metrics narrow the field; a human picks.** Pretending the number chose itself "
        "is the mistake."
    )
    st.dataframe(sweep.round(3), use_container_width=True, hide_index=True)
    show_source(S.sweep_k, "Show the code: scoring each k four ways")

    st.divider()
    st.subheader("Four algorithms, four ideas of what a cluster is")
    algorithm_names = st.multiselect(
        "Algorithms to fit",
        options=["K-Means", "Hierarchical", "DBSCAN", "MeanShift"],
        default=["K-Means", "Hierarchical", "DBSCAN"],
        help="MeanShift is the slow one — it searches for density modes rather than being "
             "told how many clusters to find.",
    )
    if not algorithm_names:
        st.info("Select at least one algorithm to fit.")
    else:
        labels = cached_labels(k, tuple(algorithm_names))
        X = cached_feature_matrix()

        scores = pd.DataFrame(
            {name: S.score_clustering(X, lab) for name, lab in labels.items()}
        ).T
        st.dataframe(scores.round(3), use_container_width=True)
        st.caption(
            "Noise points (DBSCAN's `-1`) are excluded before scoring — they are not a "
            "cluster, they are the points the algorithm refused to place — and the noise "
            "share is reported separately. Read these as a group, not a leaderboard: check "
            "`clusters`, `noise` and `smallest_cluster` before believing any of the scores."
        )
        show_source(S.score_clustering, "Show the code: scoring without labels")

        coords, variance = cached_pca()
        st.markdown(
            f"**Visual inspection** — the 2D projection keeps {variance.sum():.0%} of the "
            f"variance (PC1 {variance[0]:.0%}, PC2 {variance[1]:.0%}), enough to trust the "
            "broad picture but not to judge fine boundaries."
        )
        projection_cols = st.columns(min(len(algorithm_names), 3))
        for column, name in zip(projection_cols, algorithm_names):
            with column:
                frame = S.pca_frame(coords, labels[name])
                fig_pca = px.scatter(frame, x="PC1", y="PC2", color="Cluster", opacity=0.6,
                                     title=name,
                                     color_discrete_sequence=px.colors.qualitative.Set2)
                fig_pca.update_traces(marker=dict(size=4, line=dict(width=0)))
                fig_pca.update_layout(height=360, showlegend=False,
                                      margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig_pca, use_container_width=True)

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
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "Adjusted Rand index: 1.0 means the same customers grouped together every "
                "time, 0.0 means chance. Anything you intend to build campaigns on should be "
                "well above 0.75 — the segments have to survive next quarter's data too."
            )
        show_source(S.stability, "Show the code: resampling and measuring agreement")

        st.divider()
        st.subheader("Who are these people?")
        primary = algorithm_names[0]
        customers_attr, _ = cached_personality()
        profiles = S.profile_clusters(customers_attr, labels[primary], S.CLUSTERING_FEATURES)
        fingerprints = S.cluster_fingerprints(customers_attr, profiles, S.CLUSTERING_FEATURES)

        st.dataframe(profiles, use_container_width=True)
        fig_fingerprint = px.imshow(
            fingerprints.T, text_auto=True, color_continuous_scale="RdBu_r",
            zmin=-2, zmax=2, aspect="auto",
            labels=dict(x="Cluster", y="Feature", color="z-score"),
        )
        fig_fingerprint.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_fingerprint, use_container_width=True)
        st.markdown(f"**Reading the fingerprints ({primary}):**")
        for cluster in sorted(fingerprints.index):
            st.markdown(
                f"- **Cluster {cluster}** ({profiles.loc[cluster, 'Customers']:,} customers): "
                f"{S.name_segment(fingerprints.loc[cluster])}"
            )
        st.caption(
            "*This* is what clustering added. A spending quartile tells you a customer is in "
            "the top 25%; a fingerprint tells you they are affluent, childless and respond to "
            "campaigns — a brief you can hand to a copywriter."
        )

    show_sibling_script("clustering.py", "📄 The full lesson script: segmentation/clustering.py")


# ===========================================================================
# Put together
# ===========================================================================
with tab_compare:
    st.subheader("RFM tells you who is slipping away. CLTV tells you what that costs.")
    st.markdown(
        "Cross the two and you stop having rankings and start having a to-do list. "
        "The tiers below use the measured retention rate and the assumptions from the CLTV tab."
    )

    scored = cached_rfm()
    customers = cached_customers()
    _matrix, annual, _bands = cached_cohorts()
    gross_contribution = S.annual_contribution(customers, 0.30)
    clv = S.predictive_clv(gross_contribution, annual["retention_rate"], 0.10, 150.0)

    combined = pd.DataFrame(
        {
            "Customer ID": customers["Customer ID"],
            "Tier": S.value_tiers(clv).astype(str),
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
    st.plotly_chart(fig_cross, use_container_width=True)

    st.markdown("**The same table, read as decisions**")
    st.dataframe(S.segment_action_table(cross_tab), use_container_width=True, hide_index=True)

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
        - **Clustering** finally described the customers — and needed an entirely different
          dataset to do it, because a transaction log has no age, no income and no household
          in it.

        Three methods, three questions, three data requirements. That is the lesson.
        """
    )
