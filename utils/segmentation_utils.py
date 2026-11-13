"""
segmentation_utils.py
---------------------
UI-free, importable logic for the Customer Segmentation lesson page.

**This module imports from the sibling repo rather than re-implementing it.**
That is a deliberate departure from ``propensity_utils``/``survival_utils``,
which had to copy their siblings' logic because ``predictions/*.py`` are ``# %%``
scripts with top-level side effects and cannot be imported at all.

``segmentation/src/`` is different: ``retail_data``, ``rfm_scoring``,
``clustering_core`` and ``personality_data`` are already pure, importable,
Streamlit-free modules — exactly what a ``utils/`` module is supposed to be.
Copying ~600 lines of them here would duplicate the single source of truth and
guarantee the drift the alignment-checker exists to catch. So this module is a
thin adapter: it makes the sibling importable, re-exports its functions under one
name, and adds only what the *app* needs on top — plot-ready frames and a
diagram.

The lesson, in the order the slides tell it:
- **RFM** — aggregate a transaction log into Recency/Frequency/Monetary, score
  each 1-5, then read the segments two ways (the cube, and a business-rule ladder).
- **CLTV** — measure the retention rate from cohorts, put a euro value on every
  customer, and show how violently that value moves with the assumptions.
- **Clustering** — group customers on demographics *and* behaviour at once, then
  evaluate without labels (silhouette, stability, a 2D projection).

Two datasets, on purpose: RFM and CLTV need dated transactions, clustering needs
customer attributes. See ``dataset_roles()``.

Graceful degradation: sibling repos are git-ignored, so a fresh clone of the app
may not have ``segmentation/`` at all. ``SIBLING_AVAILABLE`` says so and the page
explains itself instead of crashing. When the repo *is* present but its data
files are not, the sibling's own loaders fall back to synthetic data and report
``source="synthetic"`` — the lesson always runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ===========================================================================
# Sibling bootstrap
# ===========================================================================

SIBLING = Path(__file__).resolve().parents[1] / "segmentation"

try:  # pragma: no cover - exercised by the missing-sibling path
    if str(SIBLING) not in sys.path:
        sys.path.insert(0, str(SIBLING))
    from src import cltv_core as _cltv
    from src import clustering_core as _clustering
    from src import personality_data as _personality
    from src import retail_data as _retail
    from src import rfm_scoring as _rfm

    SIBLING_AVAILABLE = True
    SIBLING_ERROR = ""
except ImportError as exc:  # pragma: no cover - only when the sibling is absent
    SIBLING_AVAILABLE = False
    SIBLING_ERROR = str(exc)
    _retail = _rfm = _clustering = _personality = _cltv = None


# ===========================================================================
# Re-exports — one import surface for the page
# ===========================================================================

if SIBLING_AVAILABLE:
    # A sibling checkout OLDER than this app (e.g. its develop not pulled/pushed
    # yet) imports fine but lacks newer names. Report that as "sibling
    # unavailable", with the reason, instead of crashing the page.
    try:
        # Transactions (RFM + CLTV)
        load_transactions = _retail.prepare
        customer_table = _retail.customer_table
        cohort_retention = _retail.cohort_retention
        annual_retention = _retail.annual_retention
        retention_by_recency_band = _retail.retention_by_recency_band
        is_wholesale = _retail.is_wholesale
        wholesale_profile = _retail.wholesale_profile
        clean_transactions = _retail.clean_transactions

        # RFM scoring
        build_rfm = _rfm.build_rfm
        score_rfm = _rfm.score_rfm
        rfm_cube = _rfm.rfm_cube
        rfm_ladder = _rfm.rfm_ladder
        segment_rfm = _rfm.segment_rfm
        CUBE_NAMES = _rfm.CUBE_NAMES
        LADDER_RULES = _rfm.LADDER_RULES

        # CLTV — the formulas and the lesson's business assumptions
        annual_contribution = _cltv.annual_contribution
        clv_multiplier = _cltv.clv_multiplier
        value_tiers = _cltv.clv_segments
        GROSS_MARGIN = _cltv.GROSS_MARGIN
        DISCOUNT_RATE = _cltv.DISCOUNT_RATE
        ACQUISITION_COST = _cltv.ACQUISITION_COST
        MIN_TENURE_DAYS = _cltv.MIN_TENURE_DAYS

        # Clustering
        load_personality = _personality.prepare
        CLUSTERING_FEATURES = _personality.CLUSTERING_FEATURES
        scale_features = _clustering.scale_features
        sweep_k = _clustering.sweep_k
        choose_k = _clustering.choose_k
        choose_eps = _clustering.choose_eps
        build_algorithms = _clustering.build_algorithms
        estimate_meanshift_bandwidth = _clustering.estimate_meanshift_bandwidth
        score_clustering = _clustering.score_clustering
        stability = _clustering.stability
        stability_verdict = _clustering.stability_verdict
        pca_projection = _clustering.pca_projection
        profile_clusters = _clustering.profile_clusters
        cluster_fingerprints = _clustering.cluster_fingerprints
        name_segment = _clustering.name_segment
        FAMILIES = _clustering.FAMILIES
    except AttributeError as exc:  # pragma: no cover - only with an outdated sibling
        SIBLING_AVAILABLE = False
        SIBLING_ERROR = (
            f"the segmentation sibling is older than this app expects ({exc}). "
            "Pull its branch locally, or push it so a deployment picks it up."
        )


# ===========================================================================
# What each dataset is for — the point students most often miss
# ===========================================================================


def dataset_roles() -> pd.DataFrame:
    """Which dataset feeds which method, and why it has to be that way.

    Segmentation methods do not all want the same data. RFM and CLTV need *dated
    transactions* — you cannot compute a recency without a date, or measure a
    retention rate without observing the same person twice. Clustering on "who
    the customer is" needs *customer attributes*.

    The "Blind to" column describes **these two datasets**, not transaction data
    in general: a retailer with a loyalty-card programme links receipts to
    demographics through the card, and that joined table could feed all three
    methods. The two public datasets used here are simply not linked.
    """
    return pd.DataFrame(
        [
            ("RFM", "Transaction log", "Dated purchases per customer",
             "No age, income or household in this log"),
            ("CLTV", "Transaction log", "Repeat observation over time, for retention",
             "No age, income or household in this log"),
            ("Clustering", "Customer attributes", "Demographics + behaviour per customer",
             "No dated purchases in this table"),
        ],
        columns=["Method", "Needs", "Because", "Blind to (in this lesson's data)"],
    )


def segmentation_architecture_dot() -> str:
    """A Graphviz DOT diagram of the lesson's data flow.

    ``st.graphviz_chart`` renders a raw DOT string client-side, so this needs no
    Graphviz Python package. The dashed node is the bridge a real retailer often
    has and this lesson does not: a loyalty card (or CRM login) that ties each
    receipt to a person, so one joined table could feed all three methods.
    """
    return """
digraph segmentation {
  rankdir=LR;
  node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=10];
  edge [fontname="Helvetica" fontsize=9 color="#616161"];

  subgraph cluster_data {
    label="This lesson's data"; style=dashed; color="#BDBDBD"; fontname="Helvetica"; fontsize=10;
    tx    [label="Transaction log\\nwho bought what, when" fillcolor="#2E7D32" fontcolor=white];
    attrs [label="Customer attributes\\nage, income, household" fillcolor="#2E7D32" fontcolor=white];
  }

  card [label="Loyalty card / CRM\\nlinks receipts to people\\n(not available here)"
        style="rounded,dashed" color="#757575" fontcolor="#616161"];

  clean [label="clean + B2C filter\\nsrc/retail_data.py" fillcolor="#ECEFF1"];
  agg   [label="aggregate to customers\\nbuild_rfm()" fillcolor="#ECEFF1"];
  scale [label="standardise\\nscale_features()" fillcolor="#ECEFF1"];

  rfm   [label="RFM\\nscore 1-5, cube, ladder" fillcolor="#1565C0" fontcolor=white];
  cltv  [label="CLTV\\ncohorts -> Rr -> value" fillcolor="#6A1B9A" fontcolor=white];
  clus  [label="Clustering\\n4 algorithms, k=4" fillcolor="#00838F" fontcolor=white];

  action [label="who to target,\\nhow much to spend" shape=note fillcolor="#FFF9C4"];

  tx -> clean -> agg;
  agg -> rfm;
  agg -> cltv [label="cohort retention"];
  attrs -> scale -> clus;
  tx -> card [style=dashed dir=none color="#9E9E9E"];
  card -> attrs [style=dashed dir=none color="#9E9E9E" label="join on\\ncustomer ID"];
  rfm -> action;
  cltv -> action;
  clus -> action [label="a description\\nyou can brief"];
}
"""


# ===========================================================================
# Worked examples — one customer at a time, so the numbers have a face
# ===========================================================================


def customer_orders(transactions: pd.DataFrame, customer_id) -> pd.DataFrame:
    """One customer's purchase history, one row per order (invoice).

    The raw log has one row per *product line*; RFM counts *orders*. Showing the
    collapsed table next to the R, F and M it produces is the clearest way to
    show where the three numbers come from.
    """
    rows = transactions[transactions["Customer ID"] == customer_id]
    orders = (
        rows.groupby("Invoice")
        .agg(Date=("InvoiceDate", "max"), Products=("Invoice", "size"), Amount=("Revenue", "sum"))
        .sort_values("Date")
        .reset_index()
    )
    orders["Date"] = orders["Date"].dt.date
    return orders


def example_customer(scored: pd.DataFrame, min_orders: int = 3, max_orders: int = 6) -> pd.Series:
    """A readable example: a recent Champion with a handful of orders.

    Deterministic (lowest ID that fits), so the worked example does not change
    between reruns; falls back to the median-frequency customer on odd data.
    """
    fits = scored[
        scored["Frequency"].between(min_orders, max_orders) & (scored["Segment"] == "Champions")
    ]
    if fits.empty:
        fits = scored[scored["Frequency"] == scored["Frequency"].median()]
    if fits.empty:
        fits = scored
    return fits.sort_values("Id").iloc[0]


def score_reading(scored: pd.DataFrame, customer: pd.Series) -> list[str]:
    """Explain one customer's three scores as ranks against everyone else.

    The 1-5 score is a rank, not an amount, so the honest explanation is "better
    than X% of customers", which is computed here for each dimension.
    """
    recent_share = (scored["Recency"] > customer["Recency"]).mean()
    frequent_share = (scored["Frequency"] < customer["Frequency"]).mean()
    spend_share = (scored["Monetary"] < customer["Monetary"]).mean()
    return [
        f"**R = {customer['R_Score']:.0f}**: last order {customer['Recency']:.0f} days ago — more "
        f"recent than {recent_share:.0%} of customers.",
        f"**F = {customer['F_Score']:.0f}**: {customer['Frequency']:.0f} orders — more than "
        f"{frequent_share:.0%} of customers.",
        f"**M = {customer['M_Score']:.0f}**: €{customer['Monetary']:,.0f} spent — more than "
        f"{spend_share:.0%} of customers.",
    ]


def rfm_score_cloud(scored: pd.DataFrame, jitter: float = 0.3, seed: int = 0) -> pd.DataFrame:
    """R/F/M *scores* with a little random jitter, for a readable 3D scatter.

    On the raw values the cloud is a smear: spend runs from €3 to €41,000 and
    most customers sit in one corner. On the 1-5 scores every customer lands on
    one of 125 grid points — so thousands overlap exactly. A small, seeded jitter
    spreads each grid point into a little ball, and the segments show up as
    blocks of the cube.
    """
    rng = np.random.default_rng(seed)
    cloud = scored[["Id", "Segment", "Recency", "Frequency", "Monetary"]].copy()
    for score, axis in [("R_Score", "R"), ("F_Score", "F"), ("M_Score", "M")]:
        cloud[axis] = scored[score].astype(float) + rng.uniform(-jitter, jitter, len(scored))
    return cloud


def cohort_reading(matrix: pd.DataFrame) -> str:
    """One row of the cohort heatmap read aloud, using the first full cohort."""
    cohort = matrix.index[0]
    row = matrix.iloc[0]
    month_one, month_three = row.iloc[1], row.iloc[3]
    later = row.iloc[1:13].dropna()
    return (
        f"Read the top row: of everyone whose **first** order was in {cohort}, "
        f"{month_one:.0%} ordered again the next month and {month_three:.0%} in month 3. "
        f"Over the first year it hovers between {later.min():.0%} and {later.max():.0%} each "
        "month — not the same people every month, but a steady core that keeps coming back."
    )


def clv_year_by_year(gc: float, retention_rate: float, discount_rate: float,
                     years: int = 5) -> pd.DataFrame:
    """The lifetime-value formula unrolled year by year, for one customer.

    Year t contributes ``GC * Rr**t / (1+d)**t``: the contribution, times the
    chance they are still a customer, shrunk to today's money. Summing *every*
    year gives exactly ``GC * Rr / (1 - Rr + d)`` — the closed form on the
    slide. Showing the first few rows makes the formula a story instead of a
    fraction.
    """
    t = np.arange(1, years + 1)
    still_here = retention_rate ** t
    today_factor = 1 / (1 + discount_rate) ** t
    return pd.DataFrame(
        {
            "Year": t,
            "Chance still a customer": still_here,
            "€1 then is worth today": today_factor,
            "Expected value today (€)": gc * still_here * today_factor,
        }
    )


def behavioural_retention(recency: pd.Series, band_retention: pd.DataFrame) -> pd.Series:
    """Give each customer the retention rate measured for their recency band.

    Mirrors ``segmentation/cltv.py`` section 7: the band edges are read back off
    the measured table (labels like ``"30-89d"``), so the two cannot drift
    apart, and the last band is stretched to cover today's longest-quiet
    customer.
    """
    edges = [0] + [int(label.split("-")[1].rstrip("d")) + 1 for label in band_retention.index]
    edges[-1] = max(edges[-1], int(recency.max()) + 1)
    return pd.cut(
        recency, bins=edges, labels=band_retention["RetentionRate"].to_numpy(), right=False,
        ordered=False,
    ).astype(float)


# ===========================================================================
# Plot-ready frames (the app's own additions)
# ===========================================================================


def cohort_matrix_frame(matrix: pd.DataFrame, max_periods: int = 13) -> pd.DataFrame:
    """The cohort retention matrix as percentages, trimmed for display.

    The raw matrix is 25x25 of fractions with a triangular block of NaN. This
    trims it to the periods worth showing and converts to percentages, leaving
    the NaNs *as* NaNs — they are unobserved, not zero, and a heatmap that paints
    them as 0% invents a collapse in retention that never happened.
    """
    trimmed = matrix.iloc[:, :max_periods] * 100
    trimmed.index = trimmed.index.astype(str)
    return trimmed.round(1)


def retention_curve_frame(matrix: pd.DataFrame, max_periods: int = 13) -> pd.DataFrame:
    """Long-form retention curves: one row per (cohort, period), plus an average.

    Shaped for ``px.line(..., color="Cohort")``. Period 0 is dropped: it is 100%
    by construction and its inclusion squashes the interesting part of the axis.
    """
    trimmed = matrix.iloc[:, 1:max_periods]
    long = (
        trimmed.stack()
        .rename("Retention")
        .reset_index()
        .rename(columns={"level_1": "Period"})
    )
    long["Cohort"] = long["Cohort"].astype(str)
    long["Retention"] *= 100

    average = trimmed.mean(axis=0)
    mean_rows = pd.DataFrame(
        {"Cohort": "average", "Period": average.index, "Retention": average.to_numpy() * 100}
    )
    return pd.concat([long, mean_rows], ignore_index=True)


def sensitivity_frame(
    retention_rates: list[float], discount_rates: list[float]
) -> pd.DataFrame:
    """The lifetime multiplier across a grid of both assumptions.

    The single most important table in the CLTV lesson: it shows that the answer
    is driven far more by what you assumed than by anything in the data.
    """
    return pd.DataFrame(
        {
            f"d={d:.0%}": [clv_multiplier(rr, d) for rr in retention_rates]
            for d in discount_rates
        },
        index=[f"Rr={rr:.0%}" for rr in retention_rates],
    ).round(2)


def equity_curve(
    gross_contribution: pd.Series,
    retention_rates: np.ndarray,
    discount_rate: float,
    acquisition_cost: float,
) -> pd.DataFrame:
    """Total customer equity as the retention assumption varies.

    Drives the page's headline interaction: drag Rr and watch the value of the
    entire customer base swing. Vectorised over the grid, and it never refits
    anything — the per-customer GC is fixed, only the multiplier moves — so the
    slider stays instant.
    """
    total_gc = float(gross_contribution.sum())
    n_customers = len(gross_contribution)
    multipliers = np.array([clv_multiplier(rr, discount_rate) for rr in retention_rates])
    return pd.DataFrame(
        {
            "RetentionRate": retention_rates,
            "Multiplier": multipliers,
            "CustomerEquity": total_gc * multipliers - acquisition_cost * n_customers,
        }
    )


def predictive_clv(
    gross_contribution: pd.Series,
    retention_rate: float | pd.Series,
    discount_rate: float,
    acquisition_cost: float,
) -> pd.Series:
    """``LTV_i = GC_i * (Rr / (1 - Rr + d)) - AC_i`` — the lecture's closed form.

    Takes GC as a Series (rather than the customer table the lesson script uses)
    so the page can recompute it from slider values; the multiplier itself is the
    sibling's ``cltv_core.clv_multiplier``, which accepts a scalar or a Series.
    """
    return gross_contribution * clv_multiplier(retention_rate, discount_rate) - acquisition_cost


def wholesale_comparison(transactions: pd.DataFrame) -> pd.DataFrame:
    """What the B2C filter removes, and what it does to the RFM signal.

    Takes an **unfiltered** log and reports consumers vs businesses side by side.
    The filter is a judgement encoded as a rule, so the page shows its cost
    rather than applying it silently — and the improvement in the recency
    correlation is the argument for it.
    """
    flags = is_wholesale(transactions)
    business_ids = set(flags.index[flags])
    is_business = transactions["Customer ID"].isin(business_ids)

    rows = []
    for label, subset in [
        ("Consumers (B2C)", transactions[~is_business]),
        ("Businesses", transactions[is_business]),
    ]:
        rfm = build_rfm(subset)
        correlations = rfm[["Recency", "Frequency", "Monetary"]].corr()
        rows.append(
            {
                "Population": label,
                "Customers": subset["Customer ID"].nunique(),
                "Line items": len(subset),
                "Revenue": subset["Revenue"].sum(),
                "Median order value": subset.groupby("Invoice")["Revenue"].sum().median(),
                "Median line quantity": subset["Quantity"].median(),
                "Recency vs Frequency": correlations.loc["Recency", "Frequency"],
            }
        )
    frame = pd.DataFrame(rows)
    frame["Share of revenue"] = frame["Revenue"] / frame["Revenue"].sum()
    return frame


def k_sweep_frame(sweep: pd.DataFrame) -> pd.DataFrame:
    """Normalise the k-sweep metrics onto one 0-1 axis so they can share a chart.

    Inertia and Calinski-Harabasz live on wildly different scales from the
    silhouette, and Davies-Bouldin points the other way (lower is better). This
    rescales each to 0-1 and **flips** Davies-Bouldin, so on the resulting chart
    "up" always means "better" and the curves are comparable at a glance.
    """
    out = sweep[["k"]].copy()
    for column, higher_is_better in [
        ("silhouette", True),
        ("calinski_harabasz", True),
        ("davies_bouldin", False),
        ("inertia", False),
    ]:
        values = sweep[column]
        span = values.max() - values.min()
        scaled = (values - values.min()) / span if span else values * 0
        out[column] = scaled if higher_is_better else 1 - scaled
    return out


def pca_frame(
    coords: np.ndarray, labels: np.ndarray, customers: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Plot-ready 2D projection: coordinates plus a readable cluster label."""
    frame = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1]})
    frame["Cluster"] = ["Noise" if v == -1 else f"Cluster {v}" for v in labels]
    if customers is not None:
        for column in ("Income", "Age", "Total_Spending"):
            if column in customers.columns:
                frame[column] = customers[column].to_numpy()
    return frame


def stability_frame(algorithms: dict, X: np.ndarray, labels: dict) -> pd.DataFrame:
    """Adjusted Rand index for each algorithm under resampling, with a verdict.

    The check most reporting skips: if dropping a random fifth of the customers
    reshuffles the segments, the segments were never there — and no silhouette
    score would have told you. The reverse holds too: an algorithm can find the
    same *meaningless* partition every time, so the verdict (the sibling's
    ``stability_verdict``) reads stability together with the silhouette.
    """
    rows = []
    for name, spec in algorithms.items():
        mean_ari, std_ari = stability(spec["build"], X, labels[name])
        silhouette = score_clustering(X, labels[name])["silhouette"]
        rows.append(
            {
                "Algorithm": name,
                "Family": spec["family"],
                "ARI": mean_ari,
                "Std": std_ari,
                "Silhouette": silhouette,
                "Verdict": stability_verdict(mean_ari, silhouette),
            }
        )
    return pd.DataFrame(rows)


def segment_action_table(cross_tab: pd.DataFrame) -> pd.DataFrame:
    """Turn the RFM x CLTV cross-tab into a decision table with recommended actions.

    The slides' payoff (RFM & CLTV together): RFM says who is slipping away, CLTV
    says how much that would cost. Crossing them turns two rankings into a list
    of things to actually do.
    """
    actions = {
        ("Can't Lose Them", "Premium"): "Urgent: high value, walking out — intervene now",
        ("Can't Lose Them", "High Value"): "Urgent: personal contact before they are gone",
        ("At Risk", "Premium"): "Win back with a targeted offer",
        ("At Risk", "High Value"): "Reactivation offer",
        ("Champions", "Premium"): "Protect: reward and keep them",
        ("Champions", "High Value"): "Protect: loyalty programme",
        ("Loyal Customers", "Premium"): "Grow: upsell, they can afford it",
        ("Lost", "Low Value"): "Let go — cheapest thing to stop spending on",
        ("Hibernating", "Low Value"): "Low-cost reactivation only",
        ("Recent Customers", "Low Value"): "Nurture: too early to judge",
    }
    rows = []
    for segment in cross_tab.index:
        for tier in cross_tab.columns:
            count = int(cross_tab.loc[segment, tier])
            action = actions.get((segment, str(tier)))
            if count and action:
                rows.append({"RFM segment": segment, "CLTV tier": str(tier),
                             "Customers": count, "Recommended action": action})
    return pd.DataFrame(rows).sort_values("Customers", ascending=False).reset_index(drop=True)


# ===========================================================================
# Clustering — sizes and plain-language profiles
# ===========================================================================

FEATURE_LABELS = {
    "Income": ("income", "{:,.0f}"),
    "Age": ("age", "{:.0f}"),
    "Total_Children": ("children at home", "{:.1f}"),
    "Total_Spending": ("spend over two years", "{:,.0f}"),
    "Total_Purchases": ("purchases", "{:.0f}"),
    "Total_Campaigns_Accepted": ("campaigns accepted (of 5)", "{:.1f}"),
    "NumWebVisitsMonth": ("web visits a month", "{:.1f}"),
}


def cluster_size_frame(labels: dict) -> pd.DataFrame:
    """Cluster sizes per algorithm, as shares — one row per (algorithm, cluster).

    Sorted largest first within each algorithm so a stacked bar reads left to
    right from the biggest group, and DBSCAN's noise (``-1``) is kept as its own
    "Noise" bar rather than hidden.
    """
    rows = []
    for name, lab in labels.items():
        counts = pd.Series(lab).value_counts()
        noise = counts.pop(-1) if -1 in counts.index else 0
        for rank, (cluster, n) in enumerate(counts.sort_values(ascending=False).items()):
            rows.append({"Algorithm": name, "Cluster": f"Cluster {cluster}", "Rank": rank,
                         "Customers": int(n), "Share": n / len(lab)})
        if noise:
            rows.append({"Algorithm": name, "Cluster": "Noise", "Rank": len(counts),
                         "Customers": int(noise), "Share": noise / len(lab)})
    return pd.DataFrame(rows)


def cluster_size_reading(sizes: pd.DataFrame) -> list[str]:
    """One plain sentence per algorithm: how balanced are its groups?"""
    lines = []
    for name, group in sizes.groupby("Algorithm", sort=False):
        clusters = group[group["Cluster"] != "Noise"]
        largest, smallest = clusters["Share"].max(), clusters["Share"].min()
        noise = group.loc[group["Cluster"] == "Noise", "Share"].sum()
        noise_text = f", and leaves {noise:.0%} unassigned as noise" if noise else ""
        if largest >= 0.6:
            shape = f"one giant group holds {largest:.0%} of customers"
        elif largest / smallest <= 4:
            shape = "groups of broadly similar size"
        else:
            shape = f"very uneven groups ({largest:.0%} down to {smallest:.1%})"
        lines.append(f"**{name}**: {len(clusters)} clusters — {shape}{noise_text}.")
    return lines


def describe_cluster(profile: pd.Series, fingerprint: pd.Series, top: int = 3) -> str:
    """A cluster in plain words: its most distinctive features, with real values.

    Picks the ``top`` features furthest from average (by z-score) and reports the
    cluster's actual average next to "above/below average", which is what a
    marketer can use — "income 80,500, well above average" — rather than "+1.33".
    """
    strongest = fingerprint.abs().sort_values(ascending=False).index[:top]
    parts = []
    for feature in strongest:
        z = fingerprint[feature]
        if abs(z) < 0.3:
            continue
        label, fmt = FEATURE_LABELS.get(feature, (feature, "{:.1f}"))
        size = "well " if abs(z) >= 1 else ""
        direction = "above" if z > 0 else "below"
        parts.append(f"{label} {fmt.format(profile[feature])} ({size}{direction} average)")
    return "; ".join(parts) if parts else "close to the average customer on everything"


# ===========================================================================
# Acting on it — the business case for one campaign
# ===========================================================================


def campaign_value(future_value: pd.Series, cost_per_customer: float,
                   success_rate: float) -> dict:
    """Is a retention campaign for this group worth it? Expected value in euros.

    ``future_value`` is each customer's expected future contribution
    (GC × multiplier — the acquisition cost is already spent, so it is left
    out). If the campaign keeps ``success_rate`` of customers who would
    otherwise have left, it saves that share of the group's future value, and
    costs ``cost_per_customer`` for everyone contacted.
    """
    customers = len(future_value)
    cost = customers * cost_per_customer
    saved = success_rate * float(future_value.sum())
    return {
        "customers": customers,
        "cost": cost,
        "saved": saved,
        "net": saved - cost,
        "return_per_euro": saved / cost if cost else float("nan"),
        "break_even_rate": cost / float(future_value.sum()) if future_value.sum() else float("nan"),
    }
