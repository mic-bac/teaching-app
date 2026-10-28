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
    from src import clustering_core as _clustering
    from src import personality_data as _personality
    from src import retail_data as _retail
    from src import rfm_scoring as _rfm

    SIBLING_AVAILABLE = True
    SIBLING_ERROR = ""
except ImportError as exc:  # pragma: no cover - only when the sibling is absent
    SIBLING_AVAILABLE = False
    SIBLING_ERROR = str(exc)
    _retail = _rfm = _clustering = _personality = None


# ===========================================================================
# Re-exports — one import surface for the page
# ===========================================================================

if SIBLING_AVAILABLE:
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
    pca_projection = _clustering.pca_projection
    profile_clusters = _clustering.profile_clusters
    cluster_fingerprints = _clustering.cluster_fingerprints
    name_segment = _clustering.name_segment
    FAMILIES = _clustering.FAMILIES


# ===========================================================================
# What each dataset is for — the point students most often miss
# ===========================================================================


def dataset_roles() -> pd.DataFrame:
    """Which dataset feeds which method, and why it has to be that way.

    Segmentation methods do not all want the same data. RFM and CLTV need *dated
    transactions* — you cannot compute a recency without a date, or measure a
    retention rate without observing the same person twice. Clustering needs
    *customer attributes* — a till receipt records a price and a product code,
    never an age, so no volume of transaction data will ever tell you a segment
    is "affluent households without children".

    Teaching this as a table beats asserting it: the empty cells are the lesson.
    """
    return pd.DataFrame(
        [
            ("RFM", "Transaction log", "Dated purchases per customer",
             "Cannot see who the customer is"),
            ("CLTV", "Transaction log", "Repeat observation over time, for retention",
             "Cannot see who the customer is"),
            ("Clustering", "Customer attributes", "Demographics + behaviour per customer",
             "Cannot see when anything happened"),
        ],
        columns=["Method", "Needs", "Because", "Blind to"],
    )


def segmentation_architecture_dot(
    data_source: str = "retail", personality_source: str = "kaggle"
) -> str:
    """A Graphviz DOT diagram of the lesson's data flow.

    ``st.graphviz_chart`` renders a raw DOT string client-side, so this needs no
    Graphviz Python package. Colours reflect which data source is actually live,
    so the diagram tells the truth about a synthetic fallback rather than drawing
    a picture of the happy path regardless.
    """
    retail_colour = "#2E7D32" if data_source == "retail" else "#EF6C00"
    retail_label = (
        "Online Retail II\\n1.07M invoice lines" if data_source == "retail"
        else "SYNTHETIC\\ntransaction log"
    )
    personality_colour = "#2E7D32" if personality_source == "kaggle" else "#EF6C00"
    personality_label = (
        "Customer Personality\\n2,240 customers" if personality_source == "kaggle"
        else "SYNTHETIC\\ncustomer attributes"
    )

    return f"""
digraph segmentation {{
  rankdir=LR;
  node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=10];
  edge [fontname="Helvetica" fontsize=9 color="#616161"];

  subgraph cluster_data {{
    label="Data"; style=dashed; color="#BDBDBD"; fontname="Helvetica"; fontsize=10;
    tx    [label="{retail_label}" fillcolor="{retail_colour}" fontcolor=white];
    attrs [label="{personality_label}" fillcolor="{personality_colour}" fontcolor=white];
  }}

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
  rfm -> action;
  cltv -> action;
  clus -> action [label="a description\\nyou can brief"];
}}
"""


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


def clv_multiplier(retention_rate: float, discount_rate: float) -> float:
    """``Rr / (1 - Rr + d)`` — how many years of contribution a customer is worth.

    The heart of the lecture's closed form. Kept here (rather than imported) so
    the page can call it on slider values without touching the lesson script's
    module-level constants.
    """
    return retention_rate / (1 - retention_rate + discount_rate)


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


def annual_contribution(
    customers: pd.DataFrame, gross_margin: float, min_tenure_days: int = 365
) -> pd.Series:
    """GC_i — yearly gross contribution, annualised over observed tenure.

    ``min_tenure_days`` is the modelling decision that matters most here: a
    customer who ordered once has an observed tenure of zero days. Flooring at a
    year says "we will not annualise from less than a year of history"; flooring
    at a month would record their single basket as twelve baskets a year and
    promote one-time buyers into the most valuable segment on the page.
    """
    observed_years = np.maximum(customers["TenureDays"], min_tenure_days) / 365.25
    return customers["Revenue"] / observed_years * gross_margin


def predictive_clv(
    gross_contribution: pd.Series,
    retention_rate: float | pd.Series,
    discount_rate: float,
    acquisition_cost: float,
) -> pd.Series:
    """``LTV_i = GC_i * (Rr / (1 - Rr + d)) - AC_i`` — the lecture's closed form."""
    if isinstance(retention_rate, pd.Series):
        multiplier = retention_rate.map(lambda rr: clv_multiplier(rr, discount_rate))
    else:
        multiplier = clv_multiplier(retention_rate, discount_rate)
    return gross_contribution * multiplier - acquisition_cost


def value_tiers(clv: pd.Series) -> pd.Series:
    """Cut customers into four value tiers by CLTV quartile."""
    return pd.qcut(clv, 4, labels=["Low Value", "Medium Value", "High Value", "Premium"])


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
    score would have told you.
    """
    rows = []
    for name, spec in algorithms.items():
        mean_ari, std_ari = stability(spec["build"], X, labels[name])
        rows.append(
            {
                "Algorithm": name,
                "Family": spec["family"],
                "ARI": mean_ari,
                "Std": std_ari,
                "Verdict": (
                    "solid" if mean_ari > 0.75 else "shaky" if mean_ari > 0.5 else "unreliable"
                ),
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
        ("At Risk", "Premium"): "Urgent: high value, walking out — intervene now",
        ("At Risk", "High Value"): "Win back with a targeted offer",
        ("Champions", "Premium"): "Protect: reward and keep them",
        ("Champions", "High Value"): "Protect: loyalty programme",
        ("Can't Lose Them", "Premium"): "Urgent: personal contact",
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
