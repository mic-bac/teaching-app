"""
5_Survival.py
-------------
Survival Analysis lesson page (time-to-event churn).

Surfaces the sibling ``predictions/survival.py`` script as an interactive lesson:
censoring, the Kaplan-Meier survival function, three survival models (Cox,
Random Survival Forest, Survival SVM), the metrics censoring forces us to use
(c-index, time-dependent AUC, integrated Brier score), individual survival curves
and risk strata. All logic lives in ``utils/survival_utils`` (UI-free); this page
only caches, renders, and shows the real source code that runs.

Graceful behavior: the data comes from ``utils.propensity_utils.load_churn``,
which falls back to a synthetic customer base when the sibling repo's git-ignored
CSVs are missing. Sample sizes are kept small on purpose — Cox does a full
likelihood optimisation and the SVM solves a ranking problem over pairs, so both
scale worse than a classifier.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import propensity_utils as P
from utils import survival_utils as S
from utils.teaching import source_of

st.set_page_config(page_title="Survival Analysis", layout="wide", page_icon="⏳")

ROOT = Path(__file__).resolve().parents[1]
SIBLING = ROOT / "predictions"


def show_source(obj, label: str) -> None:
    """Reveal the real function/class that runs, so code can't drift from behavior."""
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
# Cached data / model builders (keep utils Streamlit-free; cache here)
# ===========================================================================
@st.cache_data(show_spinner=False)
def cached_churn(sample_size: int):
    return P.load_churn(sample_size=sample_size)


@st.cache_resource(show_spinner=False)
def cached_survival(sample_size: int):
    frame, _ = cached_churn(sample_size)
    return S.prepare_survival(frame)


@st.cache_resource(show_spinner=False)
def cached_split(sample_size: int):
    X, y, names = cached_survival(sample_size)
    X_train, X_test, y_train, y_test = S.split_survival(X, y)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": names,
    }


@st.cache_resource(show_spinner=False)
def cached_model(model_name: str, sample_size: int):
    split = cached_split(sample_size)
    return S.MODEL_BUILDERS[model_name](split["X_train"], split["y_train"])


# ===========================================================================
# Header + the one control every tab depends on
# ===========================================================================
st.title("⏳ Survival Analysis — how long do our customers stay?")
st.markdown(
    """
    Classification answers *whether* a customer churns. Survival analysis answers
    ***when*** — and, crucially, it uses the customers who **have not churned yet**
    instead of throwing them away or mislabelling them as "negative".

    That single idea, **censoring**, is what makes this its own discipline: it changes
    how models are fitted *and* how they are graded.

    Every tab shows the **real code** that produces the result, not a slide summary.
    """
)

control, status = st.columns([2, 3])
with control:
    sample_size = st.select_slider(
        "Customers to model",
        options=[1_000, 2_000, 3_000, 5_000, 10_000],
        value=3_000,
        help="Survival models scale worse than classifiers, so this lesson samples smaller than the propensity one.",
    )
with status:
    churn_df, data_source = cached_churn(sample_size)
    if data_source == "kaggle":
        st.success(f"Kaggle churn dataset — {len(churn_df):,} customers sampled from the sibling `predictions/` repo.")
    else:
        st.warning(
            "The Kaggle CSVs were not found in `predictions/data/churn/`, so this lesson is running on a "
            "**synthetic** customer base with the same columns. Everything still works; the numbers are simulated."
        )

split = cached_split(sample_size)

tab_overview, tab_km, tab_models, tab_eval, tab_individual = st.tabs(
    [
        "🧭 Overview & Censoring",
        "📉 Kaplan-Meier",
        "🧮 Models",
        "📏 Evaluation",
        "👤 Curves & Risk Strata",
    ]
)

# ===========================================================================
# TAB 1 — Overview and censoring
# ===========================================================================
with tab_overview:
    st.subheader("Two ways to answer one question")
    st.graphviz_chart(S.survival_taxonomy_dot(), use_container_width=True)

    st.subheader("Censoring: the whole point")
    st.markdown(
        """
        Survival data needs exactly two target columns:

        | Column | Meaning |
        |---|---|
        | `event` | did churn actually happen while we were watching? (1 = yes, 0 = **censored**) |
        | `time` | how long we watched this customer (their tenure in months) |

        `event = 0` does **not** mean "this customer will never churn". It means "as of the
        cut-off date they were still here". That is real information — they survived at
        least `time` months — and Kaplan-Meier uses exactly that.

        Get this wrong and you bias everything: drop censored customers and churn looks
        catastrophic; label them "stayed" and it looks harmless.
        """
    )

    example = S.censoring_example()
    cutoff = example.attrs["cutoff"]

    fig_cens = go.Figure()
    for _, row in example.iterrows():
        fig_cens.add_trace(
            go.Scatter(
                x=[row["signup"], row["observed_until"]],
                y=[row["customer"], row["customer"]],
                mode="lines+markers",
                line=dict(color="#455a64", width=3),
                marker=dict(
                    size=[9, 14],
                    symbol=["circle", "x" if row["event"] else "circle-open"],
                    color="#c62828" if row["event"] else "#1565c0",
                ),
                showlegend=False,
                hovertemplate=f"{row['customer']}<br>observed {row['duration']} months<br>{row['status']}<extra></extra>",
            )
        )
    fig_cens.add_vline(x=cutoff, line_dash="dash", line_color="#c62828", annotation_text="study cut-off")
    fig_cens.update_layout(
        title="<b>Calendar time — everyone signs up at a different moment</b>",
        xaxis_title="Calendar month",
        height=330,
        margin=dict(t=50, b=10),
    )
    st.plotly_chart(fig_cens, use_container_width=True)

    fig_shift = go.Figure()
    for _, row in example.iterrows():
        fig_shift.add_trace(
            go.Scatter(
                x=[0, row["duration"]],
                y=[row["customer"], row["customer"]],
                mode="lines+markers",
                line=dict(color="#455a64", width=3),
                marker=dict(
                    size=[9, 14],
                    symbol=["circle", "x" if row["event"] else "circle-open"],
                    color="#c62828" if row["event"] else "#1565c0",
                ),
                showlegend=False,
                hovertemplate=f"{row['customer']}<br>{row['duration']} months<br>{row['status']}<extra></extra>",
            )
        )
    fig_shift.update_layout(
        title="<b>…so we reset every clock to t₀ = signup. ✕ = churned, ○ = still active (censored)</b>",
        xaxis_title="Months since signup",
        height=330,
        margin=dict(t=50, b=10),
    )
    st.plotly_chart(fig_shift, use_container_width=True)

    st.dataframe(example, use_container_width=True, hide_index=True)

    X, y, feature_names = cached_survival(sample_size)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(y):,}")
    c2.metric("Churned (event = 1)", f"{int(y['event'].sum()):,}")
    c3.metric("Censored (event = 0)", f"{int((~y['event']).sum()):,}")
    c4.metric("Observed months", f"{y['time'].min():.0f}–{y['time'].max():.0f}")

    st.info(
        "`Tenure` must **not** be a feature here: it *is* the time we are modelling. "
        "Feeding it in is textbook target leakage — see `EXCLUDED_FEATURES`."
    )

    show_source(S.prepare_survival, "🐍 Turning a customer table into survival data")
    show_source(S.censoring_example, "🐍 The censoring illustration above")
    show_sibling_script("survival.py", "📄 The full sibling script")

# ===========================================================================
# TAB 2 — Kaplan-Meier
# ===========================================================================
with tab_km:
    st.subheader("The survival function, estimated from data alone")
    st.latex(r"S(t) = P(T > t) \qquad\qquad \hat{S}(t) = \prod_{t_i \le t}\left(1 - \frac{d_i}{n_i}\right)")
    st.markdown(
        """
        At every time where churn happens, Kaplan-Meier multiplies in the fraction that
        survived it: $d_i$ customers churned at $t_i$, $n_i$ were still **at risk** just
        before. Censored customers stay in $n_i$ until they leave the study — that is
        precisely how their partial information gets used. No distribution is assumed;
        the curve is a staircase that only steps down where something actually happened.
        """
    )

    km = S.kaplan_meier(churn_df["Tenure"], churn_df["Churn"])

    fig_km = go.Figure()
    fig_km.add_trace(
        go.Scatter(
            x=np.concatenate([km["time"], km["time"][::-1]]),
            y=np.concatenate([km["upper"], km["lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(21,101,192,0.15)",
            line=dict(width=0),
            name="95% confidence",
            hoverinfo="skip",
        )
    )
    fig_km.add_trace(
        go.Scatter(
            x=km["time"], y=km["survival"], mode="lines", line=dict(color="#1565c0", width=3, shape="hv"), name="S(t)"
        )
    )
    fig_km.update_layout(
        title="<b>Share of customers still with us after t months</b>",
        xaxis_title="Months since signup",
        yaxis_title="S(t)",
        yaxis_range=[0, 1],
        height=470,
    )
    st.plotly_chart(fig_km, use_container_width=True)

    readouts = st.columns(4)
    for column, month in zip(readouts, (6, 12, 24, 36)):
        if month <= km["time"].max():
            value = km.loc[km["time"] <= month, "survival"].iloc[-1]
            column.metric(f"S({month} months)", f"{value * 100:.1f}%")

    st.divider()
    st.subheader("Comparing segments")
    st.markdown(
        "Splitting the curve shows *where* groups differ, not just that their averages do. "
        "A gap that opens early is an onboarding problem; one that opens late is a loyalty problem."
    )

    group_col = st.selectbox(
        "Split by", ["Contract Length", "Subscription Type", "Gender"], key="km_group"
    )
    grouped = S.km_by_group(churn_df, group_col)
    fig_group = px.line(
        grouped,
        x="time",
        y="survival",
        color=group_col,
        line_shape="hv",
        title=f"<b>Survival by {group_col.lower()}</b>",
        labels={"time": "Months since signup", "survival": "S(t)"},
    )
    fig_group.update_layout(height=470, yaxis_range=[0, 1])
    st.plotly_chart(fig_group, use_container_width=True)

    show_source(S.kaplan_meier, "🐍 The Kaplan-Meier estimator")
    show_source(S.km_by_group, "🐍 One curve per segment")

# ===========================================================================
# TAB 3 — Models
# ===========================================================================
with tab_models:
    st.subheader("Three ways to model time-to-event")
    st.table(pd.DataFrame(S.MODEL_TABLE))

    st.markdown(
        """
        Note the last column. The Survival SVM learns a **ranking** of risk, so it can tell
        you *who* churns first but not *what* their survival probability is. That is not a
        bug to hide — it decides which metrics apply to it, as the Evaluation tab shows.
        """
    )

    if st.button("🏋️ Train all three models"):
        with st.spinner(f"Fitting three survival models on {len(split['X_train']):,} customers…"):
            rows = []
            for name in S.MODEL_BUILDERS:
                model = cached_model(name, sample_size)
                rows.append(
                    {
                        "Model": name,
                        "Train c-index": S.c_index(model, split["X_train"], split["y_train"]),
                        "Test c-index": S.c_index(model, split["X_test"], split["y_test"]),
                    }
                )
        frame = pd.DataFrame(rows)
        frame["Gap (overfit)"] = frame["Train c-index"] - frame["Test c-index"]
        st.session_state["surv_cindex"] = frame

    if "surv_cindex" in st.session_state:
        frame = st.session_state["surv_cindex"]

        st.markdown(
            """
            **Concordance index** — of all customer pairs whose true order we know, how many
            does the model rank correctly? *"A churns at 3 months, B at 10, so A should get the
            higher risk."* 0.5 is coin-flipping, 1.0 is perfect. It is the survival analogue of
            ROC-AUC, and it needs nothing but a risk score — which is why it is the one metric
            **every** model here can be graded on.
            """
        )

        fig_c = go.Figure()
        fig_c.add_trace(go.Bar(x=frame["Model"], y=frame["Train c-index"], name="Train", marker_color="#90caf9"))
        fig_c.add_trace(go.Bar(x=frame["Model"], y=frame["Test c-index"], name="Test", marker_color="#1565c0"))
        fig_c.update_layout(
            title="<b>Concordance index — train vs test</b>", barmode="group", yaxis_range=[0.5, 1.0], height=430
        )
        st.plotly_chart(fig_c, use_container_width=True)

        st.dataframe(frame.round(4), use_container_width=True, hide_index=True)
        best = frame.loc[frame["Test c-index"].idxmax(), "Model"]
        st.success(f"Best ranking on held-out customers: **{best}**")
        st.caption(
            "A large train-test gap means the model memorised the training customers. "
            "Tree ensembles show the biggest gap here — that is what their flexibility costs."
        )
    else:
        st.info("Click **🏋️ Train all three models** — Cox, a survival forest and a ranking SVM, on the current sample.")

    show_source(S.fit_cox, "🐍 Cox proportional hazards")
    show_source(S.fit_rsf, "🐍 Random Survival Forest")
    show_source(S.fit_svm, "🐍 Survival SVM")
    show_source(S.c_index, "🐍 Concordance index")

# ===========================================================================
# TAB 4 — Evaluation over time
# ===========================================================================
with tab_eval:
    st.subheader("Why survival needs its own metrics")
    st.markdown(
        """
        Censoring makes MSE/RMSE inapplicable: for a censored customer the true churn time is
        unknown, so there is no error to square. Three metrics take their place.

        | Metric | Question | Needs |
        |---|---|---|
        | **c-index** | Is the *order* right? | a risk score |
        | **AUC(t)** | At month *t*, does it separate churners from stayers? | a risk score |
        | **Integrated Brier score** | Are the predicted *probabilities* right, over time? | a survival function S(t) |

        The c-index is one number for the whole horizon — but two models with the same
        c-index can behave completely differently at month 6 versus month 36, and campaign
        planning cares about exactly that difference.
        """
    )

    times, in_window = S.evaluation_window(split["y_train"], split["y_test"])
    st.info(
        f"**Evaluation window.** Both metrics reweight observations by the inverse probability of "
        f"still being uncensored, G(t) — and G hits zero at the end of the follow-up, where nobody "
        f"is left to observe. So we grade strictly inside the observed window: "
        f"{int(in_window.sum()):,} of {len(split['y_test']):,} test customers, at "
        f"t = {', '.join(f'{t:.0f}' for t in times)} months."
    )

    if st.button("📏 Score all three models over time"):
        X_window = split["X_test"][in_window]
        y_window = split["y_test"][in_window]
        auc_frames, brier_rows = [], []
        with st.spinner("Computing AUC(t) and Brier scores…"):
            for name in S.MODEL_BUILDERS:
                model = cached_model(name, sample_size)
                auc_frame, auc_mean = S.time_dependent_auc(model, split["y_train"], y_window, X_window, times)
                auc_frame["Model"] = f"{name} (mean {auc_mean:.3f})"
                auc_frames.append(auc_frame)
                brier_rows.append(
                    {"Model": name, "Integrated Brier score": S.integrated_brier(model, split["y_train"], y_window, X_window, times)}
                )
        st.session_state["surv_eval"] = (pd.concat(auc_frames, ignore_index=True), brier_rows)

    if "surv_eval" in st.session_state:
        auc_all, brier_rows = st.session_state["surv_eval"]

        fig_auc = px.line(
            auc_all,
            x="time",
            y="auc",
            color="Model",
            markers=True,
            title="<b>Time-dependent AUC — discrimination is not constant over time</b>",
            labels={"time": "Months since signup", "auc": "AUC(t)"},
        )
        fig_auc.add_hline(y=0.5, line_dash="dot", line_color="gray", annotation_text="chance")
        fig_auc.update_layout(height=450)
        st.plotly_chart(fig_auc, use_container_width=True)

        st.markdown("**Integrated Brier score** — lower is better; ~0.25 is uninformative.")
        numeric = [row for row in brier_rows if not isinstance(row["Integrated Brier score"], str)]
        if numeric:
            fig_brier = px.bar(
                pd.DataFrame(numeric),
                x="Model",
                y="Integrated Brier score",
                text_auto=".4f",
                color="Model",
            )
            fig_brier.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_brier, use_container_width=True)

        for row in brier_rows:
            value = row["Integrated Brier score"]
            if isinstance(value, str):
                st.warning(f"**{row['Model']}** — {value}")

        st.caption(
            "Unlike the c-index, the Brier score grades *calibration* as well as discrimination: "
            "a model can rank customers perfectly and still claim the wrong probabilities."
        )
    else:
        st.info("Click **📏 Score all three models over time** to compute AUC(t) and the Brier scores.")

    show_source(S.evaluation_window, "🐍 Choosing the evaluation window")
    show_source(S.time_dependent_auc, "🐍 Time-dependent AUC")
    show_source(S.integrated_brier, "🐍 Integrated Brier score (and when it does not apply)")

# ===========================================================================
# TAB 5 — Individual curves and risk strata
# ===========================================================================
with tab_individual:
    st.subheader("One customer, one whole curve")
    st.markdown(
        "This is what a classifier cannot give you. *\"80% chance of still being here in 6 months, "
        "45% in 24\"* turns into a concrete question: when is an intervention worth its cost?"
    )

    curve_model_name = st.selectbox("Model", list(S.MODEL_BUILDERS), key="curve_model_surv")
    n_customers = st.slider("Customers to draw", 1, 6, 3)
    model = cached_model(curve_model_name, sample_size)

    indices = list(range(n_customers))
    labels = [
        f"Customer {i} — {'churned' if split['y_test'][i]['event'] else 'still active'} at {split['y_test'][i]['time']:.0f}m"
        for i in indices
    ]
    curves = S.individual_survival_curves(model, split["X_test"], indices, labels)

    if isinstance(curves, str):
        st.warning(curves)
        st.caption(
            "This is the slide's point made concrete: a ranking model orders customers but has no S(t) "
            "to draw. Pick Cox or the forest to see curves."
        )
    else:
        fig_ind = px.line(
            curves,
            x="time",
            y="survival",
            color="customer",
            line_shape="hv",
            title=f"<b>Predicted survival curves — {curve_model_name}</b>",
            labels={"time": "Months since signup", "survival": "S(t)"},
        )
        fig_ind.update_layout(height=470, yaxis_range=[0, 1])
        st.plotly_chart(fig_ind, use_container_width=True)
        st.caption(
            "Where a curve crosses 50% is that customer's predicted median lifetime — a far more "
            "actionable number than a churn flag."
        )

    if curve_model_name == "Cox Proportional Hazards":
        st.markdown("**What drives the hazard?**")
        coefficients = S.cox_coefficients(model, split["feature_names"])
        fig_coef = px.bar(
            coefficients,
            x="coefficient",
            y="feature",
            orientation="h",
            color="coefficient",
            color_continuous_scale="RdYlGn_r",
            color_continuous_midpoint=0,
            title="<b>Cox coefficients — right speeds churn up, left slows it down</b>",
        )
        fig_coef.update_layout(height=430, coloraxis_showscale=False)
        st.plotly_chart(fig_coef, use_container_width=True)
        st.dataframe(coefficients.round(4), use_container_width=True, hide_index=True)
        st.caption(
            "`exp(coefficient)` is the hazard ratio: the multiplicative effect on the churn hazard of a "
            "one-standard-deviation increase in that feature. Above 1 = churns sooner."
        )

    st.divider()
    st.subheader("Risk strata — from a model to a campaign plan")
    st.markdown(
        "Split the test customers into quartiles by predicted risk, then draw each group's "
        "**observed** Kaplan-Meier curve. If the model works, the curves fan out — and that fan "
        "is the campaign plan, because it says both *who* is at risk and roughly *when*."
    )

    strata_curves, strata_summary = S.risk_strata(model, split["X_test"], split["y_test"])
    fig_strata = px.line(
        strata_curves,
        x="time",
        y="survival",
        color="risk_group",
        line_shape="hv",
        category_orders={"risk_group": ["Low", "Medium", "High", "Very high"]},
        color_discrete_sequence=["#2e7d32", "#9ccc65", "#fb8c00", "#c62828"],
        title=f"<b>Observed survival by predicted risk quartile ({curve_model_name})</b>",
        labels={"time": "Months since signup", "survival": "S(t)"},
    )
    fig_strata.update_layout(height=470, yaxis_range=[0, 1])
    st.plotly_chart(fig_strata, use_container_width=True)

    st.dataframe(
        strata_summary.style.format({"Observed churn rate": "{:.1%}", "Mean tenure": "{:.1f}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        **What survival analysis adds to the propensity page**

        - Censored customers are *used*, not discarded or mislabelled.
        - The answer is a curve, so you can pick the **moment** to act, not just the target.
        - Timing follows from the model: contact the "very high" group *before* their curve drops.
        - The same machinery powers churn-time forecasting and lifetime-value modelling.

        **Caveats worth saying out loud**

        - Cox assumes proportional hazards — check it before trusting the ratios.
        - Score-only models can be ranked but not calibrated.
        - Compare several models, by cross-validation and more than one metric.
        """
    )

    show_source(S.individual_survival_curves, "🐍 Individual survival curves")
    show_source(S.cox_coefficients, "🐍 Cox coefficients as hazard ratios")
    show_source(S.risk_strata, "🐍 Risk stratification")
    show_sibling_script("survival.py", "📄 The full sibling script")
