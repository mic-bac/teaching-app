"""
4_Propensity.py
---------------
Propensity Models lesson page (look-alike modeling).

Surfaces the sibling ``predictions/propensity.py`` script as an interactive
lesson: the look-alike time-window design, three model families, cross-validation
and hyperparameter search, the metrics that matter for imbalanced data, learning
and validation curves, and finally propensity scores cut into risk segments.
All logic lives in ``utils/propensity_utils`` (UI-free); this page only caches,
renders, and shows the real source code that runs.

Graceful behavior: the churn CSVs live in the sibling repo and are git-ignored,
so a synthetic customer base with the same columns stands in when they are
missing (the page says which one it is using). Every model is fitted on a
sample chosen with the slider, so a live demo never waits minutes for a fit.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import propensity_utils as P
from utils.teaching import source_of

st.set_page_config(page_title="Propensity Models", layout="wide", page_icon="🔮")

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


@st.cache_data(show_spinner=False)
def cached_features(sample_size: int):
    frame, _ = cached_churn(sample_size)
    return P.prepare_features(frame)


@st.cache_resource(show_spinner=False)
def cached_split(sample_size: int):
    X, y, _ = cached_features(sample_size)
    return P.split_and_scale(X, y)


@st.cache_resource(show_spinner=False)
def cached_fit(model_name: str, sample_size: int):
    """Fit one model family on the current sample and score it on the held-out test set."""
    split = cached_split(sample_size)
    X_fit, X_eval = P.training_data_for(model_name, split)
    return P.fit_and_score(P.build_model(model_name), X_fit, split["y_train"], X_eval, split["y_test"])


def _band(frame: pd.DataFrame, x: str, mean: str, std: str, name: str, color: str) -> list:
    """A mean line plus a +/- one-standard-deviation ribbon (two Plotly traces)."""
    upper, lower = frame[mean] + frame[std], frame[mean] - frame[std]
    return [
        go.Scatter(
            x=np.concatenate([frame[x], frame[x][::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor=color,
            opacity=0.15,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        go.Scatter(x=frame[x], y=frame[mean], mode="lines+markers", name=name, line=dict(color=color, width=3)),
    ]


# ===========================================================================
# Header + the one control every tab depends on
# ===========================================================================
st.title("🔮 Propensity Models — Look-alike Modeling")
st.markdown(
    """
    **"Acquiring and keeping customers costs money — who should we talk to?"**

    A propensity model answers that per customer: *how likely* is this person to
    do the thing we care about — buy, accept an offer, or churn? Because we have
    labelled history, this is **supervised classification** — but with a time
    design that plain classification exercises skip.

    Every tab shows the **real code** that produces the result, not a slide summary.
    """
)

control, status = st.columns([2, 3])
with control:
    sample_size = st.select_slider(
        "Customers to model",
        options=[2_000, 5_000, 10_000, 20_000, 50_000],
        value=10_000,
        help="The full dataset has ~505,000 customers. A sample keeps every fit on this page interactive.",
    )
with status:
    churn_df, data_source = cached_churn(sample_size)
    if data_source == "kaggle":
        st.success(f"Kaggle churn dataset — {len(churn_df):,} customers sampled from the sibling `predictions/` repo.")
    else:
        st.warning(
            "The Kaggle CSVs were not found in `predictions/data/churn/`, so this lesson is running on a "
            "**synthetic** customer base with the same columns and a planted signal. Everything still works; "
            "the numbers are simulated."
        )

split = cached_split(sample_size)

tab_overview, tab_windows, tab_data, tab_models, tab_tuning, tab_curves, tab_scores = st.tabs(
    [
        "🧭 Overview",
        "⏳ Look-alike Setup",
        "🔎 Data & Features",
        "🤖 Models",
        "🎛️ CV & Tuning",
        "📈 Learning & Validation Curves",
        "🎯 Scoring & Segments",
    ]
)

# ===========================================================================
# TAB 1 — Overview
# ===========================================================================
with tab_overview:
    st.subheader("The business question")
    st.markdown(
        """
        > *The membership is established and new signups are stagnating. We want to
        > analyse churn and know which customers are at risk, so we can take the right
        > measures. We also want to win new members from our network — who should we
        > approach?*

        Two probabilities answer this:

        - **Churn probability** — who is about to leave, so retention can act first.
        - **Purchase probability** — who looks like the people who already joined.

        They are the *same model* with a different label. Everything on this page
        works for both; we use churn because that is the labelled data we have.
        """
    )

    st.subheader("How a propensity model is built")
    st.graphviz_chart(P.propensity_pipeline_dot(), use_container_width=True)

    st.subheader("Three ways to predict behaviour")
    st.table(
        pd.DataFrame(
            [
                {
                    "Method": "Propensity models (look-alike)",
                    "Learning": "Supervised classification",
                    "Answers": "How likely is this person to act?",
                    "In this app": "This page",
                },
                {
                    "Method": "Survival analysis",
                    "Learning": "Supervised, time-to-event",
                    "Answers": "*When* will it happen?",
                    "In this app": "Survival Analysis page",
                },
                {
                    "Method": "Behavioural segmentation",
                    "Learning": "Unsupervised clustering",
                    "Answers": "*Why* do groups behave differently?",
                    "In this app": "Later lesson",
                },
            ]
        )
    )
    st.info("⏳ Continue with **Survival Analysis** in the sidebar — same business question, but *when* instead of *whether*.")

    show_sibling_script("propensity.py", "📄 The full sibling script")

# ===========================================================================
# TAB 2 — Look-alike setup (the time design)
# ===========================================================================
with tab_windows:
    st.subheader("Observation → buffer → outcome")
    st.markdown(
        """
        The most common way a propensity model fails is not a bad algorithm — it is a
        leaky **time design**. A profile is built from an *observation* period, and the
        label is read from a later *outcome* period, with a deliberate gap between them.
        """
    )

    c1, c2, c3 = st.columns(3)
    observation_weeks = c1.slider("Observation period (weeks)", 4, 52, 12)
    buffer_weeks = c2.slider("Buffer / lead time (weeks)", 0, 12, 2)
    outcome_weeks = c3.slider("Outcome period (weeks)", 1, 26, 4)

    timeline = P.lookalike_timeline(observation_weeks, buffer_weeks, outcome_weeks)

    fig_timeline = go.Figure()
    colors = {"Observation": "#66bb6a", "Buffer": "#eceff1", "Outcome": "#ffb300"}
    for _, row in timeline.iterrows():
        fig_timeline.add_trace(
            go.Bar(
                x=[row["weeks"]],
                y=["Training profile"],
                base=[row["start"]],
                orientation="h",
                name=row["window"],
                marker_color=colors[row["window"]],
                marker_line=dict(color="#607d8b", width=1),
                text=row["window"],
                textposition="inside",
                hovertemplate=f"<b>{row['window']}</b><br>weeks {row['start']}–{row['end']}<br>{row['role']}<extra></extra>",
            )
        )
    # The evaluated profile sits later in time: same shape, different customer.
    shift = observation_weeks + buffer_weeks + outcome_weeks
    for _, row in timeline.iterrows():
        if row["window"] == "Buffer":
            continue
        fig_timeline.add_trace(
            go.Bar(
                x=[row["weeks"]],
                y=["Scored today"],
                base=[row["start"] + shift],
                orientation="h",
                marker_color=colors[row["window"]],
                marker_line=dict(color="#607d8b", width=1),
                opacity=0.45,
                showlegend=False,
                hovertemplate=f"<b>{row['window']} (future customer)</b><extra></extra>",
            )
        )
    fig_timeline.update_layout(
        barmode="stack",
        height=280,
        xaxis_title="Weeks",
        margin=dict(l=10, r=10, t=30, b=10),
        title="<b>Learn from the past, score a different customer in the future</b>",
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    st.dataframe(
        timeline[["window", "start", "end", "weeks", "role"]],
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        f"With these settings a customer's features come from weeks 0–{observation_weeks}, "
        f"nothing from weeks {observation_weeks}–{observation_weeks + buffer_weeks} may be used, "
        f"and the label is whether they churned in weeks {observation_weeks + buffer_weeks}–"
        f"{observation_weeks + buffer_weeks + outcome_weeks}."
    )

    st.markdown(
        """
        **Why the buffer is not optional**

        - *Lead time.* If retention needs two weeks to prepare an offer, a model that
          predicts churn one day ahead is worthless however accurate it is.
        - *Label leakage.* The final days before a cancellation are full of give-away
          signals — auto-renew switched off, a "how do I close my account" ticket. Train
          on those and you get a model with a beautiful ROC-AUC that predicts nothing you
          can still act on.

        This Kaggle table is already aggregated to one row per customer, so the windows are
        baked in rather than something we can slide. Naming them is still the first task on
        real data, not an afterthought.
        """
    )

    show_source(P.lookalike_timeline, "🐍 Laying out the three windows")

# ===========================================================================
# TAB 3 — Data & features
# ===========================================================================
with tab_data:
    st.subheader("The customer base")

    X, y, encoders = cached_features(sample_size)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Customers", f"{len(churn_df):,}")
    m2.metric("Features", X.shape[1])
    m3.metric("Churn rate", f"{y.mean() * 100:.1f}%")
    m4.metric("Train / test", f"{len(split['X_train']):,} / {len(split['X_test']):,}")

    st.dataframe(churn_df.head(8), use_container_width=True, hide_index=True)

    st.markdown(
        """
        Good features beat clever algorithms. They come from different worlds and this
        dataset has all three: **socio-demographic** (Age, Gender), **behavioural**
        (Usage Frequency, Support Calls, Last Interaction) and **commercial**
        (Total Spend, Subscription Type, Contract Length).
        """
    )

    correlations = X.corrwith(y).sort_values()
    fig_corr = px.bar(
        correlations,
        orientation="h",
        color=correlations.values,
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        labels={"value": "Correlation with churn", "index": ""},
        title="<b>Linear correlation with churn — a first hint, not the answer</b>",
    )
    fig_corr.update_layout(height=420, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption(
        "Correlation only sees straight lines. A tree model will later find effects that "
        "barely show up here — which is exactly why we compare model families."
    )

    st.markdown(
        f"""
        **Preparation choices worth stating out loud**

        - `CustomerID` is dropped — it is a key, not a feature. Left in, a tree happily memorises it.
        - Categoricals ({", ".join(f"`{c}`" for c in encoders)}) are label-encoded, and the encoders
          are kept so a new customer can be encoded the same way later.
        - The split is **stratified**: the churn rate is identical on both sides, so a test score
          measures the model instead of an accident of sampling.
        """
    )

    show_source(P.load_churn, "🐍 Loading the data (with the synthetic fallback)")
    show_source(P.prepare_features, "🐍 Encoding features")
    show_source(P.split_and_scale, "🐍 Splitting and scaling")

# ===========================================================================
# TAB 4 — Models
# ===========================================================================
with tab_models:
    st.subheader("Three model families")
    st.markdown(
        "All three expose `predict_proba` — the entry requirement for look-alike modelling, "
        "where we need *how likely a point falls in a class*, not just the class."
    )
    st.table(
        pd.DataFrame(
            [
                {"Model": name, "Needs scaling": "Yes" if spec["needs_scaling"] else "No",
                 "Strength": spec["strength"], "Weakness": spec["weakness"]}
                for name, spec in P.MODEL_SPECS.items()
            ]
        )
    )

    model_name = st.selectbox("Fit and inspect", list(P.MODEL_SPECS), key="models_pick")
    with st.spinner(f"Fitting {model_name} on {sample_size:,} customers…"):
        fitted = cached_fit(model_name, sample_size)

    metric_cols = st.columns(6)
    for column, (metric, value) in zip(metric_cols, fitted["metrics"].items()):
        column.metric(metric, f"{value:.3f}")

    majority_share = max(float(split["y_test"].mean()), 1 - float(split["y_test"].mean()))
    st.caption(
        "Accuracy alone is a trap here: with a majority class this large, always predicting "
        f"that one class already scores {majority_share:.0%}."
    )

    left, right = st.columns(2)

    with left:
        st.markdown("**Confusion matrix** — where the errors actually are")
        confusion = P.confusion_frame(split["y_test"], fitted["predictions"])
        fig_cm = px.imshow(
            confusion,
            text_auto=True,
            color_continuous_scale="Blues",
            aspect="auto",
            labels=dict(color="Customers"),
        )
        fig_cm.update_layout(height=360, coloraxis_showscale=False, margin=dict(t=20))
        st.plotly_chart(fig_cm, use_container_width=True)
        false_positives = int(confusion.iloc[0, 1])
        false_negatives = int(confusion.iloc[1, 0])
        st.caption(
            f"{false_positives:,} false alarms (retention budget spent on customers who would have stayed) "
            f"vs {false_negatives:,} missed churners (customers lost). Which error is more expensive is a "
            "business question, not a modelling one."
        )

    with right:
        st.markdown("**ROC and Precision-Recall** — quality across *all* thresholds")
        roc = P.roc_frame(split["y_test"], fitted["probabilities"])
        pr = P.pr_frame(split["y_test"], fitted["probabilities"])

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name="ROC", line=dict(width=3)))
        fig_roc.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dot", color="gray"))
        )
        fig_roc.update_layout(
            height=170,
            margin=dict(t=30, b=10),
            title=f"ROC — AUC {fitted['metrics']['ROC-AUC']:.3f}",
            xaxis_title="False positive rate",
            yaxis_title="True positive rate",
            showlegend=False,
        )
        st.plotly_chart(fig_roc, use_container_width=True)

        fig_pr = go.Figure()
        fig_pr.add_trace(
            go.Scatter(x=pr["recall"], y=pr["precision"], mode="lines", name="PR", line=dict(width=3, color="#e67e22"))
        )
        fig_pr.add_hline(y=float(split["y_test"].mean()), line_dash="dot", line_color="gray")
        fig_pr.update_layout(
            height=170,
            margin=dict(t=30, b=10),
            title=f"Precision-Recall — AP {fitted['metrics']['PR-AUC']:.3f}",
            xaxis_title="Recall",
            yaxis_title="Precision",
            showlegend=False,
        )
        st.plotly_chart(fig_pr, use_container_width=True)
        st.caption(
            "The dotted line is the chance level: for PR that is the share of churners, "
            "so a PR curve is only impressive relative to it."
        )

    st.markdown("**All three side by side** (same split, same threshold)")
    with st.spinner("Fitting the remaining families…"):
        comparison = pd.DataFrame(
            {name: cached_fit(name, sample_size)["metrics"] for name in P.MODEL_SPECS}
        ).round(4)
    st.dataframe(comparison, use_container_width=True)
    st.success(f"Best ROC-AUC: **{comparison.loc['ROC-AUC'].idxmax()}**")

    show_source(P.build_model, "🐍 Building the three model families")
    show_source(P.fit_and_score, "🐍 Fitting and scoring")
    show_source(P.metrics_at_threshold, "🐍 The metrics, and what each one asks")

# ===========================================================================
# TAB 5 — Cross-validation and hyperparameter tuning
# ===========================================================================
with tab_tuning:
    st.subheader("Generalisation: cross-validation")
    st.markdown(
        """
        A model should not memorise the training data, it should **generalise**. A single
        train/test split is one draw of a random variable — k-fold cross-validation trains
        and tests *k* times and averages, which is a far more stable estimate.
        **Stratified** folds keep the class ratio in every fold.
        """
    )

    cv_model = st.selectbox("Model", list(P.MODEL_SPECS), key="cv_model")
    n_splits = st.slider("Folds (k)", 3, 10, 5)

    if st.button("🔁 Run cross-validation"):
        X_fit, _ = P.training_data_for(cv_model, split)
        with st.spinner(f"Training {cv_model} {n_splits} times…"):
            scores = P.cross_validate_model(P.build_model(cv_model), X_fit, split["y_train"], n_splits=n_splits)
        st.session_state["cv_scores"] = (cv_model, scores)

    if "cv_scores" in st.session_state:
        name, scores = st.session_state["cv_scores"]
        fold_frame = pd.DataFrame({"Fold": [f"Fold {i}" for i in range(1, len(scores) + 1)], "ROC-AUC": scores})
        fig_folds = px.bar(fold_frame, x="Fold", y="ROC-AUC", title=f"<b>{name} — per-fold ROC-AUC</b>")
        fig_folds.add_hline(
            y=scores.mean(), line_dash="dash", annotation_text=f"mean {scores.mean():.4f}"
        )
        fig_folds.update_layout(height=380, yaxis_range=[max(0.5, scores.min() - 0.05), 1.0])
        st.plotly_chart(fig_folds, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.metric("Mean ROC-AUC", f"{scores.mean():.4f}")
        c2.metric("Spread (std)", f"{scores.std():.4f}")
        st.caption("Report the mean ± std, not one lucky split. A large spread means the estimate itself is shaky.")
    else:
        st.info("Pick a model and click **🔁 Run cross-validation**.")

    st.divider()
    st.subheader("Hyperparameter tuning")
    st.markdown(
        """
        Hyperparameters (tree depth, regularisation strength, …) steer *how* a model learns;
        they are not learned from the data. **GridSearchCV** tries every combination;
        **RandomizedSearchCV** samples `n_iter` of them, which wins as soon as the grid gets
        large. Both score every candidate by cross-validation — so the price is
        *candidates × folds* model fits.
        """
    )

    tune_name = st.selectbox("Model to tune", list(P.MODEL_SPECS), key="tune_model")
    spec = P.MODEL_SPECS[tune_name]
    tune_cols = st.columns(2)
    tune_cv = tune_cols[0].slider("Folds", 2, 5, 3, key="tune_cv")
    if spec["search"] == "random":
        n_iter = tune_cols[1].slider("Candidates sampled (n_iter)", 3, 20, 8)
        total_grid = int(np.prod([len(v) for v in spec["grid"].values()]))
        st.caption(
            f"**RandomizedSearchCV** over {total_grid} possible combinations → "
            f"{n_iter} candidates × {tune_cv} folds = **{n_iter * tune_cv} model fits**."
        )
    else:
        n_iter = 0
        total_grid = int(np.prod([len(v) for v in spec["grid"].values()]))
        tune_cols[1].markdown(f"**GridSearchCV** — the whole grid ({total_grid} candidates)")
        st.caption(f"{total_grid} candidates × {tune_cv} folds = **{total_grid * tune_cv} model fits**.")

    st.code(repr(spec["grid"]), language="python")

    if st.button("🎛️ Search"):
        X_fit, _ = P.training_data_for(tune_name, split)
        with st.spinner("Cross-validating every candidate…"):
            st.session_state["tuned"] = (tune_name, P.tune_model(tune_name, X_fit, split["y_train"], n_iter=n_iter or 8, cv=tune_cv))

    if "tuned" in st.session_state:
        name, tuned = st.session_state["tuned"]
        baseline = cached_fit(name, sample_size)["metrics"]["ROC-AUC"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Best CV ROC-AUC", f"{tuned['best_score']:.4f}")
        c2.metric("Untuned (test)", f"{baseline:.4f}")
        c3.metric("Model fits spent", f"{tuned['n_fits']:,}")
        st.write("**Best parameters**")
        st.json(tuned["best_params"])
        st.dataframe(tuned["results"].head(10), use_container_width=True, hide_index=True)
        st.caption(
            "The CV score and the test score are different numbers measured on different data — "
            "compare tuned-vs-untuned *within* one column, never across."
        )
    else:
        st.info("Choose a model and click **🎛️ Search** to run the tuning.")

    show_source(P.cross_validate_model, "🐍 Stratified k-fold cross-validation")
    show_source(P.tune_model, "🐍 Grid vs randomized search")

# ===========================================================================
# TAB 6 — Learning and validation curves
# ===========================================================================
with tab_curves:
    st.subheader("Two diagnostics, two different questions")
    st.markdown(
        """
        | Curve | Question | Reading |
        |---|---|---|
        | **Learning curve** | Would *more data* help? | both low and together → underfitting · train high, validation low → overfitting · converging high → healthy |
        | **Validation curve** | How does *one hyperparameter* change things? | find where validation peaks, before it starts to fall |
        """
    )

    curve_model = st.selectbox("Model", list(P.MODEL_SPECS), key="curve_model")
    X_fit, _ = P.training_data_for(curve_model, split)

    if st.button("📈 Draw both curves"):
        sweep = P.VALIDATION_SWEEPS[curve_model]
        with st.spinner("Refitting the model at several training sizes and parameter values…"):
            learning = P.learning_curve_frame(P.build_model(curve_model), X_fit, split["y_train"], n_points=5)
            validation = P.validation_curve_frame(
                P.build_model(curve_model), X_fit, split["y_train"], sweep["param"], sweep["values"]
            )
        st.session_state["curves"] = (curve_model, learning, validation)

    if "curves" in st.session_state:
        name, learning, validation = st.session_state["curves"]
        sweep = P.VALIDATION_SWEEPS[name]

        fig_lc = go.Figure()
        for trace in _band(learning, "train_size", "train_mean", "train_std", "Training", "#3498db"):
            fig_lc.add_trace(trace)
        for trace in _band(learning, "train_size", "validation_mean", "validation_std", "Validation", "#e67e22"):
            fig_lc.add_trace(trace)
        fig_lc.update_layout(
            title=f"<b>Learning curve — {name}</b>",
            xaxis_title="Training samples",
            yaxis_title="ROC-AUC",
            height=420,
        )
        st.plotly_chart(fig_lc, use_container_width=True)

        gap = float(learning["train_mean"].iloc[-1] - learning["validation_mean"].iloc[-1])
        c1, c2, c3 = st.columns(3)
        c1.metric("Training (full sample)", f"{learning['train_mean'].iloc[-1]:.4f}")
        c2.metric("Validation (full sample)", f"{learning['validation_mean'].iloc[-1]:.4f}")
        c3.metric("Gap", f"{gap:+.4f}", help="A large positive gap is the signature of overfitting.")
        if gap > 0.05:
            st.warning("Training is well above validation — the model is memorising. Regularise or simplify it.")
        elif learning["validation_mean"].iloc[-1] < 0.7:
            st.warning("Both curves sit low — the model is too simple for this problem (underfitting).")
        else:
            st.success("The curves converge at a high level: this model generalises on this data.")

        fig_vc = go.Figure()
        for trace in _band(validation, "value", "train_mean", "train_std", "Training", "#3498db"):
            fig_vc.add_trace(trace)
        for trace in _band(validation, "value", "validation_mean", "validation_std", "Validation", "#e67e22"):
            fig_vc.add_trace(trace)
        fig_vc.update_layout(
            title=f"<b>Validation curve — {sweep['param']}</b>",
            xaxis_title=sweep["param"],
            xaxis_type="log" if sweep["log_scale"] else "linear",
            yaxis_title="ROC-AUC",
            height=420,
        )
        st.plotly_chart(fig_vc, use_container_width=True)
        best_value = validation.loc[validation["validation_mean"].idxmax(), "value"]
        st.info(f"{sweep['reads']}  \nBest validation score at **{sweep['param']} = {best_value}**.")
    else:
        st.info("Pick a model and click **📈 Draw both curves** — this refits it several times, so it takes a moment.")

    show_source(P.learning_curve_frame, "🐍 Learning curve")
    show_source(P.validation_curve_frame, "🐍 Validation curve")

# ===========================================================================
# TAB 7 — Scoring and risk segments
# ===========================================================================
with tab_scores:
    st.subheader("From probabilities to a campaign")
    st.markdown(
        "The deliverable is not *churn: yes/no*. It is a score per customer that marketing "
        "can sort, cut and act on — and the cut-offs are a **business** decision."
    )

    score_model = st.selectbox("Score with", list(P.MODEL_SPECS), index=1, key="score_model")
    scored = cached_fit(score_model, sample_size)

    c1, c2 = st.columns(2)
    medium_cut = c1.slider("Low | Medium cut-off", 0.05, 0.60, 0.40, step=0.05)
    high_cut = c2.slider("Medium | High cut-off", 0.60, 0.95, 0.70, step=0.05)

    segments = P.risk_segments(
        scored["probabilities"], split["y_test"], customer_ids=split["X_test"].index, low=medium_cut, high=high_cut
    )
    summary = P.segment_summary(segments)

    fig_scores = px.histogram(
        segments.assign(Outcome=segments["Actual"].map({0: "Stayed", 1: "Churned"})),
        x="Propensity",
        color="Outcome",
        nbins=40,
        barmode="overlay",
        opacity=0.7,
        color_discrete_map={"Stayed": "#2ecc71", "Churned": "#e74c3c"},
        title="<b>Propensity score distribution — a good model separates the two colours</b>",
    )
    fig_scores.add_vline(x=medium_cut, line_dash="dash", annotation_text="Low | Medium")
    fig_scores.add_vline(x=high_cut, line_dash="dash", annotation_text="Medium | High")
    fig_scores.update_layout(height=430)
    st.plotly_chart(fig_scores, use_container_width=True)

    st.dataframe(
        summary.style.format(
            {"Share": "{:.1%}", "Mean propensity": "{:.3f}", "Actual churn rate": "{:.1%}"}
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "The realised churn rate should climb steeply from Low to High. That climb *is* the "
        "business value: it says the retention budget can be aimed instead of sprayed."
    )

    st.markdown("**Where should the decision threshold sit?**")
    sweep = P.threshold_sweep(split["y_test"], scored["probabilities"])
    fig_sweep = px.line(
        sweep.melt("threshold", var_name="Metric", value_name="Score"),
        x="threshold",
        y="Score",
        color="Metric",
        title="<b>The model is fixed — the threshold is still yours to choose</b>",
    )
    fig_sweep.update_layout(height=380)
    st.plotly_chart(fig_sweep, use_container_width=True)
    st.caption(
        "Raise the threshold and precision rises while recall falls: fewer wasted offers, more customers "
        "lost. Lower it and the trade goes the other way."
    )

    st.markdown("**What drives the prediction?**")
    importance = P.feature_importance(scored["model"], split["feature_names"])
    if importance.empty:
        st.info(
            f"{score_model} exposes no built-in feature attribution — for a neural network you would reach "
            "for permutation importance or SHAP instead."
        )
    else:
        fig_imp = px.bar(
            importance.sort_values("weight"),
            x="weight",
            y="feature",
            orientation="h",
            title=f"<b>Top features — {importance['kind'].iloc[0]}</b>",
        )
        fig_imp.update_layout(height=420)
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown(
        """
        **Actions per segment**

        | Segment | Action |
        |---|---|
        | 🔴 High risk | Personal outreach, retention offer, escalate high-value accounts |
        | 🟡 Medium risk | Automated engagement, satisfaction survey, usage tips |
        | 🟢 Low risk | Standard communication, upsell and loyalty programmes |

        Swap the label from *churned* to *bought a membership* and the very same machinery
        answers the acquisition half of the business question.
        """
    )

    show_source(P.risk_segments, "🐍 Cutting scores into segments")
    show_source(P.threshold_sweep, "🐍 Sweeping the decision threshold")
    show_source(P.feature_importance, "🐍 Feature importance across model families")
    show_sibling_script("propensity.py", "📄 The full sibling script")
