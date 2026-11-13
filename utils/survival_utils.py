"""
survival_utils.py
-----------------
UI-free, importable logic for the Survival Analysis lesson page.

Adapted from the sibling ``predictions/survival.py`` teaching script (a ``# %%``
notebook with top-level side effects) into pure functions that return data and
never touch Streamlit, so the page can cache/render them and the tests can
exercise them deterministically.

Where classification asks *whether* a customer churns, survival analysis asks
*when* — and handles the customers who have not churned **yet** instead of
throwing them away. Three ideas carry the lesson:

- **Censoring.** ``event=0`` does not mean "never churns", it means "still here
  when we stopped looking". That partial information is usable, and Kaplan-Meier
  uses it.
- **The survival function** S(t) = P(T > t): the probability of still being a
  customer after t months.
- **Different metrics.** Censoring makes MSE/RMSE inapplicable; the concordance
  index, time-dependent AUC(t) and the integrated Brier score take their place.

Data loading is shared with the propensity lesson — see
``utils.propensity_utils.load_churn``, including its synthetic fallback.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.svm import FastSurvivalSVM

# Columns that must never become features: the identifier, the classification
# label, and `Tenure` — which IS the time we are modelling. Feeding it in is
# textbook target leakage.
EXCLUDED_FEATURES = ["CustomerID", "Churn", "event", "time", "Tenure"]

# The model line-up from the lecture slide, kept next to the code that fits them.
MODEL_TABLE = [
    {
        "Model": "Cox Proportional Hazards",
        "Idea": "Semi-parametric: covariates shift a shared baseline hazard proportionally",
        "Strength": "Interpretable hazard ratios per feature",
        "Limitation": "Bound to the proportional-hazards assumption",
        "Gives S(t)?": "Yes",
        "In plain words": "Learns one factor per feature that raises or lowers the risk of leaving "
                          "at every moment — a rule like \"a monthly contract doubles the risk\".",
    },
    {
        "Model": "Random Survival Forest",
        "Idea": "Ensemble of survival trees; predictions are aggregated",
        "Strength": "Captures non-linear effects, few assumptions",
        "Limitation": "Harder to interpret",
        "Gives S(t)?": "Yes",
        "In plain words": "Hundreds of decision trees sort customers into groups (\"monthly contract "
                          "and many support calls\") and note how long such groups stay; the forest "
                          "averages their answers.",
    },
    {
        "Model": "Survival SVM",
        "Idea": "The SVM idea transferred to survival data as a ranking problem",
        "Strength": "Complex, non-linear relations via kernels",
        "Limitation": "Outputs a score only — rank customers, do not calibrate them",
        "Gives S(t)?": "No",
        "In plain words": "Learns only *who leaves before whom* — like ranking runners by finishing "
                          "order without timing them.",
    },
]


# ===========================================================================
# Turning a customer table into survival data
# ===========================================================================
def prepare_survival(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Build ``(X, y, feature_names)`` for scikit-survival.

    ``y`` is a structured array of ``(event, time)`` — the format every
    scikit-survival estimator expects. ``event`` is whether churn was observed,
    ``time`` is how long we watched the customer (their tenure in months).

    Categoricals become dummies. Numerics are standardised later, in
    ``split_survival`` — *after* the split, so the test customers never
    influence the scaling (split first, then preprocess).
    """
    frame = frame.copy()
    frame["event"] = frame["Churn"].astype(bool)
    frame["time"] = frame["Tenure"].astype(float)

    feature_cols = [c for c in frame.columns if c not in EXCLUDED_FEATURES]
    X = pd.get_dummies(frame[feature_cols], drop_first=True).astype(float)  # sksurv wants floats

    y = np.array(list(zip(frame["event"], frame["time"])), dtype=[("event", bool), ("time", float)])
    return X, y, list(X.columns)


def split_survival(X: pd.DataFrame, y: np.ndarray, test_size: float = 0.3, random_state: int = 42):
    """Train/test split stratified on the event indicator, then standardise.

    Cox and the SVM optimise over a linear score where scale matters, so the
    continuous features are standardised — but only *after* the split, with the
    scaler fitted on the training customers alone. The test customers are then
    transformed with those numbers, exactly as a new customer would be after
    deployment. 0/1 dummy columns are left as they are.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y["event"]
    )
    continuous = [c for c in X_train.columns if X_train[c].nunique() > 2]
    X_train, X_test = X_train.copy(), X_test.copy()
    scaler = StandardScaler().fit(X_train[continuous])
    X_train[continuous] = scaler.transform(X_train[continuous])
    X_test[continuous] = scaler.transform(X_test[continuous])
    return X_train, X_test, y_train, y_test


# ===========================================================================
# Kaplan-Meier — the survival function without any model
# ===========================================================================
def kaplan_meier(times, events, conf_type: str | None = "log-log") -> pd.DataFrame:
    """Estimate S(t) non-parametrically.

    At every time where an event happens, Kaplan-Meier multiplies in the fraction
    that survived it::

        S(t) = PROD over t_i <= t of (1 - d_i / n_i)

    with ``d_i`` events at ``t_i`` and ``n_i`` customers still at risk just
    before. Censored customers count in ``n_i`` until they leave the study —
    which is exactly how their partial information gets used.

    Returns a frame with ``time``, ``survival`` and (when a confidence type is
    requested and the data allows it) ``lower``/``upper``.
    """
    events = np.asarray(events).astype(bool)
    times = np.asarray(times, dtype=float)

    if conf_type:
        time_points, survival, conf = kaplan_meier_estimator(events, times, conf_type=conf_type)
        return pd.DataFrame(
            {"time": time_points, "survival": survival, "lower": conf[0], "upper": conf[1]}
        )

    time_points, survival = kaplan_meier_estimator(events, times)
    return pd.DataFrame({"time": time_points, "survival": survival})


def km_by_group(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """One Kaplan-Meier curve per level of ``group_col``, stacked long.

    Splitting the curve shows *where* groups differ, not just that their averages
    do: a gap that opens early calls for onboarding work, one that opens late
    calls for loyalty work.
    """
    events = frame["Churn"].astype(bool) if "event" not in frame else frame["event"].astype(bool)
    times = frame["Tenure"].astype(float) if "time" not in frame else frame["time"].astype(float)

    curves = []
    for level in sorted(frame[group_col].unique()):
        mask = (frame[group_col] == level).to_numpy()
        curve = kaplan_meier(times[mask], events[mask], conf_type=None)
        curve[group_col] = str(level)
        curves.append(curve)
    return pd.concat(curves, ignore_index=True)


def censoring_example(n_customers: int = 6, seed: int = 0) -> pd.DataFrame:
    """A tiny illustration of censoring: calendar time vs time-since-signup.

    Every customer starts their own clock at signup, so the same study cut-off
    lands at a different point in each customer's life. Customers who have not
    churned by then are *right-censored* — the ``?`` rows in the lecture diagram.

    Returns one row per customer with the signup month, the observed duration,
    and whether churn was actually seen.
    """
    rng = np.random.default_rng(seed)
    cutoff = 20.0
    signup = np.sort(rng.uniform(0, 12, n_customers)).round(1)
    churn_at = rng.uniform(3, 22, n_customers).round(1)

    rows = []
    for i, (start, lifetime) in enumerate(zip(signup, churn_at), start=1):
        churned = start + lifetime <= cutoff
        duration = lifetime if churned else cutoff - start
        rows.append(
            {
                "customer": f"Customer {i}",
                "signup": start,
                "observed_until": start + duration,
                "duration": round(duration, 1),
                "event": bool(churned),
                "status": "churned" if churned else "censored",
            }
        )
    frame = pd.DataFrame(rows)
    frame.attrs["cutoff"] = cutoff
    return frame


# ===========================================================================
# Models
# ===========================================================================
def fit_cox(X_train, y_train, alpha: float = 0.1) -> CoxPHSurvivalAnalysis:
    """Cox proportional hazards — the interpretable workhorse."""
    return CoxPHSurvivalAnalysis(alpha=alpha).fit(X_train, y_train)


def fit_rsf(
    X_train,
    y_train,
    n_estimators: int = 50,
    max_depth: int = 5,
    random_state: int = 42,
) -> RandomSurvivalForest:
    """Random Survival Forest — non-linear effects, no proportionality assumption.

    Kept deliberately small (50 shallow trees): a forest is the slowest model on
    this page, and the teaching point survives at demo size.
    """
    return RandomSurvivalForest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=15,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    ).fit(X_train, y_train)


def fit_svm(X_train, y_train, alpha: float = 0.01, max_iter: int = 50, random_state: int = 42):
    """Survival SVM — learns a *ranking* of risk, not a survival function."""
    return FastSurvivalSVM(rank_ratio=1.0, alpha=alpha, max_iter=max_iter, random_state=random_state).fit(
        X_train, y_train
    )


MODEL_BUILDERS = {
    "Cox Proportional Hazards": fit_cox,
    "Random Survival Forest": fit_rsf,
    "Survival SVM": fit_svm,
}


# ===========================================================================
# Evaluation
# ===========================================================================
def c_index(model, X, y) -> float:
    """Concordance index: of all pairs whose true order we know, how many are ranked right?

    "A churns at 3 months, B at 10, so A should get the higher risk." 0.5 is
    coin-flipping, 1.0 is a perfect ordering. It is the survival analogue of
    ROC-AUC and works for *every* model here, because it needs nothing but a risk
    score — which is why it is the only metric the Survival SVM can be graded on.

    One sign convention: Cox and the SVM return a risk score, and
    ``RandomSurvivalForest.predict`` returns a cumulative hazard. Both are
    "higher = churns sooner", so no flipping is needed.
    """
    return float(concordance_index_censored(y["event"], y["time"], model.predict(X))[0])


def cindex_example(model, X, y) -> str:
    """The c-index made concrete: one pair the model orders right, one it gets wrong.

    A pair counts when we know who left first: customer A churned at month a and
    customer B was still with us after a (churned later, or still active).
    """
    risk = np.asarray(model.predict(X))
    events, times = y["event"], y["time"]
    right = wrong = None
    for a in np.argsort(times):
        if not events[a]:
            continue
        later = np.flatnonzero(times > times[a] + 6)  # clearly later, so the example is obvious
        for b in later[:50]:
            if right is None and risk[a] > risk[b]:
                right = (a, b)
            if wrong is None and risk[a] < risk[b]:
                wrong = (a, b)
        if right and wrong:
            break

    def months(value):
        return f"{value:.0f} month{'' if round(value) == 1 else 's'}"

    def pair(a, b, verdict):
        status = "left" if events[b] else "was still a customer"
        return (f"Customer A left after **{months(times[a])}**; customer B {status} at "
                f"**{months(times[b])}**. So A should get the higher risk score. The model gave A "
                f"**{risk[a]:.2f}** and B **{risk[b]:.2f}** — {verdict}")

    lines = []
    if right:
        lines.append("- " + pair(*right, "✅ ordered right."))
    if wrong:
        lines.append("- " + pair(*wrong, "❌ ordered wrong."))
    lines.append("\n(Risk scores have no unit — only which one is higher matters.)")
    return "\n".join(lines)


def evaluation_window(y_train: np.ndarray, y_test: np.ndarray, n_times: int = 9):
    """Pick the times at which AUC(t) and the Brier score can be estimated.

    Censoring bites in the *evaluation* too: both metrics reweight observations by
    the inverse probability of still being uncensored, G(t) — and G drops to zero
    at the very end of the follow-up, where nobody is left to observe. So we grade
    models strictly inside the observed window.

    Returns ``(times, in_window_mask)``: the evaluation times, and the mask that
    selects the test customers those times are valid for.
    """
    horizon = min(y_train["time"].max(), y_test["time"].max())
    in_window = y_test["time"] < horizon
    observed = y_test["time"][in_window & y_test["event"]]
    times = np.unique(np.percentile(observed, np.linspace(10, 90, n_times)))
    return times, in_window


def time_dependent_auc(model, y_train, y_test, X_test, times) -> tuple[pd.DataFrame, float]:
    """AUC(t): how well the model separates "churns by t" from "does not", per t.

    The c-index is one number for the whole horizon. Two models with the same
    c-index can behave completely differently at month 6 versus month 36, and
    campaign planning cares about exactly that difference.
    """
    auc_values, auc_mean = cumulative_dynamic_auc(y_train, y_test, model.predict(X_test), times)
    return pd.DataFrame({"time": times, "auc": auc_values}), float(auc_mean)


def integrated_brier(model, y_train, y_test, X_test, times) -> float | str:
    """Integrated Brier score, or an explanation of why it does not apply.

    Brier(t) is the squared error between the predicted S(t) and what actually
    happened, corrected for censoring; integrating over t gives one number where
    lower is better and ~0.25 is uninformative. Unlike the c-index it grades
    *calibration* as well as discrimination.

    Models that only learn a ranking (``FastSurvivalSVM``) have no S(t) to
    compare against reality, so they get an honest message instead of a number —
    this is the slide's point that some models return scores, not S(t).
    """
    if not hasattr(model, "predict_survival_function"):
        return (
            "This model predicts a risk *score*, not a survival function S(t), "
            "so a Brier score cannot be computed. Rank it with the c-index instead."
        )
    survival_functions = model.predict_survival_function(X_test)
    predictions = np.vstack([[fn(t) for t in times] for fn in survival_functions])
    return float(integrated_brier_score(y_train, y_test, predictions, times))


# ===========================================================================
# Interpretation and activation
# ===========================================================================
def individual_survival_curves(model, X, indices, labels=None) -> pd.DataFrame | str:
    """Predicted S(t) for a handful of specific customers, stacked long.

    This is what a classifier cannot give you: for one named customer, the whole
    curve. "80% chance of still being here in 6 months, 45% in 24" turns into a
    concrete question — when is an intervention worth its cost?
    """
    if not hasattr(model, "predict_survival_function"):
        return "This model predicts a risk score only, so it has no survival curve to draw."

    curves = []
    for position, index in enumerate(indices):
        function = model.predict_survival_function(X.iloc[[index]])[0]
        label = labels[position] if labels is not None else f"Customer {index}"
        curves.append(pd.DataFrame({"time": function.x, "survival": function.y, "customer": label}))
    return pd.concat(curves, ignore_index=True)


def survival_at(curves: pd.DataFrame, month: float) -> pd.Series:
    """Per customer: predicted share still a customer at ``month`` (step function)."""
    return curves[curves["time"] <= month].groupby("customer", sort=False)["survival"].last()


def median_lifetime(curves: pd.DataFrame) -> pd.Series:
    """Per customer: first month where the predicted survival falls to 50 % (NaN if never)."""
    below = curves[curves["survival"] <= 0.5]
    first = below.groupby("customer", sort=False)["time"].min()
    return first.reindex(curves["customer"].unique())


PROFILE_COLUMNS = ["Contract Length", "Subscription Type", "Support Calls", "Payment Delay",
                   "Usage Frequency", "Total Spend"]


def customer_snapshot(raw: pd.DataFrame, y, positions, labels, curves=None,
                      months=(6, 12, 24)) -> pd.DataFrame:
    """The customers behind the curves: who they are, what was predicted, what happened.

    ``raw`` holds the unscaled rows of the same customers (in the same order as
    ``y``), so the table speaks in contracts and support calls, not z-scores.
    """
    frame = raw.iloc[list(positions)][PROFILE_COLUMNS].reset_index(drop=True)
    frame.insert(0, "Customer", labels)
    frame["What happened"] = [
        f"left after {y['time'][i]:.0f} months" if y["event"][i]
        else f"still a customer at {y['time'][i]:.0f} months"
        for i in positions
    ]
    if curves is not None:
        for month in months:
            frame[f"Predicted still here at {month}m"] = survival_at(curves, month).values
        frame["Predicted median lifetime (months)"] = median_lifetime(curves).values
    return frame


def curve_reading(snapshot: pd.DataFrame, month: int = 12) -> str:
    """One plain sentence per customer: the prediction next to what really happened."""
    lines = []
    for _, row in snapshot.iterrows():
        median = row["Predicted median lifetime (months)"]
        when = f"half-way point at **{median:.0f} months**" if pd.notna(median) else "never drops below 50 %"
        lines.append(
            f"- **{row['Customer'].split(' —')[0]}** ({row['Contract Length'].lower()} contract, "
            f"{row['Support Calls']:.0f} support calls, {row['Payment Delay']:.0f} days payment delay): "
            f"**{row[f'Predicted still here at {month}m']:.0%}** chance of still being here after {month} "
            f"months, {when}. Reality: {row['What happened']}."
        )
    return "\n".join(lines)


def strata_reading(curves: pd.DataFrame, summary: pd.DataFrame, month: int = 12) -> str:
    """The risk-strata chart in words: how groups are built, and how far they fan out."""
    still = curves[curves["time"] <= month].groupby("risk_group")["survival"].last()
    low, top = summary.iloc[0], summary.iloc[-1]
    return (
        f"**How it is built:** the model gives each of the {int(summary['Customers'].sum()):,} test "
        f"customers a risk score. We sort them and cut the list into four equal groups of about "
        f"{int(low['Customers']):,}: the quarter the model finds safest is *Low*, the riskiest quarter "
        "*Very high*. Then each group's line shows what **really** happened to it (Kaplan-Meier, not a "
        "prediction).\n\n"
        f"**How to read it:** after {month} months, **{still.get(low['Risk group'], float('nan')):.0%}** of "
        f"the *Low* group are still customers, against **{still.get(top['Risk group'], float('nan')):.0%}** "
        f"of the *Very high* group. In the table, {low['Observed churn rate']:.0%} of *Low* have left so "
        f"far vs {top['Observed churn rate']:.0%} of *Very high*. The wider the lines fan out, the better "
        "the model separates — and the steepest early drop shows *when* to call the riskiest group: before "
        "it happens."
    )


def cox_coefficients(model, feature_names) -> pd.DataFrame:
    """Cox coefficients as log hazard ratios, plus the hazard ratio itself.

    ``exp(coef)`` is the multiplicative effect on the churn hazard of a
    one-standard-deviation increase in that feature: above 1 means churns sooner,
    below 1 means stays longer.
    """
    return (
        pd.DataFrame({"feature": list(feature_names), "coefficient": np.asarray(model.coef_)})
        .assign(hazard_ratio=lambda d: np.exp(d["coefficient"]))
        .sort_values("coefficient")
        .reset_index(drop=True)
    )


def risk_strata(model, X_test, y_test, n_groups: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split test customers into risk quantiles and draw their *observed* curves.

    Returns ``(curves, summary)``. If the model works, the curves fan out — and
    that fan is what a campaign plan is built on, because it says both who is at
    risk and roughly when.
    """
    labels = ["Low", "Medium", "High", "Very high"][:n_groups]
    groups = np.asarray(pd.qcut(model.predict(X_test), q=n_groups, labels=labels))

    curves, rows = [], []
    for level in labels:
        mask = groups == level
        curve = kaplan_meier(y_test["time"][mask], y_test["event"][mask], conf_type=None)
        curve["risk_group"] = level
        curves.append(curve)
        rows.append(
            {
                "Risk group": level,
                "Customers": int(mask.sum()),
                "Observed churn rate": float(y_test["event"][mask].mean()),
                "Mean tenure": float(y_test["time"][mask].mean()),
            }
        )
    return pd.concat(curves, ignore_index=True), pd.DataFrame(rows)


def survival_taxonomy_dot() -> str:
    """Graphviz DOT contrasting the two ways to answer the churn question.

    ``st.graphviz_chart`` renders a raw DOT string client-side, so this needs no
    Graphviz Python package.
    """
    return """
digraph survival {
  rankdir=LR;
  bgcolor="transparent";
  node [fontname="Helvetica", shape=box, style="rounded,filled", fillcolor="#ffffff", color="#90a4ae"];
  edge [fontname="Helvetica", fontsize=10, color="#607d8b"];

  question  [label="Who is at risk\\nof churning?", fillcolor="#e3f2fd"];
  classify  [label="Classification\\n\\\"WILL they churn?\\\"\\nprobability in [0,1]"];
  survive   [label="Survival analysis\\n\\\"WHEN will they churn?\\\"\\nS(t) over time", fillcolor="#c8e6c9", penwidth=3];
  censored  [label="Uses censored customers\\n(still active = information,\\nnot a negative label)", fillcolor="#fff8e1"];
  timing    [label="Act at the right MOMENT", fillcolor="#c8e6c9"];

  question -> classify;
  question -> survive;
  survive -> censored -> timing;
}
""".strip()
