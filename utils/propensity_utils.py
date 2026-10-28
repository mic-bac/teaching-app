"""
propensity_utils.py
-------------------
UI-free, importable logic for the Propensity Models lesson page.

Adapted from the sibling ``predictions/propensity.py`` teaching script. That
script is a ``# %%`` notebook with top-level side effects (it reads CSVs, prints,
and opens Plotly windows), so it can't be imported directly. Here its workflow is
refactored into pure functions that **return data** (DataFrames, arrays, fitted
models) and never touch Streamlit — so the page can cache/render them and the
tests can exercise them deterministically.

The lesson, in the order the slides tell it:
- **Look-alike setup** — observation → buffer → outcome windows, the time design
  that separates a propensity model from plain classification.
- **Classification** — logistic regression, XGBoost, neural network; all three
  chosen because they expose ``predict_proba`` (look-alike modelling needs the
  *probability*, not just the class).
- **Generalisation** — stratified cross-validation, grid/randomized search.
- **Evaluation** — confusion matrix, precision/recall/F1, ROC-AUC, PR-AUC, plus
  learning and validation curves for under-/overfitting.
- **Activation** — propensity scores cut into risk segments.

Graceful degradation: the Kaggle churn CSVs live in the sibling repo and are
git-ignored, so they may be missing on a fresh clone. ``load_churn`` then falls
back to a synthetic customer base with the same columns and a planted signal, and
says so — the lesson always runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
    validation_curve,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Default location of the sibling repo's dataset (read-only).
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "predictions" / "data" / "churn"

CHURN_COLUMNS = [
    "CustomerID",
    "Age",
    "Gender",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Subscription Type",
    "Contract Length",
    "Total Spend",
    "Last Interaction",
    "Churn",
]


# ===========================================================================
# Data loading (with a synthetic fallback so the lesson always runs)
# ===========================================================================
def make_synthetic_churn(n_customers: int = 20_000, random_state: int = 42) -> pd.DataFrame:
    """Generate a churn dataset with the same columns as the Kaggle original.

    The label is drawn from a logistic model of a few features, so the planted
    signal is real but learnable — a model trained on this reaches a believable
    ROC-AUC instead of either 0.5 or a suspicious 1.0.
    """
    rng = np.random.default_rng(random_state)

    contract = rng.choice(["Monthly", "Quarterly", "Annual"], n_customers, p=[0.2, 0.4, 0.4])
    frame = pd.DataFrame(
        {
            "CustomerID": np.arange(n_customers),
            "Age": rng.integers(18, 66, n_customers),
            "Gender": rng.choice(["Male", "Female"], n_customers),
            "Tenure": rng.integers(1, 61, n_customers),
            "Usage Frequency": rng.integers(1, 31, n_customers),
            "Support Calls": rng.integers(0, 11, n_customers),
            "Payment Delay": rng.integers(0, 31, n_customers),
            "Subscription Type": rng.choice(["Basic", "Standard", "Premium"], n_customers),
            "Contract Length": contract,
            "Total Spend": rng.uniform(100, 1000, n_customers).round(2),
            "Last Interaction": rng.integers(1, 31, n_customers),
        }
    )

    # Unhappy customers call support, pay late, spend little and were last seen
    # a while ago; long contracts hold people in place.
    logit = (
        -1.6
        + 0.28 * frame["Support Calls"]
        + 0.06 * frame["Payment Delay"]
        + 0.04 * frame["Last Interaction"]
        - 0.003 * frame["Total Spend"]
        - 0.015 * frame["Tenure"]
        + frame["Contract Length"].map({"Monthly": 0.9, "Quarterly": 0.2, "Annual": -0.4})
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    frame["Churn"] = (rng.random(n_customers) < probability).astype(int)

    return frame[CHURN_COLUMNS]


def load_churn(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    sample_size: int | None = 20_000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, str]:
    """Load the churn dataset; returns ``(dataframe, source)``.

    ``source`` is ``"kaggle"`` or ``"synthetic"`` so the page can be honest about
    which data the numbers on screen came from.

    The Kaggle download ships as two files ("training" and "testing" master).
    They are two halves of one customer base, not a modelling split, so we
    concatenate them and split later ourselves. ``sample_size`` draws a
    *stratified* subsample — the full set is ~505,000 rows and a hyperparameter
    search over that takes hours, which is not what a live lesson needs.
    """
    data_dir = Path(data_dir)
    train_csv = data_dir / "customer_churn_dataset-training-master.csv"
    test_csv = data_dir / "customer_churn_dataset-testing-master.csv"

    if train_csv.exists() and test_csv.exists():
        frame = pd.concat([pd.read_csv(train_csv), pd.read_csv(test_csv)]).reset_index(drop=True)
        frame["CustomerID"] = range(len(frame))  # the two files number customers independently
        frame = frame.dropna().reset_index(drop=True)  # one all-empty row in the Kaggle file
        frame["Churn"] = frame["Churn"].astype(int)
        source = "kaggle"
    else:
        frame = make_synthetic_churn(sample_size or 20_000, random_state)
        source = "synthetic"

    return _stratified_sample(frame, sample_size, random_state), source


def _stratified_sample(frame: pd.DataFrame, sample_size: int | None, random_state: int) -> pd.DataFrame:
    """Subsample while keeping the churn rate intact (and shuffle the strata back)."""
    if sample_size is None or sample_size >= len(frame):
        return frame
    return (
        frame.groupby("Churn", group_keys=False)
        .sample(frac=sample_size / len(frame), random_state=random_state)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )


# ===========================================================================
# Feature preparation
# ===========================================================================
def prepare_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Split into features/target and label-encode the categorical columns.

    Returns ``(X, y, encoders)``. The encoders are kept so a *new* customer can
    be encoded exactly as the training data was — forgetting this is one of the
    classic ways a model silently breaks once it is deployed.

    ``CustomerID`` is dropped: it is a key, not a feature. Left in, a tree model
    will happily memorise it.
    """
    data = frame.drop(columns=["CustomerID"]).copy()

    encoders: dict[str, LabelEncoder] = {}
    for column in data.select_dtypes(include=["object", "string"]).columns:
        encoder = LabelEncoder()
        data[column] = encoder.fit_transform(data[column])
        encoders[column] = encoder

    return data.drop(columns="Churn"), data["Churn"], encoders


def split_and_scale(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> dict:
    """Stratified train/test split plus a standardised copy of the features.

    Two versions of the same split come back: raw (trees split on thresholds and
    do not care about scale) and standardised (logistic regression and neural
    nets are gradient based and very much do). Stratifying keeps the churn rate
    identical on both sides, so a test score measures the model rather than an
    accident of sampling.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": scaler.fit_transform(X_train),
        "X_test_scaled": scaler.transform(X_test),
        "scaler": scaler,
        "feature_names": list(X_train.columns),
    }


# ===========================================================================
# Models — three families, three trade-offs
# ===========================================================================
# Every model here exposes predict_proba, which is the entry requirement for
# look-alike modelling: we need "how likely", not "yes/no".
MODEL_SPECS: dict[str, dict] = {
    "Logistic Regression": {
        "needs_scaling": True,
        "strength": "Interpretable coefficients, fast, a strong baseline",
        "weakness": "Only linear decision boundaries",
        "search": "grid",
        # One hyperparameter, six values: small enough to search exhaustively,
        # which is exactly when a grid search is the right tool.
        "grid": {"C": [0.001, 0.01, 0.1, 1, 10, 100]},
    },
    "XGBoost": {
        "needs_scaling": False,
        "strength": "Captures interactions and non-linearity, strong out of the box",
        "weakness": "Many hyperparameters, harder to explain",
        "search": "random",
        "grid": {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.3],
            "n_estimators": [100, 200],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        },
    },
    "Neural Network": {
        "needs_scaling": True,
        "strength": "Very flexible function shapes",
        "weakness": "Data hungry, slow to tune, opaque",
        "search": "random",
        "grid": {
            "hidden_layer_sizes": [(32,), (32, 16), (64, 32)],
            "alpha": [0.0001, 0.01, 0.1],
            "learning_rate_init": [0.001, 0.01],
        },
    },
}


# One curated hyperparameter per model for the validation curve. Each one is the
# knob that controls model complexity for that family, so the curve actually
# shows the under-fit / sweet-spot / over-fit shape from the lecture.
VALIDATION_SWEEPS: dict[str, dict] = {
    "Logistic Regression": {
        "param": "C",
        "values": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        "log_scale": True,
        "reads": "Low C = strong regularisation (simpler model); high C = the model is free to fit noise.",
    },
    "XGBoost": {
        "param": "max_depth",
        "values": [1, 2, 3, 5, 7, 10],
        "log_scale": False,
        "reads": "Deeper trees can express more interactions — and can memorise the training set.",
    },
    "Neural Network": {
        "param": "alpha",
        "values": [0.0001, 0.001, 0.01, 0.1, 1.0],
        "log_scale": True,
        "reads": "alpha is the L2 penalty on the weights: larger = smoother, simpler network.",
    },
}


def build_model(name: str, random_state: int = 42, **overrides):
    """Return an untrained classifier by name (see ``MODEL_SPECS``)."""
    if name == "Logistic Regression":
        params = {"random_state": random_state, "max_iter": 1000, **overrides}
        return LogisticRegression(**params)
    if name == "XGBoost":
        params = {
            "random_state": random_state,
            "eval_metric": "logloss",
            "max_depth": 4,
            "learning_rate": 0.1,
            "n_estimators": 150,
            **overrides,
        }
        return xgb.XGBClassifier(**params)
    if name == "Neural Network":
        params = {
            "hidden_layer_sizes": (32, 16),
            "alpha": 0.1,
            "random_state": random_state,
            "max_iter": 200,
            "early_stopping": True,
            **overrides,
        }
        return MLPClassifier(**params)
    raise ValueError(f"Unknown model: {name!r}. Choose one of {list(MODEL_SPECS)}.")


def training_data_for(name: str, split: dict) -> tuple:
    """Pick the scaled or raw matrices for this model family."""
    if MODEL_SPECS[name]["needs_scaling"]:
        return split["X_train_scaled"], split["X_test_scaled"]
    return split["X_train"], split["X_test"]


def fit_and_score(model, X_fit, y_fit, X_eval, y_eval, threshold: float = 0.5) -> dict:
    """Fit a classifier and return predictions plus the full metric set.

    One helper for every model keeps the comparison honest: identical data,
    identical metrics, identical threshold.
    """
    model.fit(X_fit, y_fit)
    probabilities = model.predict_proba(X_eval)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return {
        "model": model,
        "probabilities": probabilities,
        "predictions": predictions,
        "metrics": metrics_at_threshold(y_eval, probabilities, threshold),
    }


def metrics_at_threshold(y_true, probabilities, threshold: float = 0.5) -> dict[str, float]:
    """Score a set of probabilities.

    Accuracy alone is a trap on imbalanced data. Precision asks "of those we
    flagged, how many really churn?" (the cost of wasted offers); recall asks "of
    those who churn, how many did we catch?" (the cost of missed saves). ROC-AUC
    and PR-AUC are threshold-free — PR-AUC is the honest one when the positives
    are the rare and expensive class.
    """
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1-Score": f1_score(y_true, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probabilities),
        "PR-AUC": average_precision_score(y_true, probabilities),
    }


# ===========================================================================
# Evaluation artefacts (plot-ready frames)
# ===========================================================================
def confusion_frame(y_true, y_pred) -> pd.DataFrame:
    """2x2 confusion matrix with readable row/column labels."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        matrix,
        index=["Actually stayed", "Actually churned"],
        columns=["Predicted stay", "Predicted churn"],
    )


def roc_frame(y_true, probabilities) -> pd.DataFrame:
    """False-positive/true-positive rate pairs across every threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})


def pr_frame(y_true, probabilities) -> pd.DataFrame:
    """Precision/recall pairs across every threshold."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    # precision_recall_curve returns one more point than thresholds.
    return pd.DataFrame(
        {"recall": recall[:-1], "precision": precision[:-1], "threshold": thresholds}
    )


def threshold_sweep(y_true, probabilities, steps: int = 41) -> pd.DataFrame:
    """Precision/recall/F1 as a function of the decision threshold.

    Shows that the "model" and the "cut-off" are two separate decisions: the same
    probabilities become a cautious or an aggressive campaign depending on where
    the line is drawn.
    """
    rows = []
    for threshold in np.linspace(0.05, 0.95, steps):
        scored = metrics_at_threshold(y_true, probabilities, threshold)
        rows.append({"threshold": threshold, **{k: scored[k] for k in ("Precision", "Recall", "F1-Score")}})
    return pd.DataFrame(rows)


# ===========================================================================
# Generalisation: cross-validation, tuning, learning & validation curves
# ===========================================================================
def cross_validate_model(model, X, y, n_splits: int = 5, random_state: int = 42) -> np.ndarray:
    """Stratified k-fold ROC-AUC scores — one per fold.

    A single train/test split is one draw of a random variable. k-fold trains and
    tests k times and averages, which is a far more stable estimate; stratified
    folds keep the class ratio in every fold, which is mandatory here.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return cross_val_score(model, X, y, cv=cv, scoring="roc_auc")


def tune_model(
    name: str, X, y, n_iter: int = 8, cv: int = 3, random_state: int = 42
) -> dict:
    """Run the search defined in ``MODEL_SPECS`` and report what it cost.

    Grid search tries every combination; randomized search samples ``n_iter`` of
    them, which wins as soon as the grid gets large. Both score each candidate by
    cross-validation, so the price is (candidates x folds) model fits — the page
    shows that number before you press the button.
    """
    spec = MODEL_SPECS[name]
    estimator = build_model(name, random_state=random_state)

    if spec["search"] == "grid":
        search = GridSearchCV(estimator, spec["grid"], cv=cv, scoring="roc_auc", n_jobs=-1)
        n_candidates = int(np.prod([len(v) for v in spec["grid"].values()]))
    else:
        search = RandomizedSearchCV(
            estimator,
            spec["grid"],
            n_iter=n_iter,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            random_state=random_state,
        )
        n_candidates = n_iter

    search.fit(X, y)
    results = (
        pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
        .sort_values("rank_test_score")
        .reset_index(drop=True)
    )
    results["params"] = results["params"].astype(str)
    return {
        "estimator": search.best_estimator_,
        "best_params": search.best_params_,
        "best_score": search.best_score_,
        "results": results,
        "n_candidates": n_candidates,
        "n_fits": n_candidates * cv,
    }


def learning_curve_frame(
    model, X, y, cv: int = 3, n_points: int = 5, random_state: int = 42
) -> pd.DataFrame:
    """Training vs validation ROC-AUC as the training set grows.

    Reading it: both curves low and together = underfitting (model too simple);
    training high with validation far below = overfitting (model memorises);
    curves converging high = healthy. It also answers a budget question — if the
    validation curve has flattened, more data will not help.
    """
    sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        train_sizes=np.linspace(0.2, 1.0, n_points),
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
        scoring="roc_auc",
        n_jobs=-1,
    )
    return pd.DataFrame(
        {
            "train_size": sizes,
            "train_mean": train_scores.mean(axis=1),
            "train_std": train_scores.std(axis=1),
            "validation_mean": val_scores.mean(axis=1),
            "validation_std": val_scores.std(axis=1),
        }
    )


def validation_curve_frame(
    model, X, y, param_name: str, param_range: list, cv: int = 3, random_state: int = 42
) -> pd.DataFrame:
    """Training vs validation ROC-AUC as ONE hyperparameter is swept.

    Where the validation curve peaks and starts to fall is the complexity limit
    of this model on this data — worth knowing *before* launching a full search.
    """
    train_scores, val_scores = validation_curve(
        model,
        X,
        y,
        param_name=param_name,
        param_range=param_range,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
        scoring="roc_auc",
        n_jobs=-1,
    )
    return pd.DataFrame(
        {
            "value": param_range,
            "train_mean": train_scores.mean(axis=1),
            "train_std": train_scores.std(axis=1),
            "validation_mean": val_scores.mean(axis=1),
            "validation_std": val_scores.std(axis=1),
        }
    )


# ===========================================================================
# Interpretation and activation
# ===========================================================================
def feature_importance(model, feature_names, top_n: int = 10) -> pd.DataFrame:
    """Rank features by whatever notion of importance the model provides.

    Two different questions: a linear model reports the *size and direction* of
    an effect (|coefficient|), a tree model reports *how useful a feature was for
    splitting*. They often disagree, and that disagreement is worth discussing.
    """
    if hasattr(model, "coef_"):
        weights = np.abs(np.ravel(model.coef_))
        kind = "|coefficient|"
    elif hasattr(model, "feature_importances_"):
        weights = np.asarray(model.feature_importances_)
        kind = "split gain"
    else:
        # e.g. MLPClassifier — no built-in attribution at all.
        return pd.DataFrame(columns=["feature", "weight", "kind"])

    return (
        pd.DataFrame({"feature": list(feature_names), "weight": weights, "kind": kind})
        .sort_values("weight", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def risk_segments(
    probabilities, actual=None, customer_ids=None, low: float = 0.4, high: float = 0.7
) -> pd.DataFrame:
    """Cut propensity scores into Low / Medium / High risk.

    This is the deliverable — not "churn: yes/no", but a score per customer that
    marketing can sort, cut and act on. The cut-offs are a *business* decision:
    they trade retention budget spent on false positives against customers lost
    to false negatives.
    """
    probabilities = np.asarray(probabilities)
    frame = pd.DataFrame(
        {
            "CustomerID": np.arange(len(probabilities)) if customer_ids is None else np.asarray(customer_ids),
            "Propensity": probabilities,
        }
    )
    if actual is not None:
        frame["Actual"] = np.asarray(actual)
    frame["Segment"] = pd.cut(
        frame["Propensity"],
        bins=[-0.001, low, high, 1.0],
        labels=["Low risk", "Medium risk", "High risk"],
    )
    return frame


def segment_summary(segments: pd.DataFrame) -> pd.DataFrame:
    """Customers, share and (if known) realised churn rate per risk segment."""
    aggregations = {"Customers": ("Propensity", "size"), "Mean propensity": ("Propensity", "mean")}
    if "Actual" in segments.columns:
        aggregations["Actual churn rate"] = ("Actual", "mean")

    summary = segments.groupby("Segment", observed=False).agg(**aggregations)
    summary.insert(1, "Share", summary["Customers"] / summary["Customers"].sum())
    return summary.reset_index()


# ===========================================================================
# The look-alike time design (slide: observation -> buffer -> outcome)
# ===========================================================================
def lookalike_timeline(
    observation_weeks: int = 12, buffer_weeks: int = 2, outcome_weeks: int = 4
) -> pd.DataFrame:
    """Lay out the three windows of a look-alike design on a shared timeline.

    Returns one row per window with ``start``/``end`` in weeks, ready for a Gantt
    style chart.

    Why the design matters more than the algorithm:
    - **Observation** — the only data the model may see. Everything from *after*
      the cut-off is forbidden, however predictive it looks.
    - **Buffer** — the lead time the business actually needs. If retention needs
      two weeks to act, a model that predicts churn one day ahead is worthless.
      It also blocks label leakage: the final days before a cancellation are full
      of give-away signals (auto-renew switched off, a "how do I close my
      account" ticket).
    - **Outcome** — the window in which the event counts as a "yes", i.e. where
      the label is read.
    """
    rows = [
        {
            "window": "Observation",
            "start": 0,
            "end": observation_weeks,
            "role": "Build features from behaviour here",
        },
        {
            "window": "Buffer",
            "start": observation_weeks,
            "end": observation_weeks + buffer_weeks,
            "role": "Lead time to act + protection against label leakage",
        },
        {
            "window": "Outcome",
            "start": observation_weeks + buffer_weeks,
            "end": observation_weeks + buffer_weeks + outcome_weeks,
            "role": "Did the customer churn here? -> the label",
        },
    ]
    frame = pd.DataFrame(rows)
    frame["weeks"] = frame["end"] - frame["start"]
    return frame


def propensity_pipeline_dot() -> str:
    """Graphviz DOT of the propensity workflow, from history to campaign.

    ``st.graphviz_chart`` renders a raw DOT string client-side, so this needs no
    Graphviz Python package.
    """
    return """
digraph propensity {
  rankdir=LR;
  bgcolor="transparent";
  node [fontname="Helvetica", shape=box, style="rounded,filled", fillcolor="#ffffff", color="#90a4ae"];
  edge [fontname="Helvetica", fontsize=10, color="#607d8b"];

  history   [label="Historical profiles\\n(observation window)", fillcolor="#e3f2fd"];
  label     [label="Known outcome\\n(outcome window)", fillcolor="#e3f2fd"];
  train     [label="Train classifier\\nLR / XGBoost / NN"];
  validate  [label="Cross-validate\\n+ tune", fillcolor="#fff8e1"];
  score     [label="Score today's customers"];
  segments  [label="Risk segments\\nLow / Medium / High", fillcolor="#c8e6c9"];
  action    [label="Campaign", fillcolor="#c8e6c9"];

  history -> train;
  label -> train [label="supervision"];
  train -> validate -> score -> segments -> action;
}
""".strip()
