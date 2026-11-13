"""
Guard against silent drift between the app's re-implemented utils and the
sibling lesson scripts they mirror.

``propensity_utils``, ``survival_utils`` and ``recommender_utils`` *re-implement*
logic from ``# %%`` scripts that cannot be imported. Nothing in the unit tests
would notice if a sibling changed a hyperparameter, so each value below is read
straight out of the sibling's source with a regex and compared with what the app
actually runs (read from the live function signatures / model params).

Every entry carries a status:

- ``"same"``        the app must use exactly the sibling's value;
- ``"intentional"`` the app deliberately differs — usually to keep a *live demo*
                    fast — and records both values. The test still pins the
                    sibling's value, so if the sibling changes, this fails and
                    someone has to decide again rather than drift silently.

When a test here fails: look at the sibling change, then update the app (for
"same") or the recorded sibling value (for "intentional"). Skipped automatically
when a sibling repo is not checked out (they are git-ignored).
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

from utils import propensity_utils as P
from utils import recommender_utils as R
from utils import survival_utils as S

ROOT = Path(__file__).resolve().parents[1]


def sibling_value(relative_path: str, pattern: str, group: int = 1):
    """The first regex match in a sibling file, parsed as a Python literal."""
    path = ROOT / relative_path
    if not path.exists():
        pytest.skip(f"sibling file {relative_path} not checked out")
    match = re.search(pattern, path.read_text())
    assert match, f"pattern {pattern!r} no longer matches {relative_path} — the sibling was restructured"
    return ast.literal_eval(match.group(group))


def default(fn, name: str):
    return inspect.signature(fn).parameters[name].default


def model_param(model_name: str, param: str):
    model = P.build_model(model_name)
    return model.get_params()[f"model__{param}" if hasattr(model, "named_steps") else param]


PROP = "predictions/propensity.py"
SURV = "predictions/survival.py"
CF = "recommender/collaborative_filtering.py"
AR = "recommender/association_rule_mining.py"

# (label, sibling file, regex, recorded sibling value, app value getter, status)
COUPLINGS = [
    # --- propensity ---------------------------------------------------------
    ("propensity: test split share", PROP, r"test_size=([\d.]+), random_state=RANDOM_STATE, stratify=y",
     0.2, lambda: default(P.stratified_split, "test_size"), "same"),
    ("propensity: CV folds", PROP, r"StratifiedKFold\(n_splits=(\d+)",
     5, lambda: default(P.cross_validate_model, "n_splits"), "same"),
    ("propensity: logistic max_iter", PROP, r"LogisticRegression\(random_state=RANDOM_STATE, max_iter=(\d+)\)",
     1000, lambda: model_param("Logistic Regression", "max_iter"), "same"),
    ("propensity: MLP hidden layers", PROP, r"hidden_layer_sizes=(\(\d+, \d+\)),\n\s+activation",
     (32, 16), lambda: model_param("Neural Network", "hidden_layer_sizes"), "same"),
    ("propensity: MLP alpha", PROP, r"activation=\"relu\",\n\s+alpha=([\d.]+)",
     0.1, lambda: model_param("Neural Network", "alpha"), "same"),
    ("propensity: NN search grid alpha", PROP, r"\"model__alpha\": (\[[^\]]*\])",
     [0.0001, 0.01, 0.1], lambda: P.MODEL_SPECS["Neural Network"]["grid"]["alpha"], "same"),
    ("propensity: XGB search grid depth", PROP, r"\"max_depth\": (\[[^\]]*\])",
     [3, 5, 7], lambda: P.MODEL_SPECS["XGBoost"]["grid"]["max_depth"], "same"),
    # (risk cut points have two values — see test_risk_cut_points_match)
    # Live-demo choices: the untuned XGBoost is stronger in the app, and the
    # searches/curves are lighter so a button press answers in seconds.
    ("propensity: untuned XGB max_depth", PROP, r"max_depth=(\d+), learning_rate=[\d.]+, n_estimators=\d+",
     2, lambda: model_param("XGBoost", "max_depth"), "intentional"),
    ("propensity: untuned XGB n_estimators", PROP, r"max_depth=\d+, learning_rate=[\d.]+, n_estimators=(\d+)",
     100, lambda: model_param("XGBoost", "n_estimators"), "intentional"),
    ("propensity: XGB search n_iter", PROP, r"n_iter=(\d+),\n\s+cv=3",
     10, lambda: default(P.tune_model, "n_iter"), "intentional"),
    ("propensity: logistic search grid C", PROP, r"\"model__C\": (\[[^\]]*\])",
     [0.01, 0.1, 1, 10], lambda: P.MODEL_SPECS["Logistic Regression"]["grid"]["C"], "intentional"),
    # --- survival -----------------------------------------------------------
    ("survival: test split share", SURV, r"test_size=([\d.]+), random_state=RANDOM_STATE, stratify=df",
     0.3, lambda: default(S.split_survival, "test_size"), "same"),
    ("survival: Cox alpha", SURV, r"CoxPHSurvivalAnalysis\(alpha=([\d.]+)\)",
     0.1, lambda: default(S.fit_cox, "alpha"), "same"),
    ("survival: RSF trees", SURV, r"n_estimators=(\d+), max_depth=\d+, min_samples_leaf",
     50, lambda: default(S.fit_rsf, "n_estimators"), "same"),
    ("survival: RSF depth", SURV, r"n_estimators=\d+, max_depth=(\d+), min_samples_leaf",
     5, lambda: default(S.fit_rsf, "max_depth"), "same"),
    ("survival: SVM alpha", SURV, r"rank_ratio=1\.0, alpha=([\d.]+)",
     0.01, lambda: default(S.fit_svm, "alpha"), "same"),
    ("survival: SVM max_iter", SURV, r"rank_ratio=1\.0, alpha=[\d.]+, max_iter=(\d+)",
     50, lambda: default(S.fit_svm, "max_iter"), "same"),
    ("propensity: high-bias margin", PROP, r"best_val_level - lr_val_level > ([\d.]+)",
     0.02, lambda: P.BIAS_MARGIN, "same"),
    ("propensity: flat validation curve", PROP, r"vc_means.max\(\) - vc_means.min\(\) < ([\d.]+)",
     0.005, lambda: P.FLAT_CURVE_SPREAD, "same"),
    # --- recommender --------------------------------------------------------
    ("recommender: user-based neighbours", CF, r"k_neighbors=(\d+), min_neighbors=\d+",
     30, lambda: default(R.get_user_based_recommendations, "k_neighbors"), "same"),
    ("recommender: min neighbours who rated", CF, r"k_neighbors=\d+, min_neighbors=(\d+)",
     5, lambda: default(R.get_user_based_recommendations, "min_neighbors"), "same"),
    ("recommender: baseline reg", CF, r"def fit_baseline\(ratings, n_users, n_items, reg=([\d.]+)",
     10.0, lambda: default(R.fit_baseline, "reg"), "same"),
    ("recommender: baseline passes", CF, r"def fit_baseline\(.*n_iters=(\d+)\)",
     15, lambda: default(R.fit_baseline, "n_iters"), "same"),
    ("recommender: MF factors", CF, r"n_factors=(\d+), n_epochs=\d+,\n",
     20, lambda: default(R.MatrixFactorization.__init__, "n_factors"), "same"),
    ("recommender: MF epochs", CF, r"n_factors=\d+, n_epochs=(\d+),\n",
     20, lambda: default(R.MatrixFactorization.__init__, "n_epochs"), "same"),
    ("recommender: MF learning rate", CF, r"lr=([\d.]+), reg=[\d.]+, random_state",
     0.01, lambda: default(R.MatrixFactorization.__init__, "lr"), "same"),
    ("recommender: MF regularisation", CF, r"lr=[\d.]+, reg=([\d.]+), random_state",
     0.05, lambda: default(R.MatrixFactorization.__init__, "reg"), "same"),
    ("recommender: rule min confidence", AR, r"min_threshold = ([\d.]+)",
     0.1, lambda: default(R.make_rules, "min_confidence"), "same"),
    # Shorter itemsets keep Apriori fast enough to run on a button press.
    ("recommender: Apriori max_len", AR, r"max_len=(\d+)",
     5, lambda: default(R.run_apriori, "max_len"), "intentional"),
]

REGULAR = COUPLINGS


@pytest.mark.parametrize("label, path, pattern, recorded, app_value, status", REGULAR, ids=[c[0] for c in REGULAR])
def test_sibling_value_is_still_what_we_recorded(label, path, pattern, recorded, app_value, status):
    """If this fails, the sibling changed: re-decide whether the app follows."""
    assert sibling_value(path, pattern) == recorded, f"{label}: the sibling changed this value"


@pytest.mark.parametrize(
    "label, path, pattern, recorded, app_value, status",
    [c for c in REGULAR if c[5] == "same"],
    ids=[c[0] for c in REGULAR if c[5] == "same"],
)
def test_app_matches_the_sibling(label, path, pattern, recorded, app_value, status):
    assert app_value() == recorded, f"{label}: app uses {app_value()!r}, sibling uses {recorded!r}"


@pytest.mark.parametrize(
    "label, path, pattern, recorded, app_value, status",
    [c for c in REGULAR if c[5] == "intentional"],
    ids=[c[0] for c in REGULAR if c[5] == "intentional"],
)
def test_intentional_divergences_are_still_divergent(label, path, pattern, recorded, app_value, status):
    """If app and sibling now agree, promote the entry to "same" so it stays guarded."""
    assert app_value() != recorded, f"{label}: no longer differs — mark it 'same'"


def test_risk_cut_points_match():
    low = sibling_value(PROP, r"bins=\[0, ([\d.]+), ([\d.]+), 1\.0\]", 1)
    high = sibling_value(PROP, r"bins=\[0, ([\d.]+), ([\d.]+), 1\.0\]", 2)
    assert (low, high) == (default(P.risk_segments, "low"), default(P.risk_segments, "high"))


def test_segment_action_table_only_names_real_ladder_segments():
    """The action table keys on RFM segment names it does not own (they live in
    segmentation/src/rfm_scoring.py). A renamed segment would silently drop rows."""
    from utils import segmentation_utils as SEG

    if not SEG.SIBLING_AVAILABLE:
        pytest.skip("segmentation sibling not checked out")
    import pandas as pd

    names = {name for name, _ in SEG.LADDER_RULES} | {"Hibernating"}
    every_cell = pd.DataFrame(1, index=sorted(names), columns=["Low Value", "Medium Value", "High Value", "Premium"])
    every_cell.index.name = "Segment"
    used = set(SEG.segment_action_table(every_cell)["RFM segment"])
    source = inspect.getsource(SEG.segment_action_table)
    referenced = set(re.findall(r'\("([^"]+)", "(?:Low Value|Medium Value|High Value|Premium)"\)', source))
    assert referenced, "could not read the action table's keys"
    assert referenced <= names, f"action table names unknown segments: {referenced - names}"
    assert used == referenced


def test_object_detection_results_were_exported_from_the_notebook_settings():
    """Page 9 shows exported results; a notebook re-run with new settings needs a re-export."""
    import json

    notebook = ROOT / "object-detection" / "showcase.ipynb"
    exported = ROOT / "object-detection" / "results" / "detections.json"
    if not (notebook.exists() and exported.exists()):
        pytest.skip("object-detection sibling or its results not present")
    source = "".join("".join(c["source"]) for c in json.loads(notebook.read_text())["cells"])
    settings = json.loads(exported.read_text())["training"]["settings"]
    for name, key in [("EPOCHS", "epochs"), ("BATCH", "batch"), ("IMG_SIZE", "imgsz")]:
        value = int(re.search(rf"^{name}\s*=\s*(\d+)", source, re.MULTILINE).group(1))
        assert value == settings[key], f"{name}: notebook {value}, exported {settings[key]} — re-run export_results.py"
