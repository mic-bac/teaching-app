"""
forecasting_utils.py
--------------------
UI-free, importable logic for the Time Series page (and, next, the Forecasting
page) — a thin adapter over the sibling ``predictions/src/`` package.

**This module imports from the sibling rather than re-implementing it**, like
``segmentation_utils``: ``predictions/src/{sales_data, ts_core, forecast_models,
control_series}`` are already pure, Streamlit-free modules — the same splits,
metrics, baselines and model wrappers ``predictions/timeseries.py`` prints the
source of as it runs. Copying them would duplicate the single source of truth.

Two details of how it is imported:

- ``forecast_models`` imports ``ts_core`` *relatively* (``from .ts_core ...``), so
  the directory is registered as a package (``teaching.load_sibling_package``)
  rather than put on ``sys.path`` — which also keeps it from colliding with the
  segmentation sibling's own ``src`` package in the same Streamlit process.
- Only what the *page* needs on top lives here: plot-ready frames, the lesson's
  cross-validation loop and Christmas check (they are inline in the script, not
  in ``src``), the slide's Excel example, and a flow diagram.

Graceful degradation: the sibling is git-ignored, so it may be missing or older
than this app (``SIBLING_AVAILABLE`` / ``SIBLING_ERROR``); and when it is present
but the Walmart CSVs are not, its own loader falls back to synthetic sales that
mimic the measured dynamics, and says so (``source == "synthetic"``).
"""

from __future__ import annotations

import importlib
import inspect

import numpy as np
import pandas as pd

from utils.teaching import load_sibling_package

# ===========================================================================
# Sibling bootstrap
# ===========================================================================

try:  # pragma: no cover - exercised by the missing-sibling path
    _PKG = load_sibling_package("predictions", "src")
    tc = importlib.import_module(f"{_PKG}.ts_core")
    fm = importlib.import_module(f"{_PKG}.forecast_models")
    _sales = importlib.import_module(f"{_PKG}.sales_data")
    _control = importlib.import_module(f"{_PKG}.control_series")
    fp = importlib.import_module(f"{_PKG}.forecast_panel")

    # Re-exports — one import surface for the page. An outdated sibling checkout
    # imports fine but lacks newer names; that is reported, not crashed on.
    TEST_WEEKS = tc.TEST_WEEKS
    SEASONAL_PERIOD = tc.SEASONAL_PERIOD
    as_series = tc.as_series
    suitability_report = tc.suitability_report
    trend_test = tc.trend_test
    decompose = tc.decompose
    adf_report = tc.adf_report
    autocorrelation = tc.autocorrelation
    split_series = tc.split_series
    rolling_origin_splits = tc.rolling_origin_splits
    forecast_metrics = tc.forecast_metrics
    metrics_table = tc.metrics_table
    naive_forecast = tc.naive_forecast
    seasonal_naive_forecast = tc.seasonal_naive_forecast
    simple_exponential_smoothing = tc.simple_exponential_smoothing
    fit_ets = fm.fit_ets
    fit_sarima = fm.fit_sarima
    deflate = _sales.deflate
    prepare_sales = _sales.prepare
    load_control_series = _control.load_control_series

    # Forecasting lesson (Prophet, XGBoost, panels)
    WALMART_HOLIDAYS = _sales.WALMART_HOLIDAYS
    EXTERNAL_FEATURES = _sales.EXTERNAL_FEATURES
    split_panel = tc.split_panel
    fit_prophet = fm.fit_prophet
    prophet_forecast_only = fm.prophet_forecast_only
    prophet_components = fm.prophet_components
    time_ordered_validation_split = fm.time_ordered_validation_split
    fit_xgboost = fm.fit_xgboost
    fit_random_forest = fm.fit_random_forest
    fit_hybrid = fm.fit_hybrid
    build_features = fp.build_features
    score_panel = fp.score_panel
    ets_per_store = fp.ets_per_store
    CALENDAR = fp.CALENDAR
    LAGS = fp.LAGS

    SIBLING_AVAILABLE = True
    SIBLING_ERROR = ""
except (ImportError, FileNotFoundError, AttributeError) as exc:  # pragma: no cover
    SIBLING_AVAILABLE = False
    SIBLING_ERROR = str(exc)


# ===========================================================================
# Data
# ===========================================================================


def load_chain() -> dict:
    """The lesson's data in one call: chain sales, CPI, the promotion break, source.

    ``prepare`` returns three grains; the time-series lesson forecasts the chain
    total (every store summed into one weekly number — the series a board plans
    against) and needs the department panel only to locate the promotion data.
    """
    chain, _stores, depts, source = prepare_sales(verbose=False)
    sales = as_series(chain)
    return {
        "sales": sales,
        "cpi": as_series(chain, value="CPI"),
        "markdown": markdown_break(depts, sales.index),
        "source": source,
    }


def markdown_break(dept_panel: pd.DataFrame, weeks: pd.DatetimeIndex) -> dict:
    """Where promotion (MarkDown) tracking starts — a structural break, not missing data."""
    columns = [c for c in dept_panel.columns if c.startswith("MarkDown")]
    recorded = dept_panel.dropna(subset=columns, how="all")["Date"]
    first = pd.Timestamp(recorded.min())
    without = int((weeks < first).sum())
    return {"columns": columns, "first_recorded": first, "weeks_without": without, "n_weeks": len(weeks)}


# ===========================================================================
# The slide's Excel example (slides 70 / 78), computed by the sibling's code
# ===========================================================================

SLIDE_SERIES = [3, 5, 9, 20, 12, 17, 22, 23, 51, 41, 56]


def slide_smoothing_table(alpha: float = 0.4) -> tuple[pd.DataFrame, dict]:
    """The lecture's exponential-smoothing spreadsheet, row for row.

    The forecasts come from the sibling's ``simple_exponential_smoothing`` and the
    error measures from ``forecast_metrics`` — at ``alpha=0.4`` they reproduce the
    slide exactly (MAE 10.20, MSE 184.89, RMSE 13.60; the slide's MAPE 0.39 is
    the same 38.6% written as a fraction). The first row has no forecast to
    score: its "prediction" is just the first value, used to start the recursion.
    """
    y = pd.Series(SLIDE_SERIES, index=range(1, len(SLIDE_SERIES) + 1), dtype=float)
    pred = simple_exponential_smoothing(y, alpha=alpha)
    error = (y - pred).abs()
    table = pd.DataFrame(
        {"t": y.index, "y": y.values, "pred": pred.values, "abs_error": error.values,
         "abs_pct_error": (error / y).values, "squared_error": (error**2).values}
    )
    table.loc[0, ["abs_error", "abs_pct_error", "squared_error"]] = np.nan
    return table, forecast_metrics(y.iloc[1:], pred.iloc[1:])


# ===========================================================================
# Models and evaluation — the lesson's inline steps, as functions
# ===========================================================================


def sarima_label() -> str:
    """``SARIMA(p,d,q)(P,D,Q,m)`` read from the sibling's ``fit_sarima`` defaults.

    Derived, not typed: if the lesson changes its orders, the label follows.
    """
    params = inspect.signature(fit_sarima).parameters
    order = params["order"].default
    seasonal = params["seasonal_order"].default
    return f"SARIMA({','.join(map(str, order))})({','.join(map(str, seasonal))})"


def baseline_forecasts(train: pd.Series, horizon: int) -> dict[str, np.ndarray]:
    """The three zero-parameter forecasts every model has to beat."""
    return {
        "naive (last value)": naive_forecast(train, horizon),
        "seasonal naive (t-52)": seasonal_naive_forecast(train, horizon),
        "mean of history": np.repeat(train.mean(), horizon),
    }


def smoothing_sweep(sales: pd.Series, alphas=(0.2, 0.4, 0.8)) -> pd.DataFrame:
    """One-step-ahead error of simple exponential smoothing for several alphas.

    The point is how little alpha matters: the series is made of seasonality and
    simple smoothing has no seasonal term to put it in.
    """
    rows = []
    for alpha in alphas:
        smoothed = simple_exponential_smoothing(sales, alpha=alpha)
        error = forecast_metrics(sales.iloc[1:], smoothed.iloc[1:])
        rows.append({"alpha": alpha, "MAE": error["MAE"], "MAPE": error["MAPE"]})
    return pd.DataFrame(rows)


def cross_validate(sales: pd.Series, n_folds: int = 3, horizon: int | None = None) -> pd.DataFrame:
    """Rolling-origin CV of the seasonal-naive baseline and Holt-Winters.

    One row per fold: training length, test window, and the MAPE of both. The
    training window only ever grows, so no fold predicts a week an earlier fold
    trained on.
    """
    horizon = horizon or TEST_WEEKS
    rows = []
    for i, (fold_train, fold_test) in enumerate(rolling_origin_splits(sales, n_folds=n_folds, horizon=horizon)):
        baseline = seasonal_naive_forecast(fold_train, len(fold_test))
        _, ets = fit_ets(fold_train, len(fold_test))
        rows.append({
            "fold": i,
            "train_weeks": len(fold_train),
            "test_from": fold_test.index[0].date(),
            "test_to": fold_test.index[-1].date(),
            "seasonal naive MAPE": forecast_metrics(fold_test, baseline)["MAPE"],
            "Holt-Winters MAPE": forecast_metrics(fold_test, ets.values)["MAPE"],
        })
    return pd.DataFrame(rows)


def christmas_check(sales: pd.Series) -> dict:
    """Why the hardest weeks of the year cannot be evaluated with this data.

    A 52-week seasonal model needs two full cycles (104 weeks) before it can be
    fitted, and by then every Christmas in the file is already in the training
    data. The seasonal-naive baseline needs only 52 weeks, so it CAN be scored on
    a window containing Christmas — and that shows how much the quiet test window
    flatters every headline number.
    """
    week_of_year = sales.index.isocalendar().week.to_numpy()
    christmas = sales.index[np.isin(week_of_year, [51, 52])]
    earliest_testable = sales.index[2 * SEASONAL_PERIOD]

    train, test = split_series(sales, test_weeks=TEST_WEEKS)
    quiet = forecast_metrics(test, seasonal_naive_forecast(train, len(test)))["MAPE"]

    starts = [sales.index.get_loc(w) - 6 for w in christmas if sales.index.get_loc(w) - 6 > SEASONAL_PERIOD]
    result = {"christmas_weeks": list(christmas), "earliest_testable": earliest_testable,
              "quiet_window": (test.index[0], test.index[-1]), "quiet_mape": quiet,
              "christmas_window": None, "christmas_mape": None}
    if starts:
        start = max(starts)
        c_test, c_train = sales.iloc[start : start + TEST_WEEKS], sales.iloc[:start]
        result["christmas_window"] = (c_test.index[0], c_test.index[-1])
        result["christmas_mape"] = forecast_metrics(
            c_test, seasonal_naive_forecast(c_train, len(c_test))
        )["MAPE"]
    return result


def timeseries_flow_dot() -> str:
    """The lesson's order of work as a Graphviz DOT string (rendered client-side).

    The order is the lesson: diagnose before modelling, and report every model
    next to a baseline.
    """
    return """
digraph ts {
  rankdir=LR;
  node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=10 fillcolor="#ECEFF1"];
  edge [color="#616161"];
  data   [label="Weekly chain sales\\n143 weeks" fillcolor="#BBDEFB"];
  report [label="Forecastable?\\nsuitability report" fillcolor="#FFF9C4"];
  parts  [label="Decompose\\ntrend + season + residual"];
  stat   [label="Stationarity (ADF)\\nACF / PACF"];
  split  [label="Split by time\\nlast 13 weeks = test"];
  base   [label="Baselines\\nnaive, seasonal naive" fillcolor="#FFE0B2"];
  models [label="Models\\nsmoothing, Holt-Winters, SARIMA"];
  cv     [label="Rolling-origin CV\\n+ the Christmas problem"];
  verdict [label="Compare — always\\nagainst the baseline" shape=note fillcolor="#C8E6C9"];
  data -> report -> parts -> stat -> split -> base -> models -> cv -> verdict;
}
"""


# ===========================================================================
# Forecasting lesson (Prophet & XGBoost) — the script's inline steps as functions
# ===========================================================================

PANEL_FEATURES = None  # set below once the sibling is available
if SIBLING_AVAILABLE:
    PANEL_FEATURES = ["Store", "Size", "Type"] + CALENDAR + LAGS
    DEPT_FEATURES = ["Store", "Dept", "Size", "Type"] + CALENDAR + LAGS
    NO_LAGS = ["Store", "Size", "Type"] + CALENDAR


def load_grains() -> dict:
    """The same sales at three levels: chain total, 45 stores, ~3,300 departments."""
    chain, stores, depts, source = prepare_sales(verbose=False)
    return {"chain": chain, "stores": stores, "depts": depts, "source": source}


def prophet_run(train: pd.Series, horizon: int, changepoint_prior_scale: float = 0.05) -> dict:
    """Fit Prophet with the Walmart holidays; return forecast, interval and components.

    Everything returned is plain data (no model object), so a page can cache it.
    ``trend_drift_pct`` is how far the trend moves over the forecast horizon —
    the number that shows *overshooting* when the changepoint setting is loose.
    """
    model, forecast = fit_prophet(train, horizon, holidays=WALMART_HOLIDAYS, country_holidays="US",
                                  changepoint_prior_scale=changepoint_prior_scale)
    future = forecast["ds"].iloc[-horizon:]
    predicted = prophet_forecast_only(forecast, future)
    trend = forecast.set_index("ds")["trend"]
    return {
        "forecast": forecast,
        "predicted": predicted,
        "components": prophet_components(model, forecast),
        "trend_drift_pct": 100 * (trend.loc[future.iloc[-1]] / trend.loc[future.iloc[0]] - 1),
    }


def interval_coverage(actual: pd.Series, predicted: pd.DataFrame) -> float:
    """Share of actual weeks inside Prophet's uncertainty band."""
    a = actual.to_numpy()
    return float(np.mean((a >= predicted["yhat_lower"].to_numpy()) & (a <= predicted["yhat_upper"].to_numpy())))


def chain_tree_models(chain: pd.DataFrame, train_end: pd.Timestamp) -> dict:
    """XGBoost and a random forest on the single chain series — too few rows.

    Returns predictions for the test weeks and the number of rows the trees
    actually trained on (143 weeks − 52 eaten by ``lag_52`` − the test window).
    """
    features = build_features(chain)
    fit_rows = features[features["Date"] <= train_end]
    test_rows = features[features["Date"] > train_end]
    columns = CALENDAR + LAGS
    a, b, c, d = time_ordered_validation_split(fit_rows[columns], fit_rows["Weekly_Sales"])
    return {
        "XGBoost": fit_xgboost(a, c, b, d).predict(test_rows[columns]),
        "Random Forest": fit_random_forest(fit_rows[columns], fit_rows["Weekly_Sales"]).predict(test_rows[columns]),
        "train_rows": len(fit_rows),
        "features": features,
    }


def store_panel_run(store_panel: pd.DataFrame) -> dict:
    """One XGBoost for all 45 stores: accuracy, speed, early stopping, importance."""
    panel = build_features(store_panel, group="Store")
    train, test = split_panel(panel, test_weeks=TEST_WEEKS)
    a, b, c, d = time_ordered_validation_split(train[PANEL_FEATURES], train["Weekly_Sales"])
    import time

    start = time.perf_counter()
    model = fit_xgboost(a, c, b, d)
    seconds = time.perf_counter() - start
    prediction = model.predict(test[PANEL_FEATURES])
    return {
        "panel": panel, "train": train, "test": test, "split": (a, b, c, d),
        "train_rows": len(train),
        "seconds": seconds,
        "scores": {
            "seasonal naive (t-52)": forecast_metrics(test["Weekly_Sales"], test["lag_52"]),
            "XGBoost (one model)": forecast_metrics(test["Weekly_Sales"], prediction),
        },
        "importance": pd.Series(model.feature_importances_, index=PANEL_FEATURES).sort_values(),
        "n_estimators": model.n_estimators,
        "best_iteration": int(model.best_iteration),
    }


def depth_sweep(run: dict, depths=(2, 4, 6, 10)) -> pd.DataFrame:
    """Tree depth vs training and test error — the bias-variance trade-off, measured."""
    a, b, c, d = run["split"]
    rows = []
    for depth in depths:
        model = fit_xgboost(a, c, b, d, max_depth=depth)
        rows.append({
            "max_depth": depth,
            "train MAPE": forecast_metrics(c, model.predict(a))["MAPE"],
            "test MAPE": forecast_metrics(run["test"]["Weekly_Sales"], model.predict(run["test"][PANEL_FEATURES]))["MAPE"],
        })
    frame = pd.DataFrame(rows)
    frame["gap"] = frame["test MAPE"] - frame["train MAPE"]
    return frame


def external_features_cv(panel: pd.DataFrame) -> pd.DataFrame:
    """Three folds with and without temperature, fuel price, CPI and unemployment."""
    rows = []
    for name, columns in [("calendar + lags", PANEL_FEATURES),
                          ("+ temperature, fuel, CPI, unemployment", PANEL_FEATURES + EXTERNAL_FEATURES)]:
        scores = score_panel(panel, columns)
        rows.append({"feature set": name, **{f"fold {i}": v for i, v in enumerate(scores)}, "mean": np.mean(scores)})
    return pd.DataFrame(rows)


def department_grain(dept_panel: pd.DataFrame) -> dict:
    """The same model per store × department — where MAPE stops being usable."""
    features = build_features(dept_panel, group=["Store", "Dept"])
    train, test = split_panel(features, test_weeks=TEST_WEEKS)
    prediction = fit_xgboost(train[DEPT_FEATURES], train["Weekly_Sales"]).predict(test[DEPT_FEATURES])
    summed = test.assign(prediction=prediction).groupby("Date")[["Weekly_Sales", "prediction"]].sum()
    return {
        "series": features.groupby(["Store", "Dept"]).ngroups,
        "model": forecast_metrics(test["Weekly_Sales"], prediction),
        "baseline": forecast_metrics(test["Weekly_Sales"], test["lag_52"]),
        "summed": forecast_metrics(summed["Weekly_Sales"], summed["prediction"]),
        "small_rows": int((test["Weekly_Sales"] < 100).sum()),
        "test_rows": len(test),
    }


def detrending_experiment(panel: pd.DataFrame) -> pd.DataFrame:
    """Plain XGBoost vs trend + XGBoost, with and without lag features (3 folds each)."""
    rows = []
    for lags, columns in [("with lags", PANEL_FEATURES), ("no lags", NO_LAGS)]:
        for hybrid in (False, True):
            scores = score_panel(panel, columns, hybrid=hybrid)
            rows.append({"features": lags, "model": "trend + XGBoost" if hybrid else "plain XGBoost",
                         **{f"fold {i}": v for i, v in enumerate(scores)}, "mean": np.mean(scores)})
    return pd.DataFrame(rows)


def control_detrending() -> dict:
    """The same comparison on weekly CO₂ — a series with a real trend (52-week horizon)."""
    control = load_control_series()
    frame = pd.DataFrame({"Date": control.index, "Weekly_Sales": control.to_numpy()})
    features = build_features(frame)
    train, test = split_panel(features, test_weeks=52)
    columns = ["week", "week_sin", "week_cos", "t"]
    plain = fit_xgboost(train[columns], train["Weekly_Sales"]).predict(test[columns])
    _, hybrid = fit_hybrid(train, test, columns)
    return {
        "dates": test["Date"], "actual": test["Weekly_Sales"].to_numpy(), "plain": plain, "hybrid": hybrid,
        "history": train[["Date", "Weekly_Sales"]].tail(156),
        "plain_mape": forecast_metrics(test["Weekly_Sales"], plain)["MAPE"],
        "hybrid_mape": forecast_metrics(test["Weekly_Sales"], hybrid)["MAPE"],
    }
