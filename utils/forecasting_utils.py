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
    seasonal_naive_growth_forecast = tc.seasonal_naive_growth_forecast
    easter_weeks = tc.easter_weeks
    calendar_events = tc.calendar_events
    explain_misses = tc.explain_misses
    cv_interval = tc.cv_interval
    with_easter = tc.with_easter
    holt_winters_parts = fm.holt_winters_parts
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


GROWTH_LABEL = "seasonal naive × growth"


def baseline_forecasts(train: pd.Series, horizon: int) -> dict[str, np.ndarray]:
    """The zero-parameter forecasts every model has to beat, plus the one-number fix.

    The fourth is "same week last year × recent growth" — still no fitting, just
    one ratio — the explainable step between the baseline and Holt-Winters.
    """
    return {
        "naive (last value)": naive_forecast(train, horizon),
        "seasonal naive (t-52)": seasonal_naive_forecast(train, horizon),
        "mean of history": np.repeat(train.mean(), horizon),
        GROWTH_LABEL: seasonal_naive_growth_forecast(train, horizon),
    }


def growth_example(train: pd.Series, window: int | None = None) -> dict:
    """The growth factor's arithmetic, for the first forecast week.

    Mirrors ``seasonal_naive_growth_forecast`` step by step, so the page can show
    the napkin calculation with real numbers.
    """
    window = window or TEST_WEEKS
    recent = float(train.iloc[-window:].sum())
    year_before = float(train.iloc[-window - SEASONAL_PERIOD : -SEASONAL_PERIOD].sum())
    copied = float(train.iloc[len(train) - SEASONAL_PERIOD])
    return {
        "window": window, "recent": recent, "year_before": year_before,
        "growth": recent / year_before, "copied_week": train.index[len(train) - SEASONAL_PERIOD],
        "copied": copied, "forecast": copied * recent / year_before,
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


CV_MODELS = ["seasonal naive", "naive × growth", "Holt-Winters", "SARIMA"]


def cross_validate(
    sales: pd.Series, n_folds: int = 3, horizon: int | None = None, include_sarima: bool = False
) -> dict:
    """Rolling-origin CV of every forecast in the lesson — the trustworthy comparison.

    Mirrors ``timeseries.py`` section 12. Returns:

    * ``scores`` — one row per fold: test window and each model's MAPE;
    * ``errors`` — every weekly % miss per model (``(actual - forecast) / actual``),
      which ``cv_interval`` turns into an honest ± band;
    * ``forecasts`` — per fold, the test weeks and each model's forecast, so the
      page can read the misses.

    SARIMA is optional because it costs ~20 s over three folds; the rest is instant.
    """
    horizon = horizon or TEST_WEEKS
    models = {
        "seasonal naive": lambda tr, h: seasonal_naive_forecast(tr, h),
        "naive × growth": lambda tr, h: seasonal_naive_growth_forecast(tr, h),
        "Holt-Winters": lambda tr, h: fit_ets(tr, h)[1].to_numpy(),
    }
    if include_sarima:
        models["SARIMA"] = lambda tr, h: fit_sarima(tr, h)[1].to_numpy()
    rows, errors, forecasts = [], {name: [] for name in models}, []
    for i, (fold_train, fold_test) in enumerate(rolling_origin_splits(sales, n_folds=n_folds, horizon=horizon)):
        row = {"fold": i, "train_weeks": len(fold_train),
               "test_from": fold_test.index[0].date(), "test_to": fold_test.index[-1].date()}
        predictions = {}
        for name, forecast_fn in models.items():
            prediction = np.asarray(forecast_fn(fold_train, len(fold_test)), dtype=float)
            predictions[name] = prediction
            row[name] = forecast_metrics(fold_test, prediction)["MAPE"]
            errors[name].extend(100 * (fold_test.to_numpy() - prediction) / fold_test.to_numpy())
        rows.append(row)
        forecasts.append({"test": fold_test, "predictions": predictions})
    return {"scores": pd.DataFrame(rows), "errors": errors, "forecasts": forecasts,
            "models": list(models)}


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


def prophet_calendar_cv(sales: pd.Series, n_folds: int = 3, horizon: int | None = None) -> dict:
    """Which holiday calendar helps Prophet? Mirrors ``forecasting.py`` section 5.

    Three calendars across the rolling-origin folds — Walmart's four holidays,
    plus the US public calendar, or plus Easter — with Holt-Winters (the
    previous lesson's best) as the bar. Also returns the miss on the Easter week
    inside the first fold, with and without Easter in the calendar: the one
    number that shows *why* the calendar matters.
    """
    horizon = horizon or TEST_WEEKS
    easter = with_easter(WALMART_HOLIDAYS, sales.index[0], sales.index[-1] + pd.Timedelta(weeks=60))
    calendars = {
        "Prophet": (WALMART_HOLIDAYS, None),
        "Prophet + US calendar": (WALMART_HOLIDAYS, "US"),
        "Prophet + Easter": (easter, None),
    }
    events = calendar_events(sales.index, WALMART_HOLIDAYS)
    scores = {name: [] for name in [*calendars, "Holt-Winters"]}
    easter_week, easter_miss = None, {}
    for i, (fold_train, fold_test) in enumerate(rolling_origin_splits(sales, n_folds=n_folds, horizon=horizon)):
        for name, (calendar, country) in calendars.items():
            _, forecast = fit_prophet(fold_train, len(fold_test), holidays=calendar, country_holidays=country)
            prediction = prophet_forecast_only(forecast, fold_test.index)["yhat"].to_numpy()
            scores[name].append(forecast_metrics(fold_test, prediction)["MAPE"])
            if i == 0:
                misses = explain_misses(fold_test, prediction, events)
                row = misses[misses["event this year"] == "Easter"]
                if not row.empty:
                    easter_week, easter_miss[name] = row["week"].iloc[0], float(row["miss %"].iloc[0])
        scores["Holt-Winters"].append(
            forecast_metrics(fold_test, fit_ets(fold_train, len(fold_test))[1].to_numpy())["MAPE"])
    table = pd.DataFrame(
        [{"model": name, **{f"fold {i}": v for i, v in enumerate(values)}, "mean": float(np.mean(values))}
         for name, values in scores.items()]
    )
    return {"scores": table, "easter_week": easter_week, "easter_miss": easter_miss}


def prophet_component_reading(components: dict) -> dict:
    """The biggest numbers in Prophet's parts, ready to be said in euros.

    Prophet is additive, so every component is an amount added to (or taken
    from) a week's sales — which makes "the season adds €X" literally true.
    """
    out = {}
    if "yearly" in components:
        yearly = components["yearly"].drop_duplicates("ds").set_index("ds")["yearly"]
        out["season_peak"] = (yearly.idxmax(), float(yearly.max()))
        out["season_low"] = (yearly.idxmin(), float(yearly.min()))
    if "holidays" in components:
        holidays = components["holidays"].set_index("ds")["holidays"]
        holidays = holidays[holidays.abs() > 0]
        if not holidays.empty:
            out["holiday_peak"] = (holidays.abs().idxmax(), float(holidays.loc[holidays.abs().idxmax()]))
    trend = components["trend"].set_index("ds")["trend"]
    out["trend_change"] = float(trend.iloc[-1] - trend.iloc[0])
    return out


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
    # One concrete row where MAPE "explodes": a small department week with a big
    # percentage miss that is tiny in euros. Positive sales only — MAPE skips the rest.
    small = test.assign(prediction=prediction)
    small = small[(small["Weekly_Sales"] > 0) & (small["Weekly_Sales"] < 100)]
    example = None
    if not small.empty:
        pct = (small["Weekly_Sales"] - small["prediction"]).abs() / small["Weekly_Sales"]
        row = small.loc[(pct - 5).abs().idxmin()]  # a typical "~500%" case, not the most extreme
        example = {"store": int(row["Store"]), "dept": int(row["Dept"]), "date": row["Date"].date(),
                   "actual": float(row["Weekly_Sales"]), "prediction": float(row["prediction"]),
                   "pct": float(100 * abs(row["Weekly_Sales"] - row["prediction"]) / row["Weekly_Sales"])}
    return {
        "example": example,
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


# ===========================================================================
# Plain-language readings — the numbers, said in a sentence
# ===========================================================================


def money(value: float) -> str:
    """Weekly sales are tens of millions: say €1.5M, not €1,523,442."""
    sign = "−" if value < 0 else ""
    value = abs(value)
    if value >= 1e6:
        return f"{sign}€{value / 1e6:,.1f}M"
    if value >= 1e3:
        return f"{sign}€{value / 1e3:,.0f}k"
    return f"{sign}€{value:,.0f}"


def mape_in_money(mape: float, sales: pd.Series) -> str:
    """A MAPE as euros on a typical week — what the percentage means for a planner."""
    typical = float(sales.mean())
    return (f"{mape:.2f}% of a typical {money(typical)} week is about "
            f"**{money(mape / 100 * typical)}** off, per week")


def report_readings(report: dict) -> dict[str, str]:
    """One plain sentence under each number of the suitability report."""
    return {
        "grid": (f"{report['n']} weeks, one every {report['step_days']} days, no gaps — nothing to "
                 "repair before modelling." if report["regular_grid"]
                 else "Gaps or uneven steps — they must be fixed before any model."),
        "cycles": (f"{report['n']} weeks ÷ 52 = {report['cycles']:.2f} years. Each week-of-the-year "
                   f"has been seen only {int(report['cycles'])}–{int(report['cycles']) + 1} times."),
        "trend": (f"{report['trend_pct_per_year']:+.2f}% a year, but p = {report['trend_p_value']:.2f}: "
                  + ("easily chance — treat it as no trend." if report["trend_p_value"] >= 0.05
                     else "unlikely to be chance — a real trend.")),
        "season": (f"On a 0–1 scale: {report['seasonal_strength']:.2f}. Once the slow trend is set "
                   "aside, almost all of what is left " if report["seasonal_strength"] >= 0.8
                   else f"On a 0–1 scale: {report['seasonal_strength']:.2f}. Once the slow trend is set "
                   "aside, only part of what is left ") + "is the pattern that repeats every year.",
        "stationary": ("The series keeps returning to the same level — no drift to remove."
                       if report["stationary"] else "The level drifts — models must remove it first."),
    }


def cv_reading(scores: pd.DataFrame, models: list[str]) -> list[str]:
    """Who wins each fold, and how consistent the overall winner is."""
    means = scores[models].mean()
    winner = means.idxmin()
    fold_winners = scores[models].idxmin(axis=1)
    lines = [
        f"**{winner}** has the lowest average error ({means[winner]:.2f}%) and wins "
        f"{int((fold_winners == winner).sum())} of {len(scores)} folds — a more trustworthy "
        "verdict than any single test window."
    ]
    worst_fold = scores.loc[scores[models].mean(axis=1).idxmax()]
    lines.append(
        f"Every model does worst in fold {int(worst_fold['fold'])} ({worst_fold['test_from']} to "
        f"{worst_fold['test_to']}) — *Reading the misses* on the Comparison tab shows why."
    )
    if "SARIMA" in models:
        close = int(((scores["SARIMA"] - scores["seasonal naive"]).abs() < 0.1).sum())
        lines.append(
            f"SARIMA is within 0.1 points of the free baseline in {close} of {len(scores)} folds: its "
            "seasonal difference *is* \"compare with the same week last year\", so on short history "
            "it mostly reproduces it."
        )
    return lines


def misses_reading(misses: pd.DataFrame) -> str:
    """The biggest misses in one paragraph, naming the calendar event behind each."""
    explained = misses[misses["why"] != ""]
    if explained.empty:
        return ("None of the biggest misses lines up with a holiday — they are mostly the level: "
                "this year runs a little above or below last year.")
    parts = []
    for _, row in explained.iterrows():
        direction = "too low" if row["miss %"] > 0 else "too high"
        parts.append(f"week {row['week']}: forecast {abs(row['miss %']):.0f}% {direction} — {row['why']}")
    return "; ".join(parts) + "."
