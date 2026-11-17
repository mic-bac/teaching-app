"""
7_Time_Series.py
----------------
Time Series lesson page — "can this series be forecast at all, and what do the
classical models make of it?"

Surfaces ``predictions/timeseries.py`` as an interactive lesson: a suitability
report before any model, decomposition, the trend that disappears once you
deflate, stationarity and autocorrelation, splitting by time and the four error
measures (with the lecture's own spreadsheet example), baselines, exponential
smoothing, Holt-Winters and SARIMA, rolling-origin cross-validation and the
Christmas problem.

All logic lives in ``utils/forecasting_utils`` (UI-free), which imports the
sibling's own ``predictions/src`` package — the code shown here is the code the
lesson script runs. Graceful behaviour: if the sibling is missing the page says
so; if the sales CSVs are missing, the sibling falls back to synthetic sales
with the same structure and the page works unchanged.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils import forecasting_utils as F
from utils.teaching import source_of

st.set_page_config(page_title="Time Series", layout="wide", page_icon="📈")

ROOT = Path(__file__).resolve().parents[1]
SIBLING = ROOT / "predictions"
BLUE, ORANGE, GREEN, GREY, RED = "#1e88e5", "#e67e22", "#2ecc71", "#95a5a6", "#e74c3c"


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


if not F.SIBLING_AVAILABLE:
    st.title("📈 Time Series")
    st.error(
        "This lesson reads its code and data from the sibling `predictions/` repo "
        f"(`{SIBLING}`), which is not importable — or older than this app expects.\n\n"
        f"Error: `{F.SIBLING_ERROR}`"
    )
    st.stop()


# ===========================================================================
# Cached computations — everything here is cheap except SARIMA
# ===========================================================================
@st.cache_data(show_spinner="Loading weekly sales…")
def cached_chain() -> dict:
    return F.load_chain()


@st.cache_data(show_spinner=False)
def cached_report(series: pd.Series) -> dict:
    return F.suitability_report(series)


@st.cache_data(show_spinner=False)
def cached_control() -> pd.Series:
    return F.load_control_series()


@st.cache_resource(show_spinner=False)
def cached_ets(train: pd.Series, horizon: int) -> tuple[object, pd.Series, float]:
    """Resource cache: the fitted statsmodels result is kept as is (not pickled)."""
    start = time.perf_counter()
    fitted, forecast = F.fit_ets(train, horizon)
    return fitted, forecast, time.perf_counter() - start


@st.cache_data(show_spinner=False)
def cached_sarima(train: pd.Series, horizon: int) -> tuple[pd.Series, float]:
    start = time.perf_counter()
    _, forecast = F.fit_sarima(train, horizon)
    return forecast, time.perf_counter() - start


@st.cache_data(show_spinner="Cross-validating three folds…")
def cached_cv(series: pd.Series, include_sarima: bool = False) -> dict:
    return F.cross_validate(series, include_sarima=include_sarima)


data = cached_chain()
sales, cpi = data["sales"], data["cpi"]
report = cached_report(sales)
train, test = F.split_series(sales, test_weeks=F.TEST_WEEKS)


# ===========================================================================
# Header
# ===========================================================================
st.title("📈 Time Series — can this series be forecast at all?")
with st.container(border=True):
    st.markdown(
        """
        💼 **Where you'll use this** — planning next quarter's revenue or headcount, ordering stock
        before the Christmas peak, setting realistic sales targets, or spotting when this week is
        *unusually* low. Every one of those starts with the questions on this page.
        """
    )
st.markdown(
    """
    **"After all our campaigns, offers and promotions, revenue has to grow. We need to know
    how it will change so that we can plan."**

    That question hides a harder one: *what can this data actually tell us?* This lesson
    answers it **before** fitting any model — and every model result later follows from
    the answer. Every tab shows the **real code** from the sibling `predictions/` repo.
    """
)
st.caption(
    f"**The data:** weekly sales of a retail chain, all stores summed into one number per week — "
    f"{len(sales)} weeks from {sales.index[0].date()} to {sales.index[-1].date()}."
)

tab_overview, tab_parts, tab_stat, tab_split, tab_models, tab_cv, tab_compare = st.tabs(
    [
        "🧭 Forecastable?",
        "🧩 Decomposition & Trend",
        "📉 Stationarity & ACF",
        "✂️ Split & Error Measures",
        "🤖 Baselines & Models",
        "🔁 Cross-Validation",
        "🏁 Comparison",
    ]
)

# ===========================================================================
# TAB 1 — Is this series forecastable?
# ===========================================================================
with tab_overview:
    st.subheader("The order of work")
    st.graphviz_chart(F.timeseries_flow_dot(), width="stretch")

    fig = go.Figure(go.Scatter(x=sales.index, y=sales.values, mode="lines", line=dict(color=BLUE)))
    fig.update_layout(height=320, title="<b>Weekly chain sales</b>", yaxis_title="weekly sales",
                      margin=dict(t=50, b=20))
    st.plotly_chart(fig, width="stretch")

    st.subheader("The suitability report — before choosing a model")
    st.markdown(
        "Five properties decide what is possible: a **regular grid**, enough **seasonal "
        "cycles**, a real **trend** (or not), how strong the **season** is, and **stationarity**. "
        "`cycles` is the one people skip — and here it binds hardest."
    )
    c = st.columns(5)
    c[0].metric("Regular grid", "yes" if report["regular_grid"] else "no", f"every {report['step_days']} days",
                delta_color="off", delta_arrow="off")
    c[1].metric("Seasonal cycles", f"{report['cycles']:.2f}", "52-week season", delta_color="off", delta_arrow="off")
    c[2].metric("Trend", f"{report['trend_pct_per_year']:+.2f}%/yr", f"p = {report['trend_p_value']:.2f}",
                delta_color="off", delta_arrow="off", help="A slope with p ≈ 0.7 is not a small trend — it is no trend.")
    c[3].metric("Seasonal strength", f"{report['seasonal_strength']:.2f}", "0–1 scale", delta_color="off", delta_arrow="off")
    c[4].metric("Stationary (ADF)", "yes" if report["stationary"] else "no",
                f"p = {report['adf_p_value']:.1e}", delta_color="off", delta_arrow="off")

    readings = F.report_readings(report)
    st.markdown(
        f"""
- **Regular grid** — {readings['grid']}
- **Seasonal cycles** — {readings['cycles']} A model that learns "week 47 is big" has very few examples of week 47.
- **Trend** — {readings['trend']}
- **Seasonal strength** — {readings['season']}
- **Stationary** — {readings['stationary']}
"""
    )
    st.caption(
        "Reading a **p-value**: roughly, *how easily could this pattern be pure chance?* Below 0.05 we "
        "treat it as real."
    )
    control = cached_control()
    control_report = cached_report(control)
    st.markdown("**For scale: the same report on a well-conditioned series** (weekly atmospheric CO₂ — "
                "not sales data, a measuring stick).")
    st.dataframe(
        pd.DataFrame(
            {
                "Retail chain sales": [f"{report['cycles']:.2f}", f"{report['trend_r_squared']:.3f}",
                                        f"{report['seasonal_strength']:.2f}"],
                "Weekly CO₂ since 1980": [f"{control_report['cycles']:.1f}",
                                          f"{control_report['trend_r_squared']:.3f}",
                                          f"{control_report['seasonal_strength']:.2f}"],
            },
            index=["seasonal cycles of history", "trend r²", "seasonal strength"],
        ),
        width="stretch",
    )
    st.info(
        f"""**Verdict**
- ✅ The time axis is clean: {report['n']} consecutive weeks, no gaps to repair.
- ✅ Seasonality is overwhelming ({report['seasonal_strength']:.2f}) and lands exactly at lag
  {report['strongest_lag']} — so *"the same week last year"* will be hard to beat.
- ⚠️ No trend worth modelling: {report['trend_pct_per_year']:+.2f}%/year at p = {report['trend_p_value']:.2f} is a
  line through noise; a model that extrapolates it will drift.
- ⚠️ {report['cycles']:.2f} cycles is thin: every one of 52 seasonal terms rests on two or three
  observations, and one whole class of evaluation becomes impossible (Cross-Validation tab)."""
    )
    show_source(F.suitability_report, "🐍 The suitability report")
    show_sibling_script("timeseries.py", "📄 The full sibling script: predictions/timeseries.py")

# ===========================================================================
# TAB 2 — Decomposition, and the trend that isn't
# ===========================================================================
with tab_parts:
    st.subheader("observed = trend + seasonality + residual")
    model = st.radio("Decomposition model", ["additive", "multiplicative"], horizontal=True,
                     help="Multiplicative: the seasonal swing grows with the level. This chain's level is "
                          "flat, so the two barely differ.")
    parts = F.decompose(sales, period=F.SEASONAL_PERIOD, model=model)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=("Observed", "Trend (52-week moving average)", "Seasonality", "Residual"))
    for row, (column, colour) in enumerate(
        [("observed", BLUE), ("trend", ORANGE), ("seasonal", GREEN), ("residual", GREY)], start=1
    ):
        fig.add_trace(go.Scatter(x=parts["Date"], y=parts[column], mode="lines", line=dict(color=colour),
                                 showlegend=False), row=row, col=1)
    fig.update_layout(height=640, margin=dict(t=40))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "The trend panel is empty at both ends: a 52-week centred moving average has no window for the "
        "first and last 26 weeks. Not a bug — the price of the method."
    )
    if model == "additive":
        seasonal_swing = parts["seasonal"].max() - parts["seasonal"].min()
        trend_swing = parts["trend"].max() - parts["trend"].min()
        c = st.columns(3)
        c[0].metric("Seasonal swing (peak–trough)", f"{seasonal_swing:,.0f}")
        c[1].metric("Trend swing over 2.7 years", f"{trend_swing:,.0f}")
        c[2].metric("Season vs trend", f"{seasonal_swing / trend_swing:.1f}×")
        peak = parts.loc[parts["seasonal"].idxmax()]
        trough = parts.loc[parts["seasonal"].idxmin()]
        st.markdown(
            f"**Reading:** the seasonal part says how far a week sits above or below normal. The biggest "
            f"week of the year (around {peak['Date'].strftime('%d %B')}) adds **{F.money(peak['seasonal'])}** "
            f"to a normal week; the quietest (around {trough['Date'].strftime('%d %B')}) is "
            f"**{F.money(trough['seasonal'])}**. The trend, by contrast, moved only {F.money(trend_swing)} "
            f"in total over 2.7 years — the season moves this business {seasonal_swing / trend_swing:.0f}× "
            "more than the trend does."
        )
    else:
        st.caption("Multiplicative: the seasonal part is a factor (1.25 = 25% above a normal week) "
                   "instead of an amount in euros. Switch to additive to see it in money.")
    show_source(F.decompose, "🐍 Decomposition")

    st.divider()
    st.subheader("The trend that isn't — and where it went")
    st.markdown(
        "No trend is a strange result for a retail chain over three years. Two pieces of domain "
        "knowledge explain it, and both are testable with columns already in this dataset."
    )
    real = F.deflate(sales, cpi)
    nominal_trend, real_trend = F.trend_test(sales), F.trend_test(real)
    col_a, col_b = st.columns([3, 2])
    with col_a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sales.index, y=sales.values, mode="lines", name="nominal (euros)",
                                 line=dict(color=BLUE)))
        fig.add_trace(go.Scatter(x=real.index, y=real.values, mode="lines", name="real (goods, CPI-deflated)",
                                 line=dict(color=ORANGE)))
        fig.update_layout(height=360, title="<b>The same sales, in euros and in goods</b>",
                          legend=dict(orientation="h", y=-0.2), margin=dict(t=50))
        st.plotly_chart(fig, width="stretch")
    with col_b:
        st.markdown("**1 · Money shrinks.** Deflating by CPI asks whether the chain sold more *goods*.")
        st.dataframe(
            pd.DataFrame(
                {"trend %/year": [nominal_trend["pct_per_year"], real_trend["pct_per_year"]],
                 "p-value": [nominal_trend["p_value"], real_trend["p_value"]]},
                index=["nominal", "real"],
            ).style.format({"trend %/year": "{:+.2f}", "p-value": "{:.2f}"}),
            width="stretch",
        )
        st.caption(
            f"CPI {cpi.iloc[0]:.1f} → {cpi.iloc[-1]:.1f} ({100 * (cpi.iloc[-1] / cpi.iloc[0] - 1):+.1f}%). "
            "The point estimate flips sign — flat in euros, shrinking in goods — but neither is "
            "significant. The honest sentence: *nominal stability may be masking a real decline, and "
            "this data cannot settle it.*"
        )
        mb = data["markdown"]
        st.markdown(
            f"**2 · Promotions.** `{', '.join(mb['columns'])}` are recorded only from "
            f"**{mb['first_recorded'].date()}** — {mb['weeks_without']} of {mb['n_weeks']} weeks "
            f"({mb['weeks_without'] / mb['n_weeks']:.0%}) have none."
        )
        st.caption(
            "A **structural break**, not missing data: promotion tracking did not exist before. "
            "Filling zeros would tell every model *\"no campaigns for two years\"* — false. Model the later "
            "period only, or make *\"was promotion data recorded\"* a feature."
        )
    show_source(F.trend_test, "🐍 Is the trend real? (slope, p-value, r²)")

# ===========================================================================
# TAB 3 — Stationarity and autocorrelation
# ===========================================================================
with tab_stat:
    st.subheader("Stationarity — does the series keep coming back to the same level?")
    st.markdown(
        "A **stationary** series wanders around a stable level instead of drifting away. Why it "
        "matters: some models (the *I* in ARIMA) first remove a drift — pointless work if there is none. "
        "The **ADF test** checks it for us: a **p-value below 0.05 means *stationary***."
    )
    adf_level = F.adf_report(sales)
    adf_diff = F.adf_report(sales.diff().dropna())
    st.dataframe(
        pd.DataFrame(
            [{"series": "level", "ADF statistic": adf_level["statistic"], "p-value": adf_level["p_value"],
              "stationary": adf_level["stationary"]},
             {"series": "first difference", "ADF statistic": adf_diff["statistic"],
              "p-value": adf_diff["p_value"], "stationary": adf_diff["stationary"]}]
        ).style.format({"ADF statistic": "{:.2f}", "p-value": "{:.1e}"}),
        width="stretch", hide_index=True,
    )
    st.caption(
        "Our sales are already stationary — no surprise, since there is no trend. So the SARIMA model "
        "later skips the drift-removal step. Tip: a strong season can fool this test, so always look at "
        "the chart as well."
    )
    show_source(F.adf_report, "🐍 The ADF test")

    st.divider()
    st.subheader("Autocorrelation — how much does this week resemble earlier weeks?")
    st.markdown(
        "**ACF** asks: *how similar is this week to the week k weeks ago?* **PACF** asks the same, but "
        "counts only what that one week adds on its own. Bars that stick out beyond the dashed lines are "
        "real patterns; everything inside them is noise."
    )
    corr = F.autocorrelation(sales, nlags=60)
    band = 1.96 / np.sqrt(len(sales))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("ACF — autocorrelation", "PACF — partial autocorrelation"))
    for row, column in enumerate(["acf", "pacf"], start=1):
        colours = [RED if lag == F.SEASONAL_PERIOD else BLUE for lag in corr["lag"]]
        fig.add_trace(go.Bar(x=corr["lag"], y=corr[column], marker_color=colours, showlegend=False), row=row, col=1)
        for sign in (1, -1):
            fig.add_hline(y=sign * band, line_dash="dash", line_color=GREY, row=row, col=1)
    fig.update_xaxes(title_text="lag (weeks)", row=2, col=1)
    fig.update_layout(height=520, margin=dict(t=40))
    st.plotly_chart(fig, width="stretch")
    lag52 = corr.loc[corr.lag == 52, "acf"].iloc[0]
    lag1 = corr.loc[corr.lag == 1, "acf"].iloc[0]
    st.info(
        f"**Lag 52 ({lag52:+.2f}) beats lag 1 ({lag1:+.2f}).** Last year's same week tells you more about "
        "this week than last week does — which is why the seasonal-naive baseline is so hard to beat, and "
        "why every model here is built around a 52-week season. (Dashed: 95% significance band.)"
    )
    show_source(F.autocorrelation, "🐍 ACF and PACF")

# ===========================================================================
# TAB 4 — Splitting, and the four error measures
# ===========================================================================
with tab_split:
    st.subheader("Split by time — never shuffle")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train.values, mode="lines", name=f"train ({len(train)} weeks)",
                             line=dict(color=BLUE)))
    fig.add_trace(go.Scatter(x=test.index, y=test.values, mode="lines", name=f"test ({len(test)} weeks)",
                             line=dict(color=ORANGE, width=3)))
    fig.update_layout(height=300, margin=dict(t=20), legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "The last quarter is held out and never seen. Shuffling would let the model learn from March to "
        "predict February — a score for interpolation on a task that is extrapolation."
    )
    show_source(F.split_series, "🐍 Splitting a time series")

    st.divider()
    st.subheader("The lecture's spreadsheet, computed by the real code")
    st.markdown(
        "Slides 70 and 78 compute exponential smoothing and the four error measures in Excel. Here is "
        "the same table, produced by the sibling's `simple_exponential_smoothing` and `forecast_metrics`. "
        "At **α = 0.4** it reproduces the slide to the cent."
    )
    alpha = st.slider("α (smoothing weight on the latest actual)", 0.05, 0.95, 0.40, 0.05)
    table, slide_metrics = F.slide_smoothing_table(alpha)
    col_t, col_m = st.columns([3, 2])
    with col_t:
        shown = table.copy()
        for column in ["pred", "abs_error", "abs_pct_error", "squared_error"]:
            shown[column] = shown[column].map(lambda v: "—" if pd.isna(v) else f"{v:.2f}")
        st.dataframe(shown, width="stretch", hide_index=True)
        st.caption("Row 1 has no forecast to score — its value only starts the recursion.")
    with col_m:
        m = st.columns(2)
        m[0].metric("MAE", f"{slide_metrics['MAE']:.2f}")
        m[1].metric("MAPE", f"{slide_metrics['MAPE']:.1f}%", help="The slide writes it as a fraction: 0.39.")
        m = st.columns(2)
        m[0].metric("MSE", f"{slide_metrics['MSE']:.2f}")
        m[1].metric("RMSE", f"{slide_metrics['RMSE']:.2f}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=table["t"], y=table["y"], mode="lines+markers", name="y", line=dict(color=BLUE)))
        fig.add_trace(go.Scatter(x=table["t"], y=table["pred"], mode="lines+markers", name="pred",
                                 line=dict(color=ORANGE, dash="dash")))
        fig.update_layout(height=260, margin=dict(t=10, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")
    st.caption(
        "Drag α: high values chase the latest point, low values smooth through noise — and the forecast "
        "always lags a trend, because simple smoothing has no trend term."
    )
    show_source(F.simple_exponential_smoothing, "🐍 Exponential smoothing, written out")

    st.divider()
    st.subheader("Four ways to average the same errors — they disagree on purpose")
    example = F.forecast_metrics(test, F.seasonal_naive_forecast(train, len(test)))
    c = st.columns(5)
    c[0].metric("MAE", F.money(example['MAE']), help="average miss, in euros")
    c[1].metric("MSE", F.big_number(example['MSE']), help="squared euros — no interpretable unit")
    c[2].metric("RMSE", F.money(example['RMSE']), help="back in euros, big misses count more")
    c[3].metric("MAPE", f"{example['MAPE']:.2f}%", help="average miss as a percentage")
    c[4].metric("R²", f"{example['R2']:.2f}", help="share of variance explained")
    st.markdown(
        """
| Measure | Use it when… |
|---|---|
| **MAE** | the cost of being wrong is proportional to how wrong you are |
| **RMSE** | one large miss is what really hurts (a stock-out at Christmas) |
| **MAPE** | series of different sizes must be comparable — but it divides by the actual value, so it breaks on series near zero (Forecasting page) |
"""
    )
    st.markdown(
        f"""
**Reading** (the seasonal-naive forecast of the held-out quarter, a typical week ≈ {F.money(test.mean())}):
- **MAE {F.money(example['MAE'])}** — on average the forecast misses a week by that much.
- **RMSE {F.money(example['RMSE'])}** — larger than the MAE, because squaring gives the few big misses
  more weight. The bigger the gap between the two, the more the error comes from a handful of bad weeks.
- **MAPE {example['MAPE']:.2f}%** — the same miss as a share of the week: {F.mape_in_money(example['MAPE'], test)}.
- **R² {example['R2']:.2f}** — the forecast explains {max(example['R2'], 0):.0%} of the up-and-down *within* this
  quarter. Low-looking, because a quiet quarter has little up-and-down to explain; judge forecasts by MAE/MAPE.
"""
    )
    show_source(F.forecast_metrics, "🐍 MAE, MSE, RMSE, MAPE, R²")

# ===========================================================================
# TAB 5 — Baselines and models
# ===========================================================================
with tab_models:
    st.subheader("Baselines — the bar every model has to clear")
    baselines = F.baseline_forecasts(train, len(test))
    baseline_scores = {name: F.forecast_metrics(test, p)["MAPE"] for name, p in baselines.items()}
    st.dataframe(
        pd.DataFrame([{"baseline": name, "MAPE": mape} for name, mape in baseline_scores.items()])
        .style.format({"MAPE": "{:.2f}%"}),
        width="stretch", hide_index=True,
    )
    st.caption(
        f"Same data, no fitting. *Naive* repeats last week ({baseline_scores['naive (last value)']:.2f}%); "
        f"*seasonal naive* repeats the same week last year ({baseline_scores['seasonal naive (t-52)']:.2f}%). "
        "The gap between them is the value of knowing about seasonality — and nothing else. A model "
        "reported without this comparison is unevaluated."
    )
    show_source(F.seasonal_naive_forecast, "🐍 The seasonal-naive baseline")

    growth = F.growth_example(train)
    with st.container(border=True):
        st.markdown(
            f"""
**One number makes the baseline better: growth.** "Same week last year" also copies last year's
*level*. If this year runs higher, every copied week comes out too low. So:

1. The last {growth['window']} weeks sold **{F.money(growth['recent'])}**; the same weeks a year earlier
   **{F.money(growth['year_before'])}** → growth = **{growth['growth']:.3f}** ({100 * (growth['growth'] - 1):+.1f}%).
2. First forecast week: the same week last year ({growth['copied_week'].date()}) sold {F.money(growth['copied'])}
   → × {growth['growth']:.3f} = **{F.money(growth['forecast'])}**.

Result: {baseline_scores['seasonal naive (t-52)']:.2f}% → **{baseline_scores[F.GROWTH_LABEL]:.2f}%** MAPE,
with still nothing fitted. Keep this idea in mind — it is what Holt-Winters' *level* does automatically.
"""
        )
    show_source(F.seasonal_naive_growth_forecast, "🐍 Same week last year × growth")

    st.divider()
    st.subheader("Exponential smoothing — and why α barely matters here")
    sweep = F.smoothing_sweep(sales)
    st.dataframe(sweep.style.format({"alpha": "{:.1f}", "MAE": F.money, "MAPE": "{:.2f}%"}),
                 width="stretch", hide_index=True)
    st.caption(
        "Three very different α, barely a tenth of a percent apart. That *is* the result: this series is "
        "made of seasonality and simple smoothing has no seasonal term to put it in. That is exactly why "
        "Holt added a trend equation and Winters a seasonal one."
    )

    st.divider()
    st.subheader("Holt-Winters — and what it learned")
    ets_model, ets_forecast, ets_seconds = cached_ets(train, len(test))
    ets_mape = F.forecast_metrics(test, ets_forecast.values)["MAPE"]
    hw = F.holt_winters_parts(ets_model)
    st.markdown(
        f"**Holt-Winters (ETS)** keeps three running estimates and updates each one every week: "
        f"MAPE **{ets_mape:.2f}%**, fitted in {ets_seconds:.2f}s — {F.mape_in_money(ets_mape, test)}."
    )
    parts_frame = hw["parts"]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Level — where sales are now", "Trend — change per week",
                                        "Season — this week vs a normal week"))
    for row, (column, colour) in enumerate([("level", BLUE), ("trend", ORANGE), ("season", GREEN)], start=1):
        fig.add_trace(go.Scatter(x=parts_frame.index, y=parts_frame[column], mode="lines",
                                 line=dict(color=colour), showlegend=False), row=row, col=1)
    fig.update_layout(height=560, margin=dict(t=40))
    st.plotly_chart(fig, width="stretch")
    c = st.columns(3)
    c[0].metric("α — level weight", f"{hw['alpha']:.2f}", f"level now {F.money(hw['last_level'])}",
                delta_color="off", delta_arrow="off")
    c[1].metric("β — trend weight", f"{hw['beta']:.2f}",
                f"{F.money(hw['trend_per_week'])}/week = {hw['trend_pct_per_year']:+.1f}%/year",
                delta_color="off", delta_arrow="off")
    c[2].metric("γ — season weight", f"{hw['gamma']:.2f}", "how fast the season adapts",
                delta_color="off", delta_arrow="off")
    st.markdown(
        f"""
**Reading.** Each weight says how strongly the newest week changes the estimate: 1 = forget the past,
close to 0 = change your mind slowly. All three are small — right for a series this regular.
- **Level** ≈ {F.money(hw['last_level'])} a week: this is how the model lifts last year's season to this
  year's sales — the growth factor above, done continuously.
- **Trend** {F.money(hw['trend_per_week'])} a week ({hw['trend_pct_per_year']:+.1f}% a year): small, in line with
  the suitability report's *no clear trend* ({report['trend_pct_per_year']:+.2f}%/year, p = {report['trend_p_value']:.2f}).
- **Season**: the same shape as the decomposition — the Christmas peak, the January dip.
"""
    )
    with st.expander("🔎 The maths, if you're curious"):
        st.markdown(
            r"""
Three smoothing equations, each the same idea as simple exponential smoothing
(new estimate = weight × what just happened + (1 − weight) × what we believed):

$$\ell_t = \alpha\,(y_t - s_{t-52}) + (1-\alpha)(\ell_{t-1} + b_{t-1})$$
$$b_t = \beta\,(\ell_t - \ell_{t-1}) + (1-\beta)\,b_{t-1}$$
$$s_t = \gamma\,(y_t - \ell_t) + (1-\gamma)\,s_{t-52}$$

Forecast $h$ weeks ahead: $\hat y_{t+h} = \ell_t + h\,b_t + s_{t+h-52}$.
"""
        )
    show_source(F.fit_ets, "🐍 Holt-Winters")
    show_source(F.holt_winters_parts, "🐍 What Holt-Winters learned")

    st.divider()
    st.subheader("SARIMA")
    st.markdown(
        f"**{F.sarima_label()}** combines three ideas: learn from the *last few weeks*, learn from "
        "*its own recent mistakes*, and compare with *the same week last year*. The numbers in brackets are "
        "its settings; the seasonal part (0,1,1,52) means \"work with the change against the same week last "
        "year\" — which is very close to what the seasonal-naive baseline does. Looking back 52 weeks makes "
        "it slow — part of the lesson."
    )
    if st.button("🧮 Fit SARIMA (takes ~15 s)") or "sarima" in st.session_state:
        with st.spinner("Fitting SARIMA with 52 seasonal lags…"):
            st.session_state["sarima"] = cached_sarima(train, len(test))
        sarima_forecast, sarima_seconds = st.session_state["sarima"]
        sarima_mape = F.forecast_metrics(test, sarima_forecast.values)["MAPE"]
        c = st.columns(3)
        c[0].metric("SARIMA MAPE", f"{sarima_mape:.2f}%")
        c[1].metric("Fit time", f"{sarima_seconds:.1f}s")
        c[2].metric("vs Holt-Winters", f"{sarima_seconds / max(ets_seconds, 1e-9):.0f}× the compute")
        st.caption(
            f"Reading: SARIMA ({sarima_mape:.2f}%) lands between the baseline "
            f"({baseline_scores['seasonal naive (t-52)']:.2f}%) and Holt-Winters ({ets_mape:.2f}%) on this "
            "window, for far more compute. One window is one sample — the Cross-Validation tab checks "
            "whether that holds."
        )
    else:
        st.info("Click to fit SARIMA. The comparison and cross-validation tabs can include it once fitted.")
    show_source(F.fit_sarima, "🐍 SARIMA")

# ===========================================================================
# TAB 6 — Cross-validation and the Christmas problem
# ===========================================================================
with tab_cv:
    st.subheader("Rolling-origin cross-validation")
    st.markdown(
        "One split is one sample — it can flatter a model by landing on an easy stretch. For a time "
        "series the folds must respect time: train on everything up to a point, test on what follows, "
        "move the point forward. **Random k-fold would train on next Christmas to predict last October.**"
    )
    folds = F.rolling_origin_splits(sales, n_folds=3, horizon=F.TEST_WEEKS)
    fig = go.Figure()
    DAY_MS = 86_400_000  # a bar on a date axis measures its length in milliseconds
    for i, (fold_train, fold_test) in enumerate(folds):
        label = f"fold {i + 1}"
        fig.add_trace(go.Bar(y=[label], x=[(fold_train.index[-1] - fold_train.index[0]).days * DAY_MS],
                             base=[fold_train.index[0]], orientation="h", marker_color=BLUE,
                             name="train", showlegend=i == 0))
        fig.add_trace(go.Bar(y=[label], x=[((fold_test.index[-1] - fold_test.index[0]).days + 7) * DAY_MS],
                             base=[fold_test.index[0]], orientation="h", marker_color=ORANGE,
                             name="test", showlegend=i == 0))
    fig.update_layout(height=220, barmode="overlay", xaxis_type="date", margin=dict(t=10),
                      yaxis=dict(autorange="reversed"), legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")

    with_sarima = st.toggle("Include SARIMA (adds ~20 s, three more fits)", value=False)
    cv = cached_cv(sales, include_sarima=with_sarima)
    scores, cv_models = cv["scores"], cv["models"]
    st.dataframe(
        scores.style.format({name: "{:.2f}%" for name in cv_models})
        .highlight_min(subset=cv_models, axis=1, color="#c8e6c9"),
        width="stretch", hide_index=True,
    )
    st.caption("Each row is one fold: train on everything before the window, forecast its 13 weeks. "
               "Green = the best model in that fold.")
    long = scores.melt(id_vars=["fold"], value_vars=cv_models, var_name="model", value_name="MAPE")
    long["fold"] = "fold " + long["fold"].astype(str)
    fig = px.bar(long, x="fold", y="MAPE", color="model", barmode="group", text_auto=".2f",
                 color_discrete_map={"seasonal naive": ORANGE, "naive × growth": "#f4b183",
                                     "Holt-Winters": GREEN, "SARIMA": RED},
                 labels={"MAPE": "MAPE (%)", "fold": ""})
    fig.update_layout(height=340, margin=dict(t=10), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    for line in F.cv_reading(scores, cv_models):
        st.markdown(f"- {line}")
    c = st.columns(len(cv_models))
    for column, name in zip(c, cv_models):
        column.metric(name, f"mean {scores[name].mean():.2f}%",
                      f"spread {scores[name].min():.2f}–{scores[name].max():.2f}%",
                      delta_color="off", delta_arrow="off")
    show_source(F.rolling_origin_splits, "🐍 Rolling-origin splits")

    st.divider()
    st.subheader("🎄 The Christmas problem")
    xmas = F.christmas_check(sales)
    st.markdown(
        f"""
Every fold above tests a **quiet** stretch — none contains Christmas, and that is forced, not chosen:

- Christmas weeks in this data: **{', '.join(str(d.date()) for d in xmas['christmas_weeks'])}**
- A 52-week seasonal model needs **two full cycles** (104 weeks) before it can be fitted at all —
  the earliest testable week is **{xmas['earliest_testable'].date()}**.

By the time enough history exists to fit, every Christmas is already inside it. **The hardest weeks of
the retail year cannot be evaluated with this dataset.**
"""
    )
    if xmas["christmas_mape"] is not None:
        c = st.columns(3)
        c[0].metric("Seasonal naive — quiet window", f"{xmas['quiet_mape']:.2f}%",
                    f"{xmas['quiet_window'][0].date()} → {xmas['quiet_window'][1].date()}", delta_color="off", delta_arrow="off")
        c[1].metric("Seasonal naive — Christmas window", f"{xmas['christmas_mape']:.2f}%",
                    f"{xmas['christmas_window'][0].date()} → {xmas['christmas_window'][1].date()}",
                    delta_color="off", delta_arrow="off")
        c[2].metric("Christmas vs quiet", f"{xmas['christmas_mape'] / xmas['quiet_mape']:.1f}× worse")
        st.warning(
            "The baseline needs only 52 weeks, so *it* can be scored on Christmas — and does far worse there. "
            "Every headline number in this lesson comes from the quiet window and is flattered by it. "
            "**Say so when you present one.**"
        )
    show_source(F.christmas_check, "🐍 The Christmas check")

# ===========================================================================
# TAB 7 — Comparison
# ===========================================================================
with tab_compare:
    st.subheader("Which forecast to use?")
    results = dict(F.baseline_forecasts(train, len(test)))
    results["Holt-Winters (ETS)"] = ets_forecast.values
    if "sarima" in st.session_state:
        results[F.sarima_label()] = st.session_state["sarima"][0].values
    table = F.metrics_table({name: F.forecast_metrics(test, p) for name, p in results.items()})

    cv_means = scores[cv_models].mean()
    cv_winner = cv_means.idxmin()
    st.success(
        f"**Recommendation: {cv_winner}.** It has the lowest error across all three cross-validation folds "
        f"(mean {cv_means[cv_winner]:.2f}% vs {cv_means['seasonal naive']:.2f}% for *same week last year*) — "
        f"{100 * (1 - cv_means[cv_winner] / cv_means['seasonal naive']):.0f}% less error than a forecast that "
        "costs nothing. The folds matter more than the single window below: one window can be lucky."
    )

    band = F.cv_interval(cv["errors"]["Holt-Winters"], coverage=0.8)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index[-26:], y=train.values[-26:], mode="lines", name="history (last 26 weeks)",
                             line=dict(color=GREY)))
    hw_values = results["Holt-Winters (ETS)"]
    fig.add_trace(go.Scatter(
        x=list(test.index) + list(test.index[::-1]),
        y=list(hw_values * (1 + band / 100)) + list((hw_values * (1 - band / 100))[::-1]),
        fill="toself", fillcolor="rgba(46, 204, 113, 0.18)", line=dict(width=0), hoverinfo="skip",
        name=f"Holt-Winters ±{band:.1f}% (80% of past weeks)"))
    fig.add_trace(go.Scatter(x=test.index, y=test.values, mode="lines+markers", name="actual",
                             line=dict(color="black", width=3)))
    palette = {"seasonal naive (t-52)": ORANGE, F.GROWTH_LABEL: "#f4b183", "Holt-Winters (ETS)": GREEN,
               F.sarima_label(): RED, "naive (last value)": "#9b59b6", "mean of history": "#7f8c8d"}
    for name, prediction in results.items():
        fig.add_trace(go.Scatter(x=test.index, y=prediction, mode="lines", name=name,
                                 line=dict(color=palette.get(name, BLUE), dash="dash"),
                                 visible=True if name in ("seasonal naive (t-52)", "Holt-Winters (ETS)")
                                 else "legendonly"))
    fig.update_layout(height=440, title="<b>Forecasts against the held-out quarter</b> (click the legend to toggle)",
                      yaxis_title="weekly sales")
    st.plotly_chart(fig, width="stretch")
    inside = float(np.mean(np.abs(100 * (test.values - hw_values) / test.values) <= band))
    st.markdown(
        f"**How far off to expect.** In the cross-validation folds, Holt-Winters forecast "
        f"{len(cv['errors']['Holt-Winters'])} weeks; 80% of them — four weeks in five — missed by less "
        f"than **±{band:.1f}%** — about ±{F.money(band / 100 * test.mean())} on a typical week. That is the "
        f"green band; in this quarter {inside:.0%} of the actual weeks fall inside it. It is a planning range, "
        "not a guarantee — and it comes from quiet quarters only, so do not stretch it over Christmas."
    )
    show_source(F.cv_interval, "🐍 The band, read off past errors")

    st.markdown("**All models on the single held-out quarter**")
    st.dataframe(
        table.style.format({"MAE": F.money, "MSE": F.big_number, "RMSE": F.money, "MAPE": "{:.2f}%", "R2": "{:.2f}"}),
        width="stretch", hide_index=True,
    )
    if "sarima" not in st.session_state:
        st.caption("SARIMA is not included yet — fit it on the *Baselines & Models* tab.")

    st.divider()
    st.subheader("Reading the misses")
    st.markdown(
        "A score says *how much* a forecast missed. Explaining it means saying *where* and *why*. Seasonal "
        "forecasts lean on the same week a year earlier — so put the calendar of both years next to every miss. "
        "Below: the hardest fold from cross-validation."
    )
    events = F.calendar_events(sales.index, F.WALMART_HOLIDAYS)
    hardest_row = int(scores[cv_models].mean(axis=1).to_numpy().argmax())  # position, not the fold's label
    hardest = int(scores["fold"].iloc[hardest_row])
    fold = cv["forecasts"][hardest_row]
    miss_model = st.radio("Model", ["seasonal naive", "Holt-Winters"], horizontal=True, key="miss_model")
    misses = F.explain_misses(fold["test"], fold["predictions"][miss_model], events)
    misses["label"] = misses["event this year"].where(
        misses["event this year"] != "",
        misses["same week last year"].str.extract(r"\((.*)\)")[0].map(
            lambda e: f"last year: {e}" if isinstance(e, str) else ""),
    )
    fig = go.Figure(go.Bar(
        x=[str(w) for w in misses["week"]], y=misses["miss %"], text=misses["label"], textposition="outside",
        marker_color=[GREEN if m >= 0 else RED for m in misses["miss %"]],
    ))
    fig.add_hline(y=0, line_color=GREY)
    fig.update_layout(height=360, yaxis_title="miss % (+ = forecast too low)", margin=dict(t=20),
                      title=f"<b>{miss_model}, fold {hardest}</b> — weekly misses with the calendar")
    st.plotly_chart(fig, width="stretch")
    top = F.explain_misses(fold["test"], fold["predictions"][miss_model], events, top=3)
    st.markdown(f"**Reading:** {F.misses_reading(top)}")
    easter = F.easter_weeks(sales.index)
    st.info(
        f"**Easter moves.** The week before Easter was {', '.join(str(d.date()) for d in easter)}. A forecast "
        "built on *the same week last year* copies an ordinary week into this year's Easter (too low) and last "
        "year's Easter peak into an ordinary week (too high). Holt-Winters makes the same two mistakes — it has "
        "learned *week 14 of the year*, not *the week before Easter*. The fix is a holiday feature that knows "
        "where Easter falls. The chain's official holiday list does not include it — the **Forecasting** page "
        "adds it to Prophet and measures what that is worth."
    )
    st.caption(
        "The smaller misses are mostly the level: this year runs a little above last year, so copies of last "
        "year come out slightly low — what the growth factor and Holt-Winters' level correct."
    )
    with st.expander("All weeks of this fold, with the calendar"):
        st.dataframe(
            misses.drop(columns="label").style.format(
                {"actual": "{:,.0f}", "forecast": "{:,.0f}", "miss %": "{:+.1f}%"}),
            width="stretch", hide_index=True,
        )
    show_source(F.explain_misses, "🐍 Explaining the misses")

    st.divider()
    st.subheader("What the slides say, and what this data shows")
    if "SARIMA" in cv_models:
        arima_row = (
            f"**Holt-Winters wins; SARIMA ≈ the free baseline** — across the folds Holt-Winters averaged "
            f"{cv_means['Holt-Winters']:.2f}%, SARIMA {cv_means['SARIMA']:.2f}%, *same week last year* "
            f"{cv_means['seasonal naive']:.2f}%. SARIMA's seasonal difference mostly reproduces the baseline, "
            "for far more compute."
        )
    else:
        arima_row = (
            f"**Holt-Winters wins all folds** ({cv_means['Holt-Winters']:.2f}% vs {cv_means['seasonal naive']:.2f}% "
            "for the baseline). Switch on SARIMA in the Cross-Validation tab to see how close it stays to the "
            "baseline, for far more compute."
        )
    st.markdown(
        f"""
| Slide statement | On this data |
|---|---|
| Decomposition into trend, seasonality and residuals | **Holds, with an asterisk** — the split works, but here the "trend" part is just noise. Three parts ≠ three *useful* parts. |
| Stationarity: constant mean and variance | **Holds** — simply because these sales never had a trend to begin with. |
| Exponential smoothing, ARIMA as classical models | {arima_row} |
| MAE, RMSE, MAPE are indispensable | **Holds** — and picking one is a business decision. MAPE can break on small numbers (see the Forecasting page). |
| Cross-validation for time series: rolling origin, never random k-folds | **Holds** — but you can only test weeks you have enough history for, which is why Christmas stays untested. |

"""
    )
    show_sibling_script("timeseries.py", "📄 The full sibling script: predictions/timeseries.py")

st.divider()
with st.container(border=True):
    st.markdown(
        """
🎯 **What to remember**
- **Check the data before the model.** Two numbers — how many seasons of history, and whether there
  is a real trend — told us most of what the models would later find.
- **Always compare against "same week last year" — and judge on several folds.** It costs nothing;
  Holt-Winters beats it here by learning the current *level*, and wins every fold.
- **Read the misses against the calendar.** The biggest errors had a name — Easter moved — and
  a forecast you can explain is one a planner can trust. The band says how far off to expect, but
  not at Christmas, which this data cannot test.
"""
    )
