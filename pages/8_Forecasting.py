"""
8_Forecasting.py
----------------
Forecasting lesson page — "Prophet and XGBoost: are they worth it here?"

Surfaces ``predictions/forecasting.py``: Prophet's explainable components and the
changepoint knob that causes overshooting, XGBoost's need for rows (one series vs
45 stores), early stopping and tree depth, external features, the department grain
where MAPE breaks, and detrending — a fix that helps or hurts depending on the
diagnosis.

All logic lives in ``utils/forecasting_utils`` (UI-free), which imports the
sibling's ``predictions/src`` package — the code shown is the code the lesson runs.
Written for business students: plain language first, maths in collapsed boxes.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils import forecasting_utils as F
from utils.teaching import source_of

st.set_page_config(page_title="Forecasting", layout="wide", page_icon="🔮")

ROOT = Path(__file__).resolve().parents[1]
SIBLING = ROOT / "predictions"
BLUE, ORANGE, GREEN, GREY, RED, PURPLE = "#1e88e5", "#e67e22", "#2ecc71", "#95a5a6", "#e74c3c", "#8e44ad"


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
    st.title("🔮 Forecasting")
    st.error(
        "This lesson reads its code and data from the sibling `predictions/` repo "
        f"(`{SIBLING}`), which is not importable — or older than this app expects.\n\n"
        f"Error: `{F.SIBLING_ERROR}`"
    )
    st.stop()


# ===========================================================================
# Cached computations
# ===========================================================================
@st.cache_data(show_spinner="Loading weekly sales at three levels…")
def cached_grains() -> dict:
    return F.load_grains()


@st.cache_data(show_spinner="Fitting Prophet…")
def cached_prophet(train: pd.Series, horizon: int, scale: float) -> dict:
    return F.prophet_run(train, horizon, changepoint_prior_scale=scale)


@st.cache_data(show_spinner="Three holiday calendars × three folds…")
def cached_calendar_cv(series: pd.Series) -> dict:
    return F.prophet_calendar_cv(series)


@st.cache_data(show_spinner="Fitting XGBoost and a random forest on one series…")
def cached_chain_trees(chain: pd.DataFrame, train_end: pd.Timestamp) -> dict:
    return F.chain_tree_models(chain, train_end)


@st.cache_data(show_spinner="Training one XGBoost for all 45 stores…")
def cached_panel(stores: pd.DataFrame) -> dict:
    return F.store_panel_run(stores)


@st.cache_data(show_spinner="Fitting one Holt-Winters model per store…")
def cached_ets_per_store(stores: pd.DataFrame) -> dict:
    return F.ets_per_store(stores)


@st.cache_data(show_spinner="Training four tree depths…")
def cached_depths(stores: pd.DataFrame) -> pd.DataFrame:
    return F.depth_sweep(cached_panel(stores))


@st.cache_data(show_spinner="Three folds, with and without external data…")
def cached_external(stores: pd.DataFrame) -> pd.DataFrame:
    return F.external_features_cv(cached_panel(stores)["panel"])


@st.cache_data(show_spinner="One model for ~3,300 store-department series…")
def cached_departments(depts: pd.DataFrame) -> dict:
    return F.department_grain(depts)


@st.cache_data(show_spinner="Four set-ups × three folds…")
def cached_detrending(stores: pd.DataFrame) -> pd.DataFrame:
    return F.detrending_experiment(cached_panel(stores)["panel"])


@st.cache_data(show_spinner="Same test on a series with a real trend…")
def cached_control() -> dict:
    return F.control_detrending()


grains = cached_grains()
chain, stores, depts, source = grains["chain"], grains["stores"], grains["depts"], grains["source"]
sales = F.as_series(chain)
train, test = F.split_series(sales, test_weeks=F.TEST_WEEKS)
horizon = len(test)
baseline = F.seasonal_naive_forecast(train, horizon)
baseline_mape = F.forecast_metrics(test, baseline)["MAPE"]


# ===========================================================================
# Header
# ===========================================================================
st.title("🔮 Forecasting — are Prophet and XGBoost worth it here?")
st.markdown(
    """
    **"Prophet and XGBoost are supposed to be state of the art. Should we use them for our sales forecast?"**

    The honest answer is: *it depends* — and this lesson shows **on what**. We fit both models to the
    same Walmart sales as the Time Series lesson, then to the same sales split by store and by
    department, and watch the winner change. Every tab shows the **real code** from the sibling repo.
    """
)
with st.container(border=True):
    st.markdown(
        """
        💼 **Where you'll use this** — sales and demand planning per store or product, staffing for busy
        weeks, budget forecasts you can *explain* to management (Prophet), or one model that forecasts
        thousands of products at once (XGBoost). And the judgement to know when a simple forecast is enough.
        """
    )
st.caption(f"**The data:** weekly sales of a retail chain — chain total, {stores['Store'].nunique()} stores "
           f"and their departments ({sales.index[0].date()} to {sales.index[-1].date()}).")
st.caption(f"The bar to beat on the chain total: *same week last year* (seasonal naive) — MAPE "
           f"**{baseline_mape:.2f}%** on the 13 held-out weeks.")

tab_prophet, tab_rows, tab_fit, tab_external, tab_grain, tab_trend, tab_verdict = st.tabs(
    [
        "🔮 Prophet",
        "🌳 XGBoost needs rows",
        "🎚️ Overfitting",
        "🌦️ External data",
        "🔬 Grain & metrics",
        "📐 Trends trees can't see",
        "🏁 Verdict",
    ]
)

# ===========================================================================
# TAB 1 — Prophet
# ===========================================================================
with tab_prophet:
    st.subheader("A forecast you can explain")
    st.markdown(
        "Prophet builds the forecast from parts you can show to anyone: **trend + yearly season + "
        "holiday effects**. So *\"why is November up?\"* gets a real answer, not *\"the model said so\"*."
    )
    scale = st.select_slider(
        "How freely may the trend bend? (changepoint_prior_scale)",
        options=[0.01, 0.05, 0.1, 0.5], value=0.05,
        help="Low = an almost straight trend. High = the trend follows every wobble — and keeps "
             "going in whatever direction it happened to end on.",
    )
    run = cached_prophet(train, horizon, scale)
    predicted = run["predicted"]
    prophet_mape = F.forecast_metrics(test, predicted["yhat"])["MAPE"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=predicted["ds"], y=predicted["yhat_upper"], mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=predicted["ds"], y=predicted["yhat_lower"], mode="lines", fill="tonexty",
                             fillcolor="rgba(30,136,229,0.15)", line=dict(width=0), name="95% range"))
    fig.add_trace(go.Scatter(x=train.index[-40:], y=train.values[-40:], mode="lines", name="history",
                             line=dict(color=GREY)))
    fig.add_trace(go.Scatter(x=test.index, y=test.values, mode="lines+markers", name="actual",
                             line=dict(color="black", width=3)))
    fig.add_trace(go.Scatter(x=predicted["ds"], y=predicted["yhat"], mode="lines", name="Prophet",
                             line=dict(color=BLUE, dash="dash", width=3)))
    fig.update_layout(height=400, title="<b>Prophet forecast with its uncertainty range</b>",
                      yaxis_title="weekly sales", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")

    c = st.columns(3)
    c[0].metric("Prophet MAPE", f"{prophet_mape:.2f}%", f"{prophet_mape - baseline_mape:+.2f} vs same-week-last-year",
                delta_color="inverse")
    c[1].metric("Weeks inside the 95% range", f"{F.interval_coverage(test, predicted):.0%}",
                help="A 95% range should miss about 1 week in 20. Catching every week means the range is "
                     "wider than it needs to be — honest about uncertainty with little history, not a triumph.")
    c[2].metric("Trend change over the 13 weeks", f"{run['trend_drift_pct']:+.2f}%",
                help="Our sales have no real trend, so this should stay close to 0.")
    st.caption(
        "Slide the setting to 0.5 and watch the trend change grow: the model starts to believe that recent "
        "wiggles are a new direction. Over 13 weeks the damage is small — over a year, this one setting decides "
        "whether the forecast is a plan or a fantasy. Our sales have no trend, so a *low* value is right."
    )

    parts = run["components"]
    fig = make_subplots(rows=3, cols=1, vertical_spacing=0.1,
                        subplot_titles=("Trend (dotted lines = where it may bend)", "Yearly season", "Holiday effects"))
    fig.add_trace(go.Scatter(x=parts["trend"]["ds"], y=parts["trend"]["trend"], mode="lines",
                             line=dict(color=ORANGE), showlegend=False), row=1, col=1)
    for date in parts["changepoints"].loc[parts["changepoints"]["delta"].abs() > 1e-4, "ds"]:
        fig.add_vline(x=date, line_dash="dot", line_color=GREY, row=1, col=1)
    if "yearly" in parts:
        yearly = parts["yearly"].drop_duplicates("ds").sort_values("ds")
        fig.add_trace(go.Scatter(x=yearly["ds"], y=yearly["yearly"], mode="lines", line=dict(color=GREEN),
                                 showlegend=False), row=2, col=1)
    if "holidays" in parts:
        fig.add_trace(go.Bar(x=parts["holidays"]["ds"], y=parts["holidays"]["holidays"], marker_color=PURPLE,
                             showlegend=False), row=3, col=1)
    fig.update_layout(height=620, title="<b>The parts Prophet adds up</b>")
    st.plotly_chart(fig, width="stretch")
    reading = F.prophet_component_reading(parts)
    lines = []
    if "season_peak" in reading:
        (peak_day, peak), (low_day, low) = reading["season_peak"], reading["season_low"]
        lines.append(f"**Yearly season:** around {peak_day.strftime('%d %B')} it adds **{F.money(peak)}** to a "
                     f"normal week; around {low_day.strftime('%d %B')} it takes away {F.money(-low)}.")
    if "holiday_peak" in reading:
        day, effect = reading["holiday_peak"]
        lines.append(f"**Holidays:** the biggest holiday effect is the week of {day.date()}: "
                     f"**{F.money(effect)}** on top of the season.")
    lines.append(f"**Trend:** it moves {F.money(reading['trend_change'])} over the whole history — next to "
                 "nothing, as the Time Series page found.")
    st.markdown("**Reading the parts** (Prophet *adds* them up, so each is an amount in euros):\n\n"
                + "\n".join(f"- {line}" for line in lines))
    st.caption(
        "The holiday bars are knowledge we *gave* the model: Walmart's four holidays (Super Bowl, Labor Day, "
        "Thanksgiving, Christmas) plus the US public calendar. Whether a calendar actually helps is tested next."
    )
    with st.expander("🔎 The maths, if you're curious"):
        st.latex(r"y(t) = \text{trend}(t) + \text{season}(t) + \text{holidays}(t) + \text{noise}")
        st.markdown("The trend is a series of straight pieces; `changepoint_prior_scale` sets how sharply "
                    "the slope may change between them.")
    show_source(F.fit_prophet, "🐍 Fitting Prophet")
    show_source(F.prophet_components, "🐍 Pulling out the components")

    st.divider()
    st.subheader("🐣 The holiday the calendar forgot: Easter")
    st.markdown(
        "The Time Series page ended on its biggest miss: **Easter moved** (24 April 2011 → 8 April 2012), and "
        "every seasonal model copied last year's calendar. Walmart's holiday list has no Easter, and neither has "
        "the US public calendar. Prophet takes a list of dated events — so we add Easter (one row per year) and "
        "test three calendars on the same three folds as before, against Holt-Winters, the best model so far."
    )
    calendar_cv = cached_calendar_cv(sales)
    cal = calendar_cv["scores"]
    fold_cols = [c for c in cal.columns if c.startswith("fold")]
    st.dataframe(
        cal.style.format({c: "{:.2f}%" for c in [*fold_cols, "mean"]})
        .highlight_min(subset=["mean"], color="#c8e6c9"),
        width="stretch", hide_index=True,
    )
    means = cal.set_index("model")["mean"]
    if {"Prophet", "Prophet + Easter"} <= set(calendar_cv["easter_miss"]):
        before, after = calendar_cv["easter_miss"]["Prophet"], calendar_cv["easter_miss"]["Prophet + Easter"]
        c = st.columns(3)
        c[0].metric(f"Easter week {calendar_cv['easter_week']}: miss without Easter", f"{before:+.1f}%")
        c[1].metric("…with Easter in the calendar", f"{after:+.1f}%", f"{abs(after) - abs(before):+.1f} points",
                    delta_color="inverse")
        c[2].metric("Average error, Prophet + Easter", f"{means['Prophet + Easter']:.2f}%",
                    f"{means['Prophet + Easter'] - means['Holt-Winters']:+.2f} vs Holt-Winters", delta_color="inverse")
        st.markdown(
            f"""
**Reading.**
- **Easter week:** without Easter the forecast was {before:+.1f}% off; with it,
  {after:+.1f}%. Prophet had seen only two earlier Easters, so it learned about half the effect — but on the
  right week.
- **Across the folds:** Prophet + Easter averages **{means['Prophet + Easter']:.2f}%**, the lowest of all,
  against {means['Holt-Winters']:.2f}% for Holt-Winters.
- **More is not better:** the whole US public calendar made Prophet *worse*
  ({means['Prophet + US calendar']:.2f}% vs {means['Prophet']:.2f}%). Most public holidays don't move weekly
  retail sales, and every extra event is one more effect to guess from two or three examples. **The right
  holiday helps; more holidays don't.** That is Prophet's real strength: it lets you add *your* business
  knowledge.
"""
        )
    show_source(F.with_easter, "🐍 Adding Easter to the calendar")

# ===========================================================================
# TAB 2 — XGBoost needs rows
# ===========================================================================
with tab_rows:
    st.subheader("XGBoost cannot see time — so we turn time into columns")
    trees = cached_chain_trees(chain, train.index[-1])
    st.markdown(
        "A tree model only sees a table. So each week becomes a row with **calendar columns** (week, "
        "month…), **lag columns** (sales 1 week ago, 52 weeks ago) and a running **counter** `t`."
    )
    preview_cols = ["Date", "Weekly_Sales"] + F.CALENDAR + F.LAGS
    preview = trees["features"][preview_cols].tail(5).copy()
    preview["Date"] = preview["Date"].dt.date
    st.dataframe(preview.style.format({c: "{:,.0f}" for c in ["Weekly_Sales", "lag_1", "lag_52", "roll_4"]}
                                      | {"week_sin": "{:.2f}", "week_cos": "{:.2f}"}),
                 width="stretch", hide_index=True)
    last = trees["features"].iloc[-1]
    st.markdown(
        f"**Reading one row** — the week of {last['Date'].date()}: the tree sees *week {int(last['week'])} of the "
        f"year, month {int(last['month'])}*, last week's sales (`lag_1` = {F.money(last['lag_1'])}), the same week "
        f"last year (`lag_52` = {F.money(last['lag_52'])}) and the average of the last 4 weeks (`roll_4` = "
        f"{F.money(last['roll_4'])}). It must predict `Weekly_Sales` ({F.money(last['Weekly_Sales'])}) from "
        "these columns alone — it has no idea that rows follow each other in time. `week_sin`/`week_cos` "
        "put the weeks on a circle, so week 52 and week 1 count as neighbours."
    )
    show_source(F.build_features, "🐍 Building the feature table")

    st.divider()
    st.subheader("On one series: no tree model really helps")
    one = pd.DataFrame([
        {"model": "same week last year", "MAPE": baseline_mape},
        {"model": "XGBoost", "MAPE": F.forecast_metrics(test, trees["XGBoost"])["MAPE"]},
        {"model": "Random Forest", "MAPE": F.forecast_metrics(test, trees["Random Forest"])["MAPE"]},
    ]).sort_values("MAPE")
    col_t, col_r = st.columns([2, 3])
    col_t.dataframe(one.style.format({"MAPE": "{:.2f}%"}), width="stretch", hide_index=True)
    col_r.info(
        f"**Why? Too few rows, not bad tuning.** 143 weeks − 52 (the `lag_52` column needs a full year) − 13 "
        f"test weeks = only **{trees['train_rows']} rows** to learn from. XGBoost, which builds hundreds of trees, "
        "has nothing to split and loses. The random forest lands close to the baseline — steadier on tiny data, "
        "but a tie with *same week last year* is no reason to run it. More tuning won't fix this — **more rows** will."
    )

    st.divider()
    st.subheader("Give it rows: one model for all 45 stores")
    panel = cached_panel(stores)
    st.markdown(
        f"The same sales, split by store: **{panel['train_rows']:,} training rows** instead of "
        f"{trees['train_rows']}. Store size and type now become useful columns too."
    )
    scores = panel["scores"]
    rows = [{"approach": "same week last year", "MAPE": scores["seasonal naive (t-52)"]["MAPE"], "fit time": "0 s",
             "models to maintain": 0},
            {"approach": "XGBoost — one model", "MAPE": scores["XGBoost (one model)"]["MAPE"],
             "fit time": f"{panel['seconds']:.1f} s", "models to maintain": 1}]
    if st.button("🏪 Compare with one Holt-Winters model per store") or "ets_stores" in st.session_state:
        st.session_state["ets_stores"] = cached_ets_per_store(stores)
        ets = st.session_state["ets_stores"]
        rows.append({"approach": "Holt-Winters — one per store", "MAPE": ets["mape"],
                     "fit time": f"{ets['seconds']:.1f} s", "models to maintain": ets["models"]})
    st.dataframe(pd.DataFrame(rows).style.format({"MAPE": "{:.2f}%"}), width="stretch", hide_index=True)
    st.info(
        "With enough rows **XGBoost beats the baseline**. A separate classical model per store can be about as "
        "accurate — but that is 45 models to build, store and watch. With 4,000 stores it is 4,000; the booster "
        "is still **one**. That is what *forecasting at scale* means: an operations advantage more than an "
        "accuracy one."
    )
    importance = panel["importance"]
    fig = px.bar(x=importance.values, y=importance.index, orientation="h",
                 labels={"x": "importance", "y": ""}, title="<b>What the store model relies on</b>")
    fig.update_layout(height=380)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"`{importance.idxmax()}` — sales in the same week last year — carries {importance.max():.0%} of the model. "
        "In plain words: XGBoost learned to take *last year* and adjust it. Sensible — and the reason it beats that "
        "baseline only modestly."
    )
    show_source(F.ets_per_store, "🐍 One Holt-Winters model per store")

# ===========================================================================
# TAB 3 — Overfitting
# ===========================================================================
with tab_fit:
    st.subheader("When to stop learning")
    st.markdown(
        "XGBoost adds trees one after another, each fixing the previous ones' mistakes. Training error keeps "
        "falling forever — but at some point the model starts **memorising** instead of learning. **Early "
        "stopping** watches a slice of *later* weeks the model does not train on and stops when that stops improving."
    )
    c = st.columns(3)
    c[0].metric("Trees allowed", panel["n_estimators"])
    c[1].metric("Trees actually used", panel["best_iteration"] + 1)
    c[2].metric("Share of the budget", f"{(panel['best_iteration'] + 1) / panel['n_estimators']:.0%}")
    st.caption(
        f"Reading: the model was allowed {panel['n_estimators']} trees, but the error on the held-back later weeks "
        f"stopped improving after {panel['best_iteration'] + 1}. The remaining trees would only have memorised the "
        "training weeks. The check-weeks are always the *latest* ones — a random slice would let the model peek "
        "at the future it is supposed to forecast."
    )
    show_source(F.time_ordered_validation_split, "🐍 The validation slice: always the LAST weeks, never random")

    st.divider()
    st.subheader("Deeper trees: better on the past, not on the future")
    depths = cached_depths(stores)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=depths["max_depth"], y=depths["train MAPE"], mode="lines+markers",
                             name="error on training weeks", line=dict(color=BLUE)))
    fig.add_trace(go.Scatter(x=depths["max_depth"], y=depths["test MAPE"], mode="lines+markers",
                             name="error on new weeks", line=dict(color=ORANGE, width=3)))
    fig.update_layout(height=340, xaxis_title="tree depth (max_depth)", yaxis_title="MAPE %",
                      legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig, width="stretch")
    shallow, deep = depths.iloc[0], depths.iloc[-1]
    best_depth = depths.loc[depths["test MAPE"].idxmin()]
    st.info(
        f"**Reading:** from depth {int(shallow['max_depth'])} to {int(deep['max_depth'])}, the error on training "
        f"weeks falls from {shallow['train MAPE']:.1f}% to **{deep['train MAPE']:.1f}%** — but on new weeks only from "
        f"{shallow['test MAPE']:.1f}% to {deep['test MAPE']:.1f}%. The best depth for new weeks is "
        f"**{int(best_depth['max_depth'])}** ({best_depth['test MAPE']:.2f}%); beyond it the gap between the lines grows "
        f"to {deep['gap']:.1f} points. That gap *is* overfitting: the model describes these particular weeks instead "
        "of how sales work. Judge a model on the orange line, never the blue one."
    )
    st.dataframe(depths.style.format({"train MAPE": "{:.2f}%", "test MAPE": "{:.2f}%", "gap": "{:+.2f}"}),
                 width="stretch", hide_index=True)

# ===========================================================================
# TAB 4 — External data
# ===========================================================================
with tab_external:
    st.subheader("Does outside data help? Test it, don't assume it")
    st.markdown(
        "The dataset includes **temperature, fuel price, inflation (CPI) and unemployment** per store and week. "
        "Adding them is easy — the question is whether they make the forecast better."
    )
    if st.button("🌦️ Run the comparison (3 folds)") or "external" in st.session_state:
        st.session_state["external"] = cached_external(stores)
        ext = st.session_state["external"]
        st.dataframe(ext.style.format({c: "{:.2f}%" for c in ext.columns if c != "feature set"}),
                     width="stretch", hide_index=True)
        diff = ext["mean"].iloc[1] - ext["mean"].iloc[0]
        spread = ext.iloc[0, 1:4].max() - ext.iloc[0, 1:4].min()
        st.info(
            f"The extra columns change the average error by only **{diff:+.2f} points** — far smaller than the "
            f"**{spread:.2f}-point** difference between the three test periods. That is no measurable effect.\n\n"
            "Why? These numbers move slowly and hit all stores at once, so they say little about *which store* "
            "sells more *next week*. Outside data has to be **close to what you predict** — weather matters for an "
            "ice-cream maker, inflation statistics do not for next week's department-store sales."
        )
    else:
        st.info("Click to compare the model with and without the four external columns.")

# ===========================================================================
# TAB 5 — Grain & metrics
# ===========================================================================
with tab_grain:
    st.subheader("One level finer: ~3,300 store departments")
    st.markdown("The same model, the same columns — now one row per **store × department × week**. "
                "Watch what happens to the **metric**, not the model.")
    if st.button("🔬 Train the department model") or "departments" in st.session_state:
        st.session_state["departments"] = cached_departments(depts)
        dep = st.session_state["departments"]
        c = st.columns(3)
        c[0].metric("MAPE per department", f"{dep['model']['MAPE']:.0f}%")
        c[1].metric("Same predictions, summed to chain", f"{dep['summed']['MAPE']:.2f}%")
        c[2].metric("Baseline per department", f"{dep['baseline']['MAPE']:.0f}%")
        st.warning(
            f"A {dep['model']['MAPE']:.0f}% error sounds like a broken model — yet the **same predictions**, added up, "
            f"are off by only {dep['summed']['MAPE']:.2f}%. The model is fine; **the ruler broke**.\n\n"
            f"MAPE divides by the actual sales. {dep['small_rows']:,} of {dep['test_rows']:,} test rows are departments "
            "that sold under 100 in a week: being 200 off on a week that sold 40 counts as a 500% error, though it is "
            "small change in euros. When even the baseline scores absurdly, suspect the metric. At this level, use "
            "**MAE** (average miss in euros) instead."
        )
        ex = dep.get("example")
        if ex:
            st.markdown(
                f"**One such row:** store {ex['store']}, department {ex['dept']}, week of {ex['date']}: sold "
                f"**€{ex['actual']:,.0f}**, forecast €{ex['prediction']:,.0f}. A miss of "
                f"€{abs(ex['actual'] - ex['prediction']):,.0f} — pocket money for a chain selling ~€46M a week — "
                f"but MAPE records it as **{ex['pct']:.0f}%**, and that row weighs as much in the average as a "
                "department selling €50,000."
            )
    else:
        st.info("Click to train one XGBoost for all store departments (a few seconds).")

# ===========================================================================
# TAB 6 — Trends trees can't see
# ===========================================================================
with tab_trend:
    st.subheader("Trees cannot forecast beyond what they have seen")
    st.markdown(
        "A tree predicts by averaging past values — so it can **never predict a value higher than anything in "
        "its training data**. On a growing series that is a problem. The common fix: let a **straight line** "
        "handle the trend and XGBoost everything else (*trend + XGBoost*). Does that always help?"
    )
    if st.button("📐 Run the experiment") or "detrend" in st.session_state:
        st.session_state["detrend"] = (cached_detrending(stores), cached_control())
        regimes, control = st.session_state["detrend"]
        st.dataframe(regimes.style.format({c: "{:.2f}%" for c in regimes.columns if c.startswith("fold") or c == "mean"}),
                     width="stretch", hide_index=True)
        mean = regimes.set_index(["features", "model"])["mean"]
        st.info(
            f"**Without lag columns** the tree has no idea of the sales level: {mean[('no lags', 'plain XGBoost')]:.1f}% "
            f"error. Adding the straight line fixes it: {mean[('no lags', 'trend + XGBoost')]:.1f}%.\n\n"
            f"**With lag columns** the same fix makes things *worse* ({mean[('with lags', 'trend + XGBoost')]:.2f}% vs "
            f"{mean[('with lags', 'plain XGBoost')]:.2f}%): `lag_52` already tells the tree last year's level, so the "
            "extra line only adds its own guesswork.\n\n"
            "**The rule:** apply a fix because of a diagnosis, not because it is \"best practice\"."
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=control["history"]["Date"], y=control["history"]["Weekly_Sales"], mode="lines",
                                 name="history", line=dict(color=GREY)))
        fig.add_trace(go.Scatter(x=control["dates"], y=control["actual"], mode="lines", name="actual",
                                 line=dict(color="black", width=3)))
        fig.add_trace(go.Scatter(x=control["dates"], y=control["plain"], mode="lines", name="plain XGBoost",
                                 line=dict(color=RED, dash="dash")))
        fig.add_trace(go.Scatter(x=control["dates"], y=control["hybrid"], mode="lines", name="trend + XGBoost",
                                 line=dict(color=GREEN, dash="dash")))
        fig.update_layout(height=380, yaxis_title="CO₂ (ppm)", legend=dict(orientation="h", y=-0.25),
                          title="<b>A series with a real trend (weekly CO₂): plain XGBoost hits a ceiling</b>")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"Plain XGBoost flattens out at the highest level it has seen; the trend line carries it upward — "
            f"{control['plain_mape'] / control['hybrid_mape']:.1f}× smaller error ({control['plain_mape']:.3f}% → "
            f"{control['hybrid_mape']:.3f}%). Same code, opposite conclusion — decided by whether the series has a trend."
        )
    else:
        st.info("Click to compare plain XGBoost with trend + XGBoost, with and without lag columns.")
    show_source(F.fit_hybrid, "🐍 Trend + XGBoost (the hybrid)")

# ===========================================================================
# TAB 7 — Verdict
# ===========================================================================
with tab_verdict:
    st.subheader("Who wins? It depends on the level of detail")
    cal_means = cached_calendar_cv(sales)["scores"].set_index("model")["mean"]
    rows = [{"level": "chain total, best model", "result": f"Prophet + Easter {cal_means['Prophet + Easter']:.2f}% "
             f"vs Holt-Winters {cal_means['Holt-Winters']:.2f}% (3-fold average) — **wins**, thanks to one holiday"},
            {"level": "chain total (1 series)", "result": f"XGBoost {F.forecast_metrics(test, trees['XGBoost'])['MAPE']:.2f}% "
             f"vs same-week-last-year {baseline_mape:.2f}% — **loses**: too few rows"},
            {"level": "per store (45 series)", "result": f"XGBoost {scores['XGBoost (one model)']['MAPE']:.2f}% vs "
             f"{scores['seasonal naive (t-52)']['MAPE']:.2f}% — **wins**, as one fast model"}]
    if "departments" in st.session_state:
        dep = st.session_state["departments"]
        rows.append({"level": f"per department ({dep['series']:,} series)",
                     "result": f"MAPE {dep['model']['MAPE']:.0f}% — **the metric breaks**, not the model"})
    else:
        rows.append({"level": "per department (~3,300 series)", "result": "run it on the *Grain & metrics* tab"})
    st.table(pd.DataFrame(rows).set_index("level"))

    st.markdown(
        f"""
| The slides say | On this data |
|---|---|
| **Prophet**: explainable parts — trend, season, holidays | **Holds** — and its accuracy comes from what you tell it: with Easter added it has the lowest average error ({cal_means['Prophet + Easter']:.2f}%); the whole US calendar made it worse ({cal_means['Prophet + US calendar']:.2f}%). |
| **XGBoost**: strong with many rows; loses on one short series; cannot continue a trend | **Holds, every part** — see *XGBoost needs rows* and *Trends trees can't see*. |
| Feature engineering matters; outside data *can* be added | **Holds** — last year's sales did most of the work; the outside data added nothing measurable. |
| Forecasting at scale | **The real case for XGBoost** — one model instead of one per store. |
| Pick the level, the baseline and the metric first | **The lesson of this page** — the winner changed with each. |
"""
    )
    typical = test.mean()
    st.markdown(
        f"**In euros:** on a typical {F.money(typical)} week, the chain-level winner's "
        f"{cal_means['Prophet + Easter']:.2f}% average miss is about **{F.money(cal_means['Prophet + Easter'] / 100 * typical)}**, "
        f"against {F.money(cal_means['Holt-Winters'] / 100 * typical)} for Holt-Winters — a small difference of "
        f"about {F.money((cal_means['Holt-Winters'] - cal_means['Prophet + Easter']) / 100 * typical)} a week on average "
        "(Holt-Winters still wins one of the three folds). The bigger gain is on the weeks that matter: the Easter "
        "week miss is halved — earned by one row of domain knowledge per year, not by a more complex model."
    )
    st.info(
        "**Why no LSTM (deep learning)?** It is powerful on long, detailed series — energy use per hour, say, "
        "with 8,760 values a year. Our chain total has 143. A network with more parameters than data points "
        "memorises the past and forecasts noise. Knowing when *not* to use a tool is part of the skill."
    )
    with st.container(border=True):
        st.markdown(
            """
🎯 **What to remember**
- **Start simple.** "Same week last year" is free — every model has to beat it to earn its place.
- **Your business knowledge is a feature.** Telling Prophet where Easter falls gave the lowest error on the
  chain total — but the *right* holidays help, not all of them.
- **Match the tool, the level and the metric to the data:** Prophet to *explain* a forecast, XGBoost for
  *many* series — and the same model can look broken or brilliant depending on level and metric.
"""
        )
    show_sibling_script("forecasting.py", "📄 The full sibling script: predictions/forecasting.py")
