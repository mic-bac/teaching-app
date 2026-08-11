"""
3_Recommender.py
----------------
Recommender Systems lesson page.

Surfaces the sibling ``recommender/`` repo's three method families as an
interactive lesson: Content-Based Filtering, Collaborative Filtering
(neighborhood + a bias baseline + matrix factorization), and Association Rules.
All logic lives in ``utils/recommender_utils`` (UI-free); this page only caches,
renders, and shows the real source code that runs.

Graceful behavior: the small MovieLens/Groceries CSVs are loaded from the sibling
repo on the fly and cached. Expensive similarity rows are computed on demand
rather than as giant N×N matrices, so the page stays responsive.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from utils import recommender_utils as R
from utils.teaching import source_of

st.set_page_config(page_title="Recommender Systems", layout="wide", page_icon="🎯")

ROOT = Path(__file__).resolve().parents[1]
SIBLING = ROOT / "recommender"


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


def _rules_network(rules: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """A simple circular network of the top rules by lift (items = nodes)."""
    top = rules.nlargest(top_n, "lift")
    edges = [
        (a, c, row["lift"])
        for _, row in top.iterrows()
        for a in row["antecedents"]
        for c in row["consequents"]
    ]
    nodes = sorted({n for a, c, _ in edges for n in (a, c)})
    pos = {
        n: (np.cos(2 * np.pi * k / len(nodes)), np.sin(2 * np.pi * k / len(nodes)))
        for k, n in enumerate(nodes)
    }

    edge_x, edge_y = [], []
    for a, c, _ in edges:
        edge_x += [pos[a][0], pos[c][0], None]
        edge_y += [pos[a][1], pos[c][1], None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(width=1, color="#b0bec5"), hoverinfo="none"))
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
        mode="markers+text", text=nodes, textposition="top center",
        marker=dict(size=18, color="#1e88e5", line=dict(width=1, color="white")),
        hovertemplate="<b>%{text}</b><extra></extra>",
    ))
    fig.update_layout(
        showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
    )
    return fig


# ===========================================================================
# Cached data / model builders (keep utils Streamlit-free; cache here)
# ===========================================================================
@st.cache_data(show_spinner=False)
def cached_movies():
    return R.load_movies()


@st.cache_data(show_spinner=False)
def cached_ratings():
    return R.load_ratings()


@st.cache_data(show_spinner=False)
def cached_groceries():
    return R.load_groceries()


@st.cache_resource(show_spinner=False)
def cached_tfidf():
    return R.build_tfidf(cached_movies())


@st.cache_resource(show_spinner=False)
def cached_user_item():
    return R.build_user_item_matrix(cached_ratings(), cached_movies())


@st.cache_resource(show_spinner=False)
def cached_user_sim():
    return R.user_similarity(cached_user_item())


@st.cache_resource(show_spinner=False)
def cached_item_user_sparse():
    return R.build_item_user_sparse(cached_user_item())


@st.cache_resource(show_spinner=False)
def cached_encoded():
    enc, u2i, i2i, i2it = R.encode_ratings(cached_ratings())
    train, test = train_test_split(enc, test_size=0.25, random_state=42)
    return enc, train, test, u2i, i2i, i2it


@st.cache_resource(show_spinner=False)
def cached_baseline():
    _, train, test, u2i, i2i, _ = cached_encoded()
    mu, b_u, b_i = R.fit_baseline(train, len(u2i), len(i2i))
    test_rmse = R.rmse(R.predict_baseline(mu, b_u, b_i, test[:, 0], test[:, 1]), test[:, 2])
    return mu, b_u, b_i, test_rmse


@st.cache_data(show_spinner=False)
def cached_transactions():
    return R.build_transactions(cached_groceries())


@st.cache_data(show_spinner=False)
def cached_itemsets(min_support: float):
    return R.run_apriori(cached_transactions(), min_support=min_support)


# ===========================================================================
# Page header
# ===========================================================================
st.title("🎯 Recommender Systems")

st.markdown(
    """
How does Netflix decide what to autoplay, or Amazon fill *"frequently bought
together"*? This lesson walks through the three classic answers, each on a real
dataset you can poke at:

- **Content-Based Filtering** — *"similar items"*, from the items' own features.
- **Collaborative Filtering** — *"people like you also liked…"*, from rating patterns.
- **Association Rules** — *"bought together"*, from market-basket transactions.

Every tab shows the **real code** that produces the result, not a slide summary.
"""
)

tab_overview, tab_cbf, tab_nb, tab_mf, tab_ar = st.tabs(
    [
        "🧭 Overview",
        "🎬 Content-Based",
        "👥 Collaborative — Neighborhood",
        "🔢 Collaborative — Matrix Factorization",
        "🛒 Association Rules",
    ]
)

# ===========================================================================
# OVERVIEW TAB
# ===========================================================================
with tab_overview:
    st.header("Which method, and when?")
    st.markdown(
        """
There are two useful ways to look at recommenders. **Methodically**, they split by
*what data they lean on*: item features, user ratings, or raw transactions.
**Practically**, they differ in whether they *personalize* to an individual and
what they need to get started.
"""
    )
    st.graphviz_chart(R.taxonomy_dot())

    st.subheader("The three families at a glance")
    st.table(
        pd.DataFrame(
            [
                {
                    "Method": "Content-Based Filtering",
                    "Personalized?": "Per item (to your history)",
                    "Needs": "Item features (e.g. genres)",
                    "Strength": "Works for new/rare items, no other users needed",
                    "Weakness": "Obvious recs; heavy feature engineering",
                },
                {
                    "Method": "Collaborative Filtering",
                    "Personalized?": "Yes (individual)",
                    "Needs": "A user–item rating matrix",
                    "Strength": "Domain-free; finds latent taste",
                    "Weakness": "Cold start & sparsity (few ratings)",
                },
                {
                    "Method": "Association Rules",
                    "Personalized?": "No (whole population)",
                    "Needs": "Transactions only",
                    "Strength": "Simple, great for cross-selling",
                    "Weakness": "Popularity bias; not personalized",
                },
            ]
        )
    )
    st.caption(
        "This page follows the lecture: content-based → collaborative "
        "(neighborhood, then bias-corrected matrix factorization) → association rules."
    )

# ===========================================================================
# CONTENT-BASED TAB
# ===========================================================================
with tab_cbf:
    st.header("Content-Based Filtering")
    st.markdown(
        """
The idea: **describe each item by its content, then recommend the nearest items.**
Here each movie is described by its *genres*. We turn those genres into a vector
with **TF-IDF** (rare genres weigh more than common ones), then rank other movies
by **cosine similarity** — the angle between genre vectors. No other users are
involved, so this works even for brand-new movies.
"""
    )

    movies = cached_movies()
    _, tfidf, title_to_idx = cached_tfidf()
    titles = sorted(title_to_idx.index)

    default = titles.index("Toy Story (1995)") if "Toy Story (1995)" in titles else 0
    choice = st.selectbox("Pick a movie you liked", titles, index=default)

    recs = R.get_recommendations(choice, tfidf, movies, title_to_idx, top_n=10)
    if isinstance(recs, str):
        st.warning(recs)
    else:
        st.markdown(f"**Because you liked _{choice}_, you might also like:**")
        st.dataframe(
            recs.style.format({"similarity": "{:.3f}"}),
            use_container_width=True,
            hide_index=True,
        )
        ils = R.intra_list_similarity(list(recs["title"]), tfidf, title_to_idx)
        st.metric("Intra-list similarity (diversity)", f"{ils:.3f}")
        st.caption(
            "Closer to 1 means the recommendations are very alike — content-based "
            "filtering is accurate but tends to give *obvious* suggestions."
        )

    st.divider()
    st.subheader("Build a taste profile from several favorites")
    st.markdown(
        "Real users like more than one thing. We can average several movies' genre "
        "vectors — **weighted** by how much you like each — into one *profile* vector "
        "and recommend against that."
    )
    faves = st.multiselect(
        "Your favorite movies",
        titles,
        default=[t for t in ["Toy Story (1995)", "Jumanji (1995)"] if t in titles],
    )
    if faves:
        weights = [
            st.slider(f"Weight — {t}", 0.1, 1.0, 1.0, 0.1, key=f"w_{t}")
            for t in faves
        ]
        profile_recs = R.get_recommendations_for_user_profile(
            faves, weights, tfidf, movies, title_to_idx, top_n=10
        )
        if isinstance(profile_recs, str):
            st.warning(profile_recs)
        else:
            st.dataframe(
                profile_recs.style.format({"similarity": "{:.3f}"}),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("Pick at least one favorite to build a profile.")

    with st.expander("🗺️ Show a 2D genre map (PCA of the TF-IDF vectors)"):
        pca = PCA(n_components=2)
        coords = pca.fit_transform(tfidf.toarray())
        plot_df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1],
                                "title": movies["title"], "genres": movies["genres"]})
        plot_df["highlight"] = np.where(plot_df["title"] == choice, choice, "other movies")
        fig = px.scatter(
            plot_df, x="x", y="y", color="highlight",
            hover_name="title", hover_data=["genres"],
            color_discrete_map={choice: "#e53935", "other movies": "#b0bec5"},
            title="Movies arranged by genre similarity (TF-IDF → PCA)",
        )
        fig.update_layout(legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    show_source(R.build_tfidf, "🐍 How TF-IDF is built")
    show_source(R.get_recommendations, "🐍 How recommendations are ranked")
    show_sibling_script("content_based_filtering.py", "📄 The full sibling script")

# ===========================================================================
# COLLABORATIVE — NEIGHBORHOOD TAB
# ===========================================================================
with tab_nb:
    st.header("Collaborative Filtering — Neighborhood methods")
    st.markdown(
        """
Now we ignore item features entirely and learn from **ratings**. Neighborhood
methods find the nearest *neighbors* by cosine similarity:

- **User-based** — find users who rate like you, recommend what *they* liked.
- **Item-based** — find items rated similarly by the same people (Amazon's
  *"customers who bought this…"*).
"""
    )

    user_item = cached_user_item()
    user_sim = cached_user_sim()
    ratings = cached_ratings()
    movies = cached_movies()

    col_user, col_item = st.columns(2)

    with col_user:
        st.subheader("👤 User-based")
        uid = st.selectbox("Pick a user", list(user_item.index), index=0)
        ub = R.get_user_based_recommendations(uid, user_item, user_sim)
        st.markdown("**Recommended for this user (weighted by neighbor similarity):**")
        st.dataframe(
            ub.rename("score").reset_index().rename(columns={"title": "movie"}),
            use_container_width=True,
            hide_index=True,
        )
        with st.expander(f"What user {uid} already rated highly"):
            top_rated = (
                ratings[ratings["userId"] == uid]
                .merge(movies, on="movieId")
                .sort_values("rating", ascending=False)[["title", "rating"]]
                .head(10)
            )
            st.dataframe(top_rated, use_container_width=True, hide_index=True)

    with col_item:
        st.subheader("🎬 Item-based")
        item_sparse, item_labels = cached_item_user_sparse()
        default_m = item_labels.index("Toy Story (1995)") if "Toy Story (1995)" in item_labels else 0
        movie = st.selectbox("Pick a movie", item_labels, index=default_m, key="nb_movie")
        ib = R.get_item_based_recommendations(movie, item_sparse, item_labels)
        if isinstance(ib, str):
            st.warning(ib)
        else:
            st.markdown(f"**Movies most similar to _{movie}_ by co-rating:**")
            st.dataframe(
                ib.rename("similarity").reset_index().rename(columns={"index": "movie"}),
                use_container_width=True,
                hide_index=True,
            )

    show_source(R.get_user_based_recommendations, "🐍 User-based recommendations (weighted)")
    show_source(R.get_item_based_recommendations, "🐍 Item-based recommendations")
    show_sibling_script("collaborative_filtering.py", "📄 The full sibling script")

# ===========================================================================
# COLLABORATIVE — MATRIX FACTORIZATION TAB
# ===========================================================================
with tab_mf:
    st.header("Collaborative Filtering — Matrix Completion")
    st.markdown(
        """
The rating matrix is mostly **empty** — most people rate few movies. *Matrix
completion* fills the blanks. We build it in two steps, exactly as in the lecture.
"""
    )

    _, train, test, u2i, i2i, i2it = cached_encoded()
    ratings = cached_ratings()
    movies = cached_movies()
    mu, b_u, b_i, base_rmse = cached_baseline()

    # ---- Step 1: the bias baseline --------------------------------------
    st.subheader("Step 1 · Bias-corrected baseline")
    st.markdown(
        r"""
The simplest sensible guess corrects for two biases:

$$\hat{r}(u,i) = \mu + b_u + b_i$$

**μ** is the global average rating, **bᵤ** says whether a user rates high or low in
general, and **bᵢ** says whether a movie is loved or panned. It's robust and easy
to read — but every user gets the *same* ranking, just shifted. Pick a user and a
movie to see the decomposition:
"""
    )
    c1, c2 = st.columns(2)
    demo_user = c1.selectbox("User", list(u2i.keys()), index=0, key="bias_user")
    movie_titles = sorted(movies["title"])
    demo_title = c2.selectbox(
        "Movie", movie_titles,
        index=movie_titles.index("Toy Story (1995)") if "Toy Story (1995)" in movie_titles else 0,
        key="bias_movie",
    )
    demo_movie_id = movies.loc[movies["title"] == demo_title, "movieId"].iloc[0]
    if demo_movie_id in i2i:
        u_idx, i_idx = u2i[demo_user], i2i[demo_movie_id]
        pred = float(np.clip(mu + b_u[u_idx] + b_i[i_idx], 0.5, 5.0))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Global μ", f"{mu:.2f}")
        m2.metric("User bias bᵤ", f"{b_u[u_idx]:+.2f}")
        m3.metric("Movie bias bᵢ", f"{b_i[i_idx]:+.2f}")
        m4.metric("Predicted rating", f"{pred:.2f}")
    else:
        st.info("That movie has no ratings, so it has no learned bias — pick another.")
    st.caption(f"Baseline test-set RMSE: **{base_rmse:.4f}** (lower is better).")

    st.divider()

    # ---- Step 2: matrix factorization -----------------------------------
    st.subheader("Step 2 · Matrix factorization (learned taste)")
    st.markdown(
        r"""
The baseline knows *who rates high* and *what's popular*, but not individual taste.
Matrix factorization adds a short **latent-factor** vector to each user and movie:

$$\hat{r}(u,i) = \mu + b_u + b_i + p_u \cdot q_i$$

The dot product $p_u \cdot q_i$ captures dimensions the model discovers on its own
(*"action-ness"*, *"for-kids-ness"*). We learn everything with **stochastic
gradient descent** — click below to watch the training error fall.
"""
    )

    epochs = st.slider("Training epochs", 5, 30, 20, step=5)
    if st.button("🚀 Train the model"):
        model = R.MatrixFactorization(len(u2i), len(i2i), n_factors=20, n_epochs=epochs)
        history: list[dict] = []
        progress = st.progress(0.0, text="Starting…")
        chart_ph = st.empty()

        def _on_epoch(epoch, train_rmse):
            history.append({"epoch": epoch, "train RMSE": train_rmse})
            progress.progress(epoch / epochs,
                              text=f"Epoch {epoch}/{epochs} — train RMSE {train_rmse:.4f}")
            chart_ph.line_chart(pd.DataFrame(history).set_index("epoch"))

        with st.spinner("Running SGD over the observed ratings…"):
            model.fit(train, on_epoch_end=_on_epoch)
        progress.empty()

        test_rmse = R.rmse(model.predict(test[:, 0], test[:, 1]), test[:, 2])
        st.session_state["mf_model"] = model
        st.session_state["mf_test_rmse"] = test_rmse

    if "mf_model" in st.session_state:
        model = st.session_state["mf_model"]
        test_rmse = st.session_state["mf_test_rmse"]

        st.markdown("#### Accuracy — did the latent factors help?")
        cbar, cnum = st.columns([2, 1])
        cbar.bar_chart(
            pd.DataFrame(
                {"test RMSE": [base_rmse, test_rmse]},
                index=["Baseline (bias only)", "Matrix factorization"],
            ),
            sort=False,
        )
        delta = test_rmse - base_rmse
        cnum.metric("MF test RMSE", f"{test_rmse:.4f}", f"{delta:+.4f} vs baseline",
                    delta_color="inverse")
        cnum.caption("Lower is better, so a negative delta means MF beats the baseline.")

        st.markdown("#### Recommendations from the trained model")
        rec_user = st.selectbox("Recommend for user", list(u2i.keys()), index=0, key="mf_user")
        mf_recs = R.get_mf_recommendations(
            rec_user, model, movies, ratings, u2i, i2i, i2it, num_recommendations=10
        )
        st.dataframe(
            mf_recs.style.format({"predicted_rating": "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("🗺️ Show the learned 2D 'taste space' (PCA of movie factors)"):
            coords = PCA(n_components=2).fit_transform(model.q)
            taste = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1],
                                  "movieId": [i2it[k] for k in range(len(i2it))]})
            taste = taste.merge(movies, on="movieId")
            fig = px.scatter(
                taste.sample(min(1500, len(taste)), random_state=42),
                x="x", y="y", hover_name="title", hover_data=["genres"],
                title="Movies close together are rated similarly (learned factors → PCA)",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Set the number of epochs and click **🚀 Train the model** to begin.")

    show_source(R.fit_baseline, "🐍 The bias baseline")
    show_source(R.MatrixFactorization.fit, "🐍 The matrix-factorization training loop")
    show_sibling_script("collaborative_filtering.py", "📄 The full sibling script")

# ===========================================================================
# ASSOCIATION RULES TAB
# ===========================================================================
with tab_ar:
    st.header("Association Rules")
    st.markdown(
        """
No ratings, no features — just **transactions**. We mine rules of the form
*"if a basket contains **X**, it often also contains **Y**"* from grocery receipts,
scored by three numbers:

- **Support** — how often X and Y appear together (popularity).
- **Confidence** — of baskets with X, how many also have Y (reliability).
- **Lift** — how much more than chance they co-occur (**>1** = a real association).

The **Apriori** algorithm finds frequent itemsets, pruning any below a support
threshold, then turns them into rules.
"""
    )

    transactions = cached_transactions()
    st.caption(
        f"{transactions.shape[0]:,} baskets · {transactions.shape[1]} distinct items "
        "(each basket = one member on one day)."
    )

    c1, c2 = st.columns(2)
    min_support = c1.slider("Minimum support", 0.001, 0.02, 0.001, step=0.001, format="%.3f")
    min_confidence = c2.slider("Minimum confidence", 0.05, 0.50, 0.10, step=0.05)

    with st.spinner("Mining frequent itemsets with Apriori…"):
        itemsets = cached_itemsets(min_support)
        rules = R.make_rules(itemsets, min_confidence)

    st.markdown(f"**{len(itemsets)} frequent itemsets → {len(rules)} rules.**")

    st.subheader("Most popular items (by support)")
    support = R.item_support(transactions, top=15)
    fig_supp = px.bar(
        x=support.values, y=support.index, orientation="h",
        labels={"x": "Support (fraction of baskets)", "y": ""},
        color=support.values, color_continuous_scale="Blues",
    )
    fig_supp.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    st.plotly_chart(fig_supp, use_container_width=True)

    if len(rules) == 0:
        st.warning("No rules at these thresholds — try lowering support or confidence.")
    else:
        st.subheader("Top rules by lift")
        top = rules.nlargest(15, "lift")[["rule_str", "support", "confidence", "lift"]]
        st.dataframe(
            top.rename(columns={"rule_str": "rule"}).style.format(
                {"support": "{:.4f}", "confidence": "{:.3f}", "lift": "{:.2f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Support vs. confidence (bubble size & color = lift)")
        fig_scatter = px.scatter(
            rules, x="support", y="confidence", size="lift", color="lift",
            color_continuous_scale="Viridis", hover_name="rule_str",
            labels={"support": "Support", "confidence": "Confidence", "lift": "Lift"},
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("Rule network (top rules by lift)")
        st.plotly_chart(_rules_network(rules), use_container_width=True)

        st.divider()
        st.subheader("🛒 Frequently bought together")
        antecedent_items = sorted({item for a in rules["antecedents"] for item in a})
        default_i = antecedent_items.index("whole milk") if "whole milk" in antecedent_items else 0
        basket_item = st.selectbox("A customer just added…", antecedent_items, index=default_i)
        basket = R.get_basket_recommendations(basket_item, rules)
        if isinstance(basket, str):
            st.warning(basket)
        else:
            st.markdown(f"**Suggest alongside _{basket_item}_:**")
            st.dataframe(
                basket.style.format({"support": "{:.4f}", "confidence": "{:.3f}", "lift": "{:.2f}"}),
                use_container_width=True,
                hide_index=True,
            )

    show_source(R.run_apriori, "🐍 Running Apriori")
    show_source(R.get_basket_recommendations, "🐍 Turning rules into recommendations")
    show_sibling_script("association_rule_mining.py", "📄 The full sibling script")
