"""
recommender_utils.py
--------------------
UI-free, importable logic for the Recommender Systems lesson page.

Adapted from the sibling ``recommender/`` repo's teaching scripts
(``content_based_filtering.py``, ``collaborative_filtering.py``,
``association_rule_mining.py``). Those scripts are ``# %%`` notebooks with
top-level side effects (they read CSVs, print, and open Plotly windows), so they
can't be imported directly. Here their algorithms are refactored into pure
functions that **return data** (DataFrames, arrays, a trained model) and never
touch Streamlit — so the page can cache/render them and the tests can exercise
them deterministically.

Three method families, matching the lecture slides:
- **Content-Based Filtering** — TF-IDF over genres + cosine similarity.
- **Collaborative Filtering** — neighborhood (user/item), a bias-corrected
  baseline (matrix completion), and from-scratch matrix factorization.
- **Association Rules** — Apriori + support/confidence/lift on market baskets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from utils.teaching import load_sibling_module

# Default location of the sibling repo's datasets (read-only).
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "recommender" / "data"


# ===========================================================================
# Data loaders
# ===========================================================================
def load_movies(data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load MovieLens movies; genres become space-joined words for TF-IDF."""
    movies = pd.read_csv(Path(data_dir) / "movie" / "movies.csv")
    movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)
    return movies


def load_ratings(data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load the MovieLens user–movie ratings table."""
    return pd.read_csv(Path(data_dir) / "movie" / "ratings.csv")


def _groceries_module():
    """The sibling's own loader (``recommender/src/groceries_data.py``)."""
    return load_sibling_module("recommender", "src/groceries_data.py")


def load_groceries() -> tuple[pd.DataFrame, str]:
    """Return ``(frame, source)`` — the Kaggle Groceries file, or synthetic baskets.

    Imported from the sibling rather than copied: the synthetic generator plants
    known shopping missions, and the page compares what Apriori finds against
    exactly those (``planted_missions()``). ``source`` is ``"kaggle"`` or
    ``"synthetic"`` so the page can say which one the rules came from.
    """
    return _groceries_module().load_groceries()


def planted_missions() -> pd.DataFrame:
    """The ground truth behind the synthetic baskets, as a table."""
    missions = _groceries_module().PLANTED_MISSIONS
    return pd.DataFrame(
        [{"Mission": name, "Items bought together": ", ".join(items)}
         for name, (items, _, _) in missions.items()]
    )


# ===========================================================================
# Content-Based Filtering (TF-IDF + cosine similarity)
# ===========================================================================
def build_tfidf(movies_df: pd.DataFrame):
    """Vectorize genres with TF-IDF.

    Returns ``(vectorizer, tfidf_matrix, title_to_idx)`` where ``title_to_idx``
    maps a movie title to its (first) row index in ``movies_df``.
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(movies_df["genres"])
    title_to_idx = pd.Series(movies_df.index, index=movies_df["title"])
    # Keep the first index if a title happens to appear twice.
    title_to_idx = title_to_idx[~title_to_idx.index.duplicated(keep="first")]
    return vectorizer, tfidf_matrix, title_to_idx


def genre_set(genres: str) -> set[str]:
    """The genres of one movie as a set (``load_movies`` joined them with spaces)."""
    return set(genres.split()) - {"(no", "genres", "listed)"}


def all_genres(movies_df: pd.DataFrame) -> list[str]:
    """Every genre in the catalogue, alphabetically."""
    return sorted(set().union(*movies_df["genres"].map(genre_set)))


def shared_genres(liked: str, other: str) -> str:
    """``"4 of 5"`` — how many of the liked movie's genres the other movie also has."""
    mine = genre_set(liked)
    return f"{len(mine & genre_set(other))} of {len(mine)}"


def genre_highlight(movies_df: pd.DataFrame, genres: list[str]) -> pd.Series:
    """Per movie, which of the chosen genres it has ("Comedy + Romance"), else "other"."""
    def label(g):
        have = [x for x in genres if x in genre_set(g)]
        return " + ".join(have) if have else "other"
    return movies_df["genres"].map(label)


def get_recommendations(title, tfidf_matrix, movies_df, title_to_idx, top_n=10):
    """Recommend movies whose genres are most similar to ``title``.

    We compute the cosine similarity of this ONE movie against all movies
    (a single sparse row × the matrix) rather than a full 9742×9742 matrix —
    cheap in memory and time, which matters for a live, interactive page.
    """
    if title not in title_to_idx:
        return f"Movie with title '{title}' not found."
    idx = int(title_to_idx[title])
    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).ravel()
    order = np.argsort(sims)[::-1]
    top_idx = order[order != idx][:top_n]  # never recommend the movie itself
    out = movies_df.iloc[top_idx][["title", "genres"]].copy()
    out["similarity"] = sims[top_idx]
    return out.reset_index(drop=True)


def intra_list_similarity(rec_titles, tfidf_matrix, title_to_idx) -> float:
    """Average pairwise cosine similarity within a recommendation list.

    A *diversity* metric: lower means the recommendations are less alike. Content-
    based filtering tends to score high here (it suggests very similar items).
    """
    idxs = [int(title_to_idx[t]) for t in rec_titles if t in title_to_idx]
    if len(idxs) < 2:
        return 0.0
    sub = cosine_similarity(tfidf_matrix[idxs])
    n = len(idxs)
    total = (sub.sum() - np.trace(sub)) / 2  # upper triangle, excl. diagonal
    return float(total / (n * (n - 1) / 2))


def get_recommendations_for_user_profile(titles, weights, tfidf_matrix,
                                         movies_df, title_to_idx, top_n=10):
    """Recommend against a weighted-average "profile" of several liked movies."""
    idxs, kept_weights = [], []
    for t, w in zip(titles, weights):
        if t in title_to_idx:
            idxs.append(int(title_to_idx[t]))
            kept_weights.append(w)
    if not idxs:
        return "Error: none of the provided movies were found."
    profile = np.average(tfidf_matrix[idxs].toarray(), axis=0, weights=kept_weights)
    sims = cosine_similarity(profile.reshape(1, -1), tfidf_matrix).ravel()
    exclude = set(idxs)  # drop the input movies
    order = np.argsort(sims)[::-1]
    top_idx = [j for j in order if j not in exclude][:top_n]
    out = movies_df.iloc[top_idx][["title", "genres"]].copy()
    out["similarity"] = sims[top_idx]
    return out.reset_index(drop=True)


# ===========================================================================
# Collaborative Filtering — neighborhood (user- and item-based)
# ===========================================================================
def build_user_item_matrix(ratings_df, movies_df) -> pd.DataFrame:
    """Pivot ratings into a users × movie-titles matrix (NaN = not rated)."""
    df = ratings_df.merge(movies_df, on="movieId")
    return df.pivot_table(index="userId", columns="title", values="rating")


def user_similarity(user_item_matrix) -> pd.DataFrame:
    """Cosine similarity between every pair of users (small: ~610×610)."""
    sparse = csr_matrix(user_item_matrix.fillna(0).values)
    sim = cosine_similarity(sparse)
    return pd.DataFrame(sim, index=user_item_matrix.index, columns=user_item_matrix.index)


def get_user_based_recommendations(user_id, user_item_matrix, user_sim_df,
                                   num_recommendations=10, k_neighbors=30, min_neighbors=5):
    """Similarity-WEIGHTED average of the k most similar users' ratings.

    ``min_neighbors``: a movie only one neighbour rated (with 5 stars) would score
    a perfect 5.0 — the top of the list would fill with obscure films tied at 5.0
    in arbitrary order. That is the sparsity problem, live. Requiring several
    neighbours to have rated a movie keeps only recommendations with evidence
    behind them; ``neighbors_who_rated`` shows how much.
    """
    similar_users = user_sim_df[user_id].sort_values(ascending=False).iloc[1:k_neighbors + 1]
    neighbor_ratings = user_item_matrix.loc[similar_users.index]
    weights = similar_users.values.reshape(-1, 1)

    # A movie a neighbor hasn't rated (NaN) contributes 0 to both the weighted
    # sum and the summed weights, so it doesn't drag the average down.
    rated_mask = neighbor_ratings.notna()
    weighted_sum = (neighbor_ratings.fillna(0) * weights).sum(axis=0)
    weight_totals = (rated_mask * weights).sum(axis=0)
    recs = pd.DataFrame({
        "score": weighted_sum / weight_totals.replace(0, np.nan),
        "neighbors_who_rated": rated_mask.sum(axis=0),
    })

    user_rated = user_item_matrix.loc[user_id].dropna().index
    recs = recs.drop(user_rated, errors="ignore")
    recs = recs[recs["neighbors_who_rated"] >= min_neighbors]
    # Best score first; among equal scores, the one more neighbours vouch for.
    return recs.sort_values(["score", "neighbors_who_rated"], ascending=False).head(num_recommendations)


def build_item_user_sparse(user_item_matrix):
    """Return ``(items×users sparse matrix, item_labels)`` for item-based CF."""
    filled = user_item_matrix.fillna(0)
    return csr_matrix(filled.values.T), list(filled.columns)


def get_item_based_recommendations(movie_title, item_user_sparse, item_labels,
                                   num_recommendations=10):
    """Movies most similar to ``movie_title`` by how users co-rated them.

    Computes one movie's similarity row on demand (no dense item×item matrix).
    """
    if movie_title not in item_labels:
        return f"Movie '{movie_title}' not found in the dataset."
    col = item_labels.index(movie_title)
    sims = cosine_similarity(item_user_sparse[col], item_user_sparse).ravel()
    order = np.argsort(sims)[::-1]
    top_idx = order[order != col][:num_recommendations]  # exclude the movie itself
    return pd.Series(sims[top_idx], index=[item_labels[i] for i in top_idx])


# ===========================================================================
# Collaborative Filtering — matrix completion (bias baseline + factorization)
# ===========================================================================
def encode_ratings(ratings_df):
    """Encode ratings as integer (user_idx, item_idx, rating) rows.

    Returns ``(encoded, user_to_idx, item_to_idx, idx_to_item)``.
    """
    user_ids = ratings_df["userId"].unique()
    item_ids = ratings_df["movieId"].unique()
    user_to_idx = {u: k for k, u in enumerate(user_ids)}
    item_to_idx = {m: k for k, m in enumerate(item_ids)}
    idx_to_item = {k: m for m, k in item_to_idx.items()}
    encoded = np.column_stack([
        ratings_df["userId"].map(user_to_idx).values,
        ratings_df["movieId"].map(item_to_idx).values,
        ratings_df["rating"].values,
    ]).astype(float)
    return encoded, user_to_idx, item_to_idx, idx_to_item


def rmse(predictions, targets) -> float:
    """Root Mean Squared Error — the standard rating-prediction accuracy metric."""
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


def fit_baseline(ratings, n_users, n_items, reg=10.0, n_iters=15):
    """Bias-corrected baseline r_hat = mu + b_u + b_i (matrix completion).

    Estimates user/item biases by alternating damped averages over the *observed*
    ratings only; ``reg`` shrinks biases of users/items with few ratings toward 0.
    Returns ``(mu, b_u, b_i)``.
    """
    users = ratings[:, 0].astype(int)
    items = ratings[:, 1].astype(int)
    vals = ratings[:, 2]

    mu = vals.mean()
    b_u = np.zeros(n_users)
    b_i = np.zeros(n_items)
    for _ in range(n_iters):
        item_resid = vals - mu - b_u[users]
        b_i = np.bincount(items, weights=item_resid, minlength=n_items) / (
            np.bincount(items, minlength=n_items) + reg)
        user_resid = vals - mu - b_i[items]
        b_u = np.bincount(users, weights=user_resid, minlength=n_users) / (
            np.bincount(users, minlength=n_users) + reg)
    return mu, b_u, b_i


def predict_baseline(mu, b_u, b_i, users, items):
    """Vectorized baseline prediction, clipped to the valid rating range."""
    users = np.asarray(users, dtype=int)
    items = np.asarray(items, dtype=int)
    return np.clip(mu + b_u[users] + b_i[items], 0.5, 5.0)


class MatrixFactorization:
    """A minimal SGD matrix-factorization recommender with user/item biases.

    Predicts r_hat(u, i) = mu + b_u + b_i + p_u . q_i, learning the biases and the
    latent factors p_u / q_i by stochastic gradient descent with L2 regularization
    over the observed ratings. Written from scratch so every step is inspectable.
    """

    def __init__(self, n_users, n_items, n_factors=20, n_epochs=20,
                 lr=0.01, reg=0.05, random_state=42):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr          # learning rate for the gradient steps
        self.reg = reg        # L2 regularization strength
        self.random_state = random_state

    def fit(self, ratings, on_epoch_end=None):
        """Train with SGD. ``on_epoch_end(epoch, train_rmse)`` streams progress."""
        rng = np.random.default_rng(self.random_state)
        self.mu = ratings[:, 2].mean()
        self.b_u = np.zeros(self.n_users)
        self.b_i = np.zeros(self.n_items)
        # Small random factors break symmetry so SGD can specialize them.
        self.p = rng.normal(0, 0.1, (self.n_users, self.n_factors))
        self.q = rng.normal(0, 0.1, (self.n_items, self.n_factors))

        users = ratings[:, 0].astype(int)
        items = ratings[:, 1].astype(int)
        vals = ratings[:, 2]
        n = len(ratings)

        for epoch in range(self.n_epochs):
            for idx in rng.permutation(n):  # shuffle each epoch
                u, i, r = users[idx], items[idx], vals[idx]
                pred = self.mu + self.b_u[u] + self.b_i[i] + self.p[u] @ self.q[i]
                err = r - pred
                self.b_u[u] += self.lr * (err - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (err - self.reg * self.b_i[i])
                p_u = self.p[u].copy()
                self.p[u] += self.lr * (err * self.q[i] - self.reg * p_u)
                self.q[i] += self.lr * (err * p_u - self.reg * self.q[i])
            if on_epoch_end is not None:
                on_epoch_end(epoch + 1, rmse(self.predict(users, items), vals))
        return self

    def predict(self, users, items, clip=True):
        """Vectorized prediction for arrays of user/item indices.

        ``clip=True`` caps to the 0.5-5 star range — right for RMSE against real
        ratings. For RANKING use ``clip=False``: a generous rater's favourites can
        all score above 5, and capping would tie them at 5.0 and lose the order.
        """
        users = np.asarray(users, dtype=int)
        items = np.asarray(items, dtype=int)
        dot = np.einsum("ij,ij->i", self.p[users], self.q[items])
        preds = self.mu + self.b_u[users] + self.b_i[items] + dot
        return np.clip(preds, 0.5, 5.0) if clip else preds


def get_mf_recommendations(user_id, model, movies_df, ratings_df,
                           user_to_idx, item_to_idx, idx_to_item, num_recommendations=10):
    """Top-N unseen movies for ``user_id``, ranked on the UNCAPPED model score.

    ``predicted_rating`` is the score capped to the star scale (what a user would
    see); ``model_score`` is what the ranking actually uses.
    """
    if user_id not in user_to_idx:
        return f"User '{user_id}' not found in the dataset."
    u = user_to_idx[user_id]
    rated = set(ratings_df[ratings_df["userId"] == user_id]["movieId"])
    candidates = [item_to_idx[m] for m in item_to_idx if m not in rated]

    scores = model.predict(np.full(len(candidates), u), np.array(candidates), clip=False)
    top = np.argsort(scores)[::-1][:num_recommendations]
    top_movie_ids = [idx_to_item[candidates[k]] for k in top]
    score_by_id = {idx_to_item[candidates[k]]: scores[k] for k in top}

    recs = movies_df[movies_df["movieId"].isin(top_movie_ids)][["movieId", "title", "genres"]].copy()
    recs["model_score"] = recs["movieId"].map(score_by_id)
    recs["predicted_rating"] = recs["model_score"].clip(0.5, 5.0)
    return recs.sort_values("model_score", ascending=False)[
        ["title", "genres", "predicted_rating", "model_score"]].reset_index(drop=True)


def baseline_recommendations(user_id, b_i, ratings_df, movies_df,
                             item_to_idx, idx_to_item, num_recommendations=10):
    """What the bias baseline would recommend: unseen movies by movie bias b_i.

    mu and b_u are the same for every movie of one user, so they cannot change
    the ORDER — every user gets the same list, minus what they have already seen.
    """
    rated = set(ratings_df.loc[ratings_df["userId"] == user_id, "movieId"])
    order = np.argsort(b_i)[::-1]
    top = [idx_to_item[k] for k in order if idx_to_item[k] not in rated][:num_recommendations]
    titles = movies_df.set_index("movieId").loc[top, "title"]
    return list(titles)


def explain_mf_prediction(model, u: int, i: int) -> dict:
    """One prediction taken apart: average + user bias + movie bias + taste match."""
    parts = {
        "average rating μ": model.mu,
        "user bias bᵤ": model.b_u[u],
        "movie bias bᵢ": model.b_i[i],
        "taste match pᵤ·qᵢ": float(model.p[u] @ model.q[i]),
    }
    parts["prediction"] = sum(parts.values())
    return parts


def rating_grid(ratings_df, movies_df, n_users=6, n_movies=5) -> pd.DataFrame:
    """A small corner of the rating matrix, with gaps — the problem in one picture.

    The most-rated movies, and users who rated some but not all of them, so the
    grid shows both known ratings and blanks for the model to fill.
    """
    popular = ratings_df["movieId"].value_counts().index[:n_movies]
    sub = ratings_df[ratings_df["movieId"].isin(popular)]
    counts = sub.groupby("userId").size()
    users = counts[(counts >= 2) & (counts <= n_movies - 2)].index[:n_users]
    grid = sub[sub["userId"].isin(users)].pivot(index="userId", columns="movieId", values="rating")
    grid = grid.reindex(index=users, columns=popular)
    grid.columns = movies_df.set_index("movieId").loc[popular, "title"]
    return grid


def complete_grid(grid, model, ratings_df, movies_df, user_to_idx, item_to_idx) -> pd.DataFrame:
    """Fill the blanks of ``rating_grid`` with the model's predictions (known ratings stay)."""
    title_to_id = movies_df.drop_duplicates("title").set_index("title")["movieId"]
    filled = grid.copy()
    for user in grid.index:
        for title in grid.columns:
            if pd.isna(grid.loc[user, title]):
                u, i = user_to_idx[user], item_to_idx[title_to_id[title]]
                filled.loc[user, title] = float(model.predict([u], [i])[0])
    return filled


# ===========================================================================
# Association Rules (Apriori + support / confidence / lift)
# ===========================================================================
def build_transactions(groceries_df: pd.DataFrame) -> pd.DataFrame:
    """Group line items into baskets and one-hot encode them.

    Each (Member_number, Date) pair is treated as one basket/receipt.
    """
    df = groceries_df.copy()
    df["Invoice"] = df["Member_number"].astype(str) + df["Date"]
    transactions = df.groupby("Invoice")["itemDescription"].apply(list).values
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    return pd.DataFrame(te_ary, columns=te.columns_)


def item_support(transaction_df: pd.DataFrame, top: int = 20) -> pd.Series:
    """Fraction of baskets containing each item, most popular first."""
    return transaction_df.mean().sort_values(ascending=False).head(top)


def run_apriori(transaction_df, min_support=0.001, max_len=3) -> pd.DataFrame:
    """Frequent itemsets via Apriori, annotated with itemset size."""
    itemsets = apriori(transaction_df, min_support=min_support,
                       use_colnames=True, max_len=max_len)
    itemsets["itemset_size"] = itemsets["itemsets"].apply(len)
    return itemsets


def make_rules(itemsets, min_confidence=0.1) -> pd.DataFrame:
    """Derive association rules and add readable "A → B" strings."""
    if len(itemsets) == 0:
        return pd.DataFrame()
    rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    if len(rules) == 0:
        return rules
    rules["antecedent_str"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["consequent_str"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
    rules["rule_str"] = rules["antecedent_str"] + " → " + rules["consequent_str"]
    return rules


def read_rule(rule: pd.Series, n_baskets: int) -> str:
    """One rule in a plain sentence — what support, confidence and lift *say*.

    The three numbers answer three business questions: how often does this
    happen at all (support → a basket count), how reliable is it (confidence),
    and is it more than coincidence (lift vs. 1).
    """
    together = round(rule["support"] * n_baskets)
    return (
        f"**Reading the top rule:** of the baskets with *{rule['antecedent_str']}*, "
        f"**{rule['confidence']:.0%}** also contain *{rule['consequent_str']}* — that is "
        f"**{rule['lift']:.1f}× more often** than in an average basket (lift). But it happens in "
        f"only {together:,} of {n_baskets:,} baskets (support), so check that a rule is "
        "frequent enough to matter before you build a promotion on it."
    )


def rule_arithmetic(rule: pd.Series, transaction_df: pd.DataFrame) -> str:
    """The three numbers of one rule, computed from basket counts in plain sight."""
    n = len(transaction_df)
    has_x = transaction_df[list(rule["antecedents"])].all(axis=1)
    has_y = transaction_df[list(rule["consequents"])].all(axis=1)
    x, y, both = int(has_x.sum()), int(has_y.sum()), int((has_x & has_y).sum())
    X, Y = rule["antecedent_str"], rule["consequent_str"]
    return (
        f"Out of **{n:,}** baskets, **{x:,}** contain *{X}*, **{y:,}** contain *{Y}*, "
        f"and **{both:,}** contain both.\n\n"
        f"- **Support** = both ÷ all = {both:,} ÷ {n:,} = **{both / n:.4f}** — how common the pair is.\n"
        f"- **Confidence** = both ÷ baskets with *{X}* = {both:,} ÷ {x:,} = **{both / x:.1%}** — "
        f"how often a basket with *{X}* also has *{Y}*.\n"
        f"- **Lift** = confidence ÷ how often *{Y}* is bought anyway = {both / x:.1%} ÷ {y / n:.1%} "
        f"= **{(both / x) / (y / n):.2f}** — above 1 means *{Y}* turns up with *{X}* more often than "
        "chance predicts."
    )


def support_reading(support: pd.Series) -> str:
    """The popularity chart in one sentence, with its top item as the example."""
    item, share = support.index[0], support.iloc[0]
    one_in = f" — about 1 in {round(1 / share)}" if share < 0.5 else ""
    return (
        f"**How to read it:** *{item}* is in **{share:.0%}** of baskets{one_in}. Items this "
        "common turn up next to everything, so popularity alone "
        "says nothing about what belongs *together*. That is what the rules below measure."
    )


def scatter_reading(rules: pd.DataFrame) -> str:
    """The support/confidence bubble chart in plain words, with real examples."""
    strong = rules.nlargest(1, "lift").iloc[0]
    sure = rules.nlargest(1, "confidence").iloc[0]
    text = (
        "**How to read it:** every bubble is one rule. *Right* = the pair is common, "
        "*up* = the rule is reliable, *big and yellow* = strong lift. The best rules for a "
        "promotion sit up and to the right, and are big.\n\n"
        f"- Biggest bubble: *{strong['rule_str']}* — lift **{strong['lift']:.1f}**, "
        f"in {strong['support']:.2%} of baskets."
    )
    if sure["rule_str"] != strong["rule_str"]:
        text += (f"\n- Highest bubble: *{sure['rule_str']}* — {sure['confidence']:.0%} confidence, "
                 f"lift {sure['lift']:.1f}. Reliable, but check its lift: high confidence can come "
                 "from the right-hand item simply being popular.")
    return text


def rule_edges(rules: pd.DataFrame, top_n: int = 10) -> list[tuple[str, str, float]]:
    """``(from item, to item, lift)`` for the strongest rules — the network's lines."""
    top = rules.nlargest(top_n, "lift")
    return [(a, c, row["lift"]) for _, row in top.iterrows()
            for a in row["antecedents"] for c in row["consequents"]]


def network_reading(edges: list[tuple[str, str, float]]) -> str:
    """The rule network in one sentence: its best-connected item as the example."""
    links: dict[str, set[str]] = {}
    for a, c, _ in edges:
        links.setdefault(a, set()).add(c)
        links.setdefault(c, set()).add(a)
    hub = max(links, key=lambda k: len(links[k]))
    partners = ", ".join(f"*{p}*" for p in sorted(links[hub]))
    return (
        "**How to read it:** each dot is a product; a line means one of the strongest rules "
        f"connects them. Example: *{hub}* is linked to {partners}. A cluster like this can hint "
        "at a *shopping mission* (candidates for one shelf or bundle) — check each rule's lift "
        "and support in the table before acting on it."
    )


def get_basket_recommendations(item, rules, top_n=5):
    """Recommend items frequently bought together with ``item``.

    Uses only rules whose antecedent is EXACTLY ``item`` and whose lift is > 1:
    in {whole milk, baby food} -> {baby cosmetics} milk is a bystander, and a
    rule with lift <= 1 is no association at all (or a negative one). Ranked by
    lift (strength), then confidence (reliability).
    """
    if rules is None or len(rules) == 0:
        return "No association rules available — generate rules first."
    single = rules["antecedents"].apply(lambda a: a == frozenset({item}))
    # Suggest single products: a consequent like {baby food, rolls/buns} just drags a
    # staple along with the real partner.
    one_item = rules["consequents"].apply(len) == 1
    matches = rules[single & one_item & (rules["lift"] > 1)].copy()
    if matches.empty:
        return (f"No positive association (lift > 1) for '{item}' — it is bought about as "
                "often with everything as chance predicts. Popular is not the same as related.")
    matches["recommendation"] = matches["consequents"].apply(lambda c: ", ".join(sorted(c)))
    matches = matches.sort_values(["lift", "confidence"], ascending=False)
    matches = matches.drop_duplicates(subset="recommendation")
    return matches[["recommendation", "support", "confidence", "lift"]].head(top_n).reset_index(drop=True)


# ===========================================================================
# Overview diagram
# ===========================================================================
def taxonomy_dot() -> str:
    """Graphviz DOT of the recommender-method taxonomy (rendered client-side).

    Mirrors ``teaching.architecture_dot`` — a raw DOT string, so the page needs no
    Graphviz Python package.
    """
    return """
digraph recommenders {
    rankdir=LR;
    bgcolor="transparent";
    node [fontname="Helvetica", shape=box, style="rounded,filled", fillcolor="#ffffff"];
    edge [fontname="Helvetica", fontsize=10];

    root [label="Recommendation\\nmethods", fillcolor="#e3f2fd"];

    cbf  [label="Content-Based\\nFiltering\\n(genres → TF-IDF)", fillcolor="#c8e6c9"];
    cf   [label="Collaborative\\nFiltering\\n(user ratings)", fillcolor="#c8e6c9"];
    ar   [label="Association\\nRules\\n(market baskets)", fillcolor="#c8e6c9"];

    nb   [label="Neighborhood\\n(user / item-based)"];
    mb   [label="Model-based\\n(bias + matrix\\nfactorization)"];

    root -> cbf;
    root -> cf;
    root -> ar;
    cf -> nb;
    cf -> mb;
}
""".strip()
