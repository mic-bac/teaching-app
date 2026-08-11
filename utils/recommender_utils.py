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


def load_groceries(data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load the Groceries market-basket transactions."""
    return pd.read_csv(Path(data_dir) / "groceries" / "Groceries_dataset.csv")


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
                                   num_recommendations=10, k_neighbors=10):
    """Similarity-WEIGHTED average of the k most similar users' ratings."""
    similar_users = user_sim_df[user_id].sort_values(ascending=False).iloc[1:k_neighbors + 1]
    neighbor_ratings = user_item_matrix.loc[similar_users.index]
    weights = similar_users.values.reshape(-1, 1)

    # A movie a neighbor hasn't rated (NaN) contributes 0 to both the weighted
    # sum and the summed weights, so it doesn't drag the average down.
    rated_mask = neighbor_ratings.notna()
    weighted_sum = (neighbor_ratings.fillna(0) * weights).sum(axis=0)
    weight_totals = (rated_mask * weights).sum(axis=0)
    scores = weighted_sum / weight_totals.replace(0, np.nan)

    user_rated = user_item_matrix.loc[user_id].dropna().index
    scores = scores.drop(user_rated, errors="ignore")
    return scores.nlargest(num_recommendations)


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

    def predict(self, users, items):
        """Vectorized prediction for arrays of user/item indices."""
        users = np.asarray(users, dtype=int)
        items = np.asarray(items, dtype=int)
        dot = np.einsum("ij,ij->i", self.p[users], self.q[items])
        preds = self.mu + self.b_u[users] + self.b_i[items] + dot
        return np.clip(preds, 0.5, 5.0)


def get_mf_recommendations(user_id, model, movies_df, ratings_df,
                           user_to_idx, item_to_idx, idx_to_item, num_recommendations=10):
    """Top-N unseen movies for ``user_id`` by predicted rating."""
    if user_id not in user_to_idx:
        return f"User '{user_id}' not found in the dataset."
    u = user_to_idx[user_id]
    rated = set(ratings_df[ratings_df["userId"] == user_id]["movieId"])
    candidates = [item_to_idx[m] for m in item_to_idx if m not in rated]

    preds = model.predict(np.full(len(candidates), u), np.array(candidates))
    top = np.argsort(preds)[::-1][:num_recommendations]
    top_movie_ids = [idx_to_item[candidates[k]] for k in top]
    pred_by_id = {idx_to_item[candidates[k]]: preds[k] for k in top}

    recs = movies_df[movies_df["movieId"].isin(top_movie_ids)][["movieId", "title", "genres"]].copy()
    recs["predicted_rating"] = recs["movieId"].map(pred_by_id)
    return recs.sort_values("predicted_rating", ascending=False)[
        ["title", "genres", "predicted_rating"]].reset_index(drop=True)


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


def get_basket_recommendations(item, rules, top_n=5):
    """Recommend items frequently bought together with ``item``.

    Looks up rules whose antecedent contains the item and returns the consequents,
    ranked by lift (strength) then confidence (reliability).
    """
    if rules is None or len(rules) == 0:
        return "No association rules available — generate rules first."
    matches = rules[rules["antecedents"].apply(lambda a: item in a)].copy()
    if matches.empty:
        return f"No rules found with '{item}' on the antecedent side."
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
