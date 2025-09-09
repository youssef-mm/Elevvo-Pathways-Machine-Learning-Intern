# Movie Recommender System with Selectable Metrics

import pandas as pd
import numpy as np
import requests
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
import math


# TMDb API for Posters

API_KEY = "81f5580c4bb6c74b539d051282b52ba5"
BASE_URL = "https://api.themoviedb.org/3"
IMG_BASE = "https://image.tmdb.org/t/p/w500"

def get_movie_poster(title):
    try:
        response = requests.get(f"{BASE_URL}/search/movie", params={"api_key": API_KEY, "query": title}).json()
        if response.get("results"):
            poster_path = response["results"][0].get("poster_path")
            if poster_path:
                return IMG_BASE + poster_path
    except:
        pass
    return None


# Load Data

@st.cache_data
def load_data():
    columns = ["user_id", "item_id", "rating", "timestamp"]
    data = pd.read_csv("u.data", sep="\t", names=columns)
    item_cols = ["item_id", "title", "release_date", "video_release_date", "IMDb_URL",
                 "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
                 "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
                 "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    items = pd.read_csv("u.item", sep="|", names=item_cols, encoding="latin-1")
    return data, items

ratings, movies = load_data()


# Precompute Matrices

@st.cache_resource
def precompute_models():
    matrix = ratings.pivot(index="user_id", columns="item_id", values="rating").fillna(0)
    user_sim = pd.DataFrame(cosine_similarity(matrix), index=matrix.index, columns=matrix.index)
    item_sim = pd.DataFrame(cosine_similarity(matrix.T), index=matrix.columns, columns=matrix.columns)
    svd = TruncatedSVD(n_components=40, random_state=42)
    U = svd.fit_transform(matrix)
    VT = svd.components_
    svd_preds = pd.DataFrame(np.clip(np.dot(U, VT), 0, 5), index=matrix.index, columns=matrix.columns)
    return matrix, user_sim, item_sim, svd_preds

user_item_matrix, user_sim, item_sim, svd_preds = precompute_models()


# Recommendation Functions

def recommend_usercf(user_id, n_recs=5):
    sims = user_sim[user_id].drop(user_id).sort_values(ascending=False)
    top_users = sims.index[:5]
    scores = user_item_matrix.loc[top_users].mean().sort_values(ascending=False)
    scores = scores[user_item_matrix.loc[user_id] == 0]
    return scores.index[:n_recs]

def recommend_itemcf(user_id, n_recs=5):
    scores = {}
    user_ratings = user_item_matrix.loc[user_id]
    for movie, rating in user_ratings.items():
        if rating > 0:
            sims = item_sim[movie].drop(movie)
            for m, s in sims.items():
                if user_ratings[m] == 0:
                    scores[m] = scores.get(m, 0) + s * rating
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [m for m, _ in ranked[:n_recs]]

def recommend_svd(user_id, n_recs=5):
    unrated = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id]==0].index
    preds = svd_preds.loc[user_id, unrated].sort_values(ascending=False)[:n_recs]
    return preds.index


# Evaluation Functions

def evaluate_precision(user_id, recs, k=5):
    relevant = ratings[(ratings['user_id']==user_id) & (ratings['rating']>=4)]['item_id'].values
    if len(relevant) == 0: return None
    hits = len(set(recs).intersection(set(relevant)))
    return hits / k

def evaluate_recall(user_id, recs, k=5):
    relevant = ratings[(ratings['user_id']==user_id) & (ratings['rating']>=4)]['item_id'].values
    if len(relevant) == 0: return None
    hits = len(set(recs).intersection(set(relevant)))
    return hits / len(relevant)

def evaluate_f1(user_id, recs, k=5):
    precision = evaluate_precision(user_id, recs, k) or 0
    recall = evaluate_recall(user_id, recs, k) or 0
    if precision + recall == 0: return 0
    return 2 * (precision * recall) / (precision + recall)

def evaluate_map(user_id, recs, k=5):
    relevant = ratings[(ratings['user_id']==user_id) & (ratings['rating']>=4)]['item_id'].values
    if len(relevant) == 0: return None
    hits, sum_precisions = 0, 0
    for i, rec in enumerate(recs, start=1):
        if rec in relevant:
            hits += 1
            sum_precisions += hits / i
    return sum_precisions / min(len(relevant), k)


# Streamlit UI

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender System")

# Sidebar
with st.sidebar:
    st.header("Settings")
    user_id = st.number_input("Enter User ID:", min_value=int(ratings['user_id'].min()),
                              max_value=int(ratings['user_id'].max()), value=1)
    top_n = st.slider("Number of Recommendations:", 3, 15, 5)
    method = st.selectbox("Choose Method:", ["User-based CF", "Item-based CF", "SVD"])
    st.subheader("Select Evaluation Metrics")
    metrics = st.multiselect("Metrics:", ["Precision", "Recall", "F1", "MAP", "RMSE"], default=["Precision"])

# Recommend Button
if st.button("Recommend Movies"):
    if method == "User-based CF":
        recs = recommend_usercf(user_id, n_recs=top_n)
    elif method == "Item-based CF":
        recs = recommend_itemcf(user_id, n_recs=top_n)
    else:
        recs = recommend_svd(user_id, n_recs=top_n)

    st.subheader(f"Top {top_n} Recommendations for User {user_id}")

    # Display Grid
    n_cols = 4
    rows = (len(recs)+n_cols-1)//n_cols
    for r in range(rows):
        cols = st.columns(n_cols)
        for c in range(n_cols):
            idx = r*n_cols + c
            if idx >= len(recs): break
            movie_id = recs[idx]
            movie_title = movies[movies["item_id"]==movie_id]["title"].values[0]
            poster = get_movie_poster(movie_title)
            with cols[c]:
                if poster:
                    st.image(poster, caption=movie_title, width=150)
                else:
                    st.write(movie_title)

    # Display Selected Metrics
    st.subheader("📊 Evaluation Metrics")
    if "Precision" in metrics:
        precision = evaluate_precision(user_id, recs, top_n)
        st.write(f"✅ Precision@{top_n}: {precision:.2f}" if precision is not None else "⚠️ Not available")
    if "Recall" in metrics:
        recall = evaluate_recall(user_id, recs, top_n)
        st.write(f"🔄 Recall@{top_n}: {recall:.2f}" if recall is not None else "⚠️ Not available")
    if "F1" in metrics:
        f1 = evaluate_f1(user_id, recs, top_n)
        st.write(f"🎯 F1@{top_n}: {f1:.2f}")
    if "MAP" in metrics:
        map_score = evaluate_map(user_id, recs, top_n)
        st.write(f"📌 MAP@{top_n}: {map_score:.2f}" if map_score is not None else "⚠️ Not available")
