# Movie Recommender System 

import pandas as pd
import numpy as np
import requests
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# TMDb API

API_KEY = "81f5580c4bb6c74b539d051282b52ba5"
BASE_URL = "https://api.themoviedb.org/3"
IMG_BASE = "https://image.tmdb.org/t/p/w500"

def get_movie_poster(title):
    url = f"{BASE_URL}/search/movie"
    params = {"api_key": API_KEY, "query": title}
    try:
        response = requests.get(url, params=params).json()
        if response.get("results"):
            poster_path = response["results"][0].get("poster_path")
            if poster_path:
                return IMG_BASE + poster_path
    except:
        pass
    return None

# Load Data (cached)

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


# Precompute Matrices (cached)

@st.cache_resource
def precompute_models():
    matrix = ratings.pivot(index="user_id", columns="item_id", values="rating").fillna(0)
    # User similarity
    user_sim = cosine_similarity(matrix)
    user_sim = pd.DataFrame(user_sim, index=matrix.index, columns=matrix.index)
    # Item similarity
    item_sim = cosine_similarity(matrix.T)
    item_sim = pd.DataFrame(item_sim, index=matrix.columns, columns=matrix.columns)
    # SVD
    svd = TruncatedSVD(n_components=40, random_state=42)
    U = svd.fit_transform(matrix)
    VT = svd.components_
    preds = np.dot(U, VT)
    preds_df = pd.DataFrame(np.clip(preds, 0, 5), index=matrix.index, columns=matrix.columns)
    return matrix, user_sim, item_sim, preds_df

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
    rated = user_item_matrix.loc[user_id]
    unrated = rated[rated == 0].index
    preds = svd_preds.loc[user_id, unrated].sort_values(ascending=False)[:n_recs]
    return preds.index


# Evaluation Function (Precision@K)

def evaluate_precision(user_id, recs, k=5):
    relevant = ratings[(ratings['user_id'] == user_id) & (ratings['rating'] >= 4)]['item_id'].values
    if len(relevant) == 0:
        return None
    hits = len(set(recs).intersection(set(relevant)))
    return hits / k


# Streamlit UI

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender System ")

# Sidebar
with st.sidebar:
    st.header("Settings")
    user_id = st.number_input("Enter User ID (1–943):", min_value=1, max_value=943, value=1)
    top_n = st.slider("Number of Recommendations:", min_value=3, max_value=15, value=5)
    method = st.selectbox("Choose Method:", ["User-based CF", "Item-based CF", "SVD"])
    evaluate = st.checkbox("Evaluate Model (Precision@K)")

# Button 
if st.button("Recommend Movies"):
    if method == "User-based CF":
        recs = recommend_usercf(user_id, n_recs=top_n)
    elif method == "Item-based CF":
        recs = recommend_itemcf(user_id, n_recs=top_n)
    else:
        recs = recommend_svd(user_id, n_recs=top_n)

    st.subheader(f"Top {top_n} Recommendations for User {user_id}")

    #  Grid
    n_cols = 4  
    rows = (len(recs) + n_cols - 1) // n_cols  # عدد الصفوف
    for r in range(rows):
        cols = st.columns(n_cols)
        for c in range(n_cols):
            idx = r * n_cols + c
            if idx >= len(recs):
                break
            movie_id = recs[idx]
            movie_title = movies[movies["item_id"] == movie_id]["title"].values[0]
            poster = get_movie_poster(movie_title)
            with cols[c]:
                if poster:
                    st.image(poster, caption=movie_title, width=150)
                else:
                    st.write(movie_title)

    # Evaluation
    if evaluate:
        precision = evaluate_precision(user_id, recs, k=top_n)
        if precision is not None:
            st.write(f"✅ Precision@{top_n}: {precision:.2f}")
        else:
            st.write("⚠️ No relevant movies found for evaluation.")
