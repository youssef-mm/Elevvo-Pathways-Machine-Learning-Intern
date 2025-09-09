🎬 Movie Recommender System (MRS)

This project implements a Movie Recommender System using the MovieLens 100k dataset
.
It applies Collaborative Filtering (User-based & Item-based) and Matrix Factorization (SVD) to recommend movies for users.
The project also includes an interactive Streamlit web app with movie posters fetched from the TMDb API.

🚀 Features

User-based Collaborative Filtering (CF)

Item-based Collaborative Filtering (CF)

Matrix Factorization (SVD) with TruncatedSVD

Evaluation with Precision@K

Movie posters & metadata via TMDb API

Interactive UI with Streamlit

🛠️ Tech Stack

Python (Pandas, NumPy, Scikit-learn)

Streamlit (Web App)

TMDb API (Movie Posters)

📂 Dataset

The project uses the MovieLens 100k dataset:

u.data: Ratings (user_id, item_id, rating, timestamp)

u.item: Movie metadata (titles, genres, release dates, etc.)

⚡ Usage
1. Clone repo & install dependencies
git clone https://github.com/yourusername/movie-recommender.git
cd movie-recommender
pip install -r requirements.txt

2. Run Jupyter Examples

The file MRS.py contains the class for recommendation. Example usage:

columns = ['user_id', 'item_id', 'rating', 'timestamp']
data = pd.read_csv('u.data', sep='\t', names=columns)

mrs = MRS(data, user_column='user_id', movie_column='item_id', rating_column='rating')
print(mrs.recommend(1))   # Basic recommendations

3. Run Streamlit App
streamlit run app.py

📊 Example

User-based Precision@10 ≈ 0.26

Item-based Precision@10 ≈ 0.23

SVD Precision@5 ≈ 0.53

🌟 Demo UI

Choose User ID & method (User-based CF, Item-based CF, or SVD).

View recommended movies with titles + posters.

Evaluate with Precision@K.

🔑 API Key Setup

To display posters, you need a TMDb API key
.
Replace inside app.py:

API_KEY = "YOUR_TMDB_API_KEY"

📌 Future Work

Add hybrid recommendation (content + CF)

Improve evaluation metrics (Recall, F1, RMSE)

Deploy online (Streamlit Cloud / Hugging Face Spaces)