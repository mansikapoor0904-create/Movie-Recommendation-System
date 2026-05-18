import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("movies.csv")

# Remove missing values
movies = movies.dropna()

# Convert genres into numerical values
vectorizer = TfidfVectorizer(stop_words='english')

tfidf_matrix = vectorizer.fit_transform(movies['genres'])

# Calculate similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create index mapping
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Recommendation function
def recommend_movies(title):

    idx = indices[title]

    similarity_scores = list(enumerate(cosine_sim[idx]))

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similarity_scores = similarity_scores[1:6]

    movie_indices = [i[0] for i in similarity_scores]

    return movies['title'].iloc[movie_indices]

# Streamlit UI
st.title("Movie Recommendation System")

movie_list = movies['title'].values

selected_movie = st.selectbox(
    "Select a Movie",
    movie_list
)

if st.button("Recommend"):

    recommendations = recommend_movies(selected_movie)

    st.write("Top 5 Recommended Movies:")

    for movie in recommendations:
        st.write(movie)