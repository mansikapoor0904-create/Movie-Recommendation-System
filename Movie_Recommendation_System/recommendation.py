#Import Libraries

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Load Dataset

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

print(movies.head())
print(ratings.head())

#Data Cleaning

#Check Missing Values

print(movies.isnull().sum())
print(ratings.isnull().sum())

#Remove Missing Values

movies = movies.dropna()
ratings = ratings.dropna()

#Convert Genres into Numerical Data

#TF-IDF Vectorization

vectorizer = TfidfVectorizer(stop_words='english')

tfidf_matrix = vectorizer.fit_transform(movies['genres'])

#Calculate Similarity

#Cosine Similarity

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

#Create Movie Index

indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

#Create Recommendation Function

def recommend_movies(title, cosine_sim=cosine_sim):

    idx = indices[title]

    similarity_scores = list(enumerate(cosine_sim[idx]))

    similarity_scores = sorted(similarity_scores,
                               key=lambda x: x[1],
                               reverse=True)

    similarity_scores = similarity_scores[1:6]

    movie_indices = [i[0] for i in similarity_scores]

    return movies['title'].iloc[movie_indices]

#Test Recommendation System

movie_name = input("Enter Movie Name: ")

print("\nRecommended Movies:\n")

print(recommend_movies(movie_name))

#Create User-Movie Matrix

user_movie_matrix = ratings.pivot_table(index='userId',
                                        columns='movieId',
                                        values='rating')

#Fill Missing Values

user_movie_matrix = user_movie_matrix.fillna(0)

#Calculate Similarity Between Users

from sklearn.metrics.pairwise import cosine_similarity

user_similarity = cosine_similarity(user_movie_matrix)

#Streamlit Code

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("movies.csv")

movies = movies.dropna()

vectorizer = TfidfVectorizer(stop_words='english')

tfidf_matrix = vectorizer.fit_transform(movies['genres'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()


def recommend_movies(title):
    idx = indices[title]

    similarity_scores = list(enumerate(cosine_sim[idx]))

    similarity_scores = sorted(similarity_scores,
                               key=lambda x: x[1],
                               reverse=True)

    similarity_scores = similarity_scores[1:6]

    movie_indices = [i[0] for i in similarity_scores]

    return movies['title'].iloc[movie_indices]

st.title("Movie Recommendation System")

movie_list = movies['title'].values

selected_movie = st.selectbox("Select a Movie", movie_list)

if st.button("Recommend"):

    recommendations = recommend_movies(selected_movie)

    st.write("Top 5 Recommended Movies:")

    for movie in recommendations:
        st.write(movie)

        
