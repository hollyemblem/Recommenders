import os
import pandas as pd
from builder import matrix_factorization_calculator as mf

#all_ratings = mf.load_all_ratings("data/movies.csv")     

movies_df = pd.read_csv("data/movies.csv")
ratings_df = pd.read_csv("data/ratings.csv")

combined_ratings_df = pd.merge(ratings_df, movies_df, on='movieId', how='inner')

combined_ratings_reduced_df = combined_ratings_df[['userId', 'movieId', 'rating', 'genres','timestamp']]
print(combined_ratings_reduced_df.head())
#columns = ['user_id', 'movie_id', 'rating', 'type', 'rating_timestamp']

all_ratings = mf.load_all_ratings(combined_ratings_reduced_df)
print(all_ratings.head())

#To do  - inner join the pandas datasets