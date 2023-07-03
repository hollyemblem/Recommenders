import os
import pandas as pd
from builder import matrix_factorization_calculator as mf
from builder import item_similarity_calculator as isc 
 
##Dataset preparation
movies_df = pd.read_csv("data/movies.csv")
ratings_df = pd.read_csv("data/ratings.csv")
combined_ratings_df = pd.merge(ratings_df, movies_df, on='movieId', how='inner')
combined_ratings_reduced_df = combined_ratings_df[['userId', 'movieId', 'rating', 'genres','timestamp']]

#Obtaining ratings

all_ratings = mf.load_all_ratings(combined_ratings_reduced_df)

cor, movies = isc.ItemSimilarityMatrixBuilder.build(all_ratings, save=False)
print(cor)