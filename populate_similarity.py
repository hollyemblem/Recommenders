import os
import pandas as pd
from builder import matrix_factorization_calculator as mf

#all_ratings = mf.load_all_ratings("data/movies.csv")     

df = pd.read_csv("data/movies.csv")
print(df.head())

#To do  - inner join the pandas datasets