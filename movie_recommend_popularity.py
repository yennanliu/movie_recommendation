# python 3 


import numpy as np
import pandas as pd 
import time
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib


# -------------------------------------

# help function 
# data preprocess 
def get_data():
    route='datasets/ml-latest-small/'
    #df_movie = pd.read_csv(route +'movies.csv')
    df_ratings = pd.read_csv(route +'ratings.csv')
    return df_ratings

def data_preprocess(df):
    df['view_count']=df.groupby(['userId','movieId']).count().reset_index()['rating']
    movie_grouped = df.groupby(['movieId']).agg({'view_count': 'count'}).reset_index()
    grouped_sum = movie_grouped['view_count'].sum()
    movie_grouped['percentage']  = movie_grouped['view_count'].div(grouped_sum)*100
    movie_grouped.sort_values(['view_count', 'movieId'], ascending = [0,1])
    return movie_grouped




# -------------------------------------

# ops 





if __name__ == '__main__':
	df_ratings = get_data()
	movie_grouped = data_preprocess(df_ratings)
	print (movie_grouped)








