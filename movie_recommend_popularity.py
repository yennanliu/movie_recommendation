# python 3 


# ref 
# https://github.com/llSourcell/recommender_live/blob/master/Song%20Recommender_Python.ipynb
# https://towardsdatascience.com/how-to-build-a-simple-song-recommender-296fcbc8c85



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


def get_train_test_data(df):
	train_data, test_data = train_test_split(df, test_size = 0.20, random_state=0)
	return train_data, test_data



# -------------------------------------

# ops 



# ML 



if __name__ == '__main__':
	df_ratings = get_data()
	movie_grouped = data_preprocess(df_ratings)
	print (movie_grouped)
	train_data, test_data = get_train_test_data(df_ratings)








