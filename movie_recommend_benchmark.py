# python 3 
import pandas as pd 
from sklearn.cross_validation import train_test_split
import numpy as np
import time
from sklearn.externals import joblib

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

def get_benchmark_feature(df):
	# total view 
	view_count_total = df.groupby(['movieId']).count().reset_index()[['movieId','rating']]
	view_count_total.columns = ['movieId','total_view']
	# avg rating 
	df_avg_rating = df.groupby(['movieId']).mean().reset_index()[['movieId','rating']]
	df_avg_rating.columns = ['movieId','avg_rating']
	# merge 
	df_ratings_viewcount = pd.merge(df_ratings, view_count_total, on='movieId')
	df_ratings_viewcount = pd.merge(df_ratings_viewcount, df_avg_rating, on='movieId')
	return df_ratings_viewcount

def get_rating_deviation(df):
    df_ = df.copy()
    userlist = list(set(df.userId))
    rating_divation = []
    for user in userlist:
        df_user = df[df.userId == user]
        #avg_user_rating = df_user.rating.mean()
        std_user_rating = df_user.rating.std()
        rating_divation.append(std_user_rating)
    df_std_rating = pd.DataFrame({'userId':userlist, 'std_rating': rating_divation })
    print (df_std_rating.head(10))
    #return df_std_rating
    # merge back 
    df_ = pd.merge(df_, df_std_rating, on='userId')
    return df_

# model 
def recommend(df_feature):
	# get movie with top view counts and avg_rating 
	recommend_list = df_ratings_viewcount.groupby('movieId')\
                                     .mean()[['total_view','avg_rating']]\
                                     .sort_values(['total_view','avg_rating'],ascending=False)\
                                     .reset_index()
	# return top 20 movie as recommendation 
	recommend_list_ = recommend_list.head(20) 
	print ('recommend list  : ')
	print (recommend_list_)
	return recommend_list_

def recommend_by_user(df_feature):
	pass 

if __name__ == '__main__':
	df_ratings = get_data()
	# get rating deviation 
	df_std_rating = get_rating_deviation(df_ratings)
	print (df_std_rating)

	# get movie rating 
	df_ratings_viewcount = get_benchmark_feature(df_ratings)
	print (df_ratings_viewcount)
	recommend_movie = recommend(df_ratings_viewcount)
	#movie_grouped = data_preprocess(df_ratings)
	#print (movie_grouped)
	#train_data, test_data = get_train_test_data(df_ratings)
