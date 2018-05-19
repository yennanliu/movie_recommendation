
# python 3 

# ops
import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np
import time
from sklearn.externals import joblib


# ml 
from sklearn import preprocessing
from sklearn import cluster, tree
from sklearn.decomposition import PCA



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




def get_user_movie_metrix(df):
	cols = ['userId', 'movieId', 'rating']
	df_ = df[cols]
	df_ratings_grouped = df_.groupby(['userId', 'movieId']).sum()
	df_ratings_pivot = pd.pivot_table(df_ratings_grouped,columns=['movieId'],index=['userId'],aggfunc=np.sum)
	# approach 1 : fill n/a data with 0 
	df_ratings_pivot_ = df_ratings_pivot.fillna(0)
	df_ratings_pivot_.columns= list(set(df_.movieId))
	df_ratings_pivot_.reset_index()
	# approach 2 : fill n/a data with mean 
	#df_ratings_pivot_ = df_ratings_pivot.fillna(0)
	
	X = df_ratings_pivot_.iloc[:,1:].fillna(0)
	df_ratings_pivot_std = df_ratings_pivot_.copy()
	# standardize 
	for i in X:
		df_ratings_pivot_std[i] = preprocessing.scale(df_ratings_pivot_std[i])
	# pca : modify dimension form N  ->  2 
	# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
	pca = PCA(n_components=2)
	pca.fit(df_ratings_pivot_std)
	print (pca.fit_transform(df_ratings_pivot_std))
	
	print (df_ratings_pivot_)
	print (df_ratings_pivot_std)
	return df_ratings_pivot_,df_ratings_pivot_std


# -------------------------------------
# model 






# -------------------------------------


if __name__ == '__main__':
	df_ratings = get_data()
	# get user-movie matrix
	df_user_movie_matrix = get_user_movie_metrix(df_ratings)






