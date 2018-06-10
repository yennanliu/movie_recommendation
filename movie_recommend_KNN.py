
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


def fix_extreme_value(column):
    print (column.max())
    fixed_column = [x if x < 100 else column.mean()  for x in column] 
    return  fixed_column


def get_user_movie_metrix(df):
	cols = ['userId', 'movieId', 'rating']
	df_ = df[cols]
	df_ratings_grouped = df_.groupby(['userId', 'movieId']).sum()
	df_ratings_pivot = pd.pivot_table(df_ratings_grouped,columns=['movieId'],index=['userId'],aggfunc=np.sum)
	
	#### approach 1 : fill n/a data with 0  ####
	#df_ratings_pivot_ = df_ratings_pivot.fillna(0)
	#df_ratings_pivot_.columns= list(set(df_.movieId))
	#df_ratings_pivot_.reset_index()

	#### approach 2 : fill n/a data with mean  ####
	for index, row in df_ratings_pivot.iterrows():
		row_mean = df_ratings_pivot.iloc[index-1].mean()
		#print (row_mean)
		df_ratings_pivot.iloc[index-1].fillna(row_mean, inplace=True)

	df_ratings_pivot_ = df_ratings_pivot.copy()  

	#### approach 3 : work with "timestamp" ####
	# dev 
	
	X = df_ratings_pivot_.iloc[:,1:].fillna(0)
	df_ratings_pivot_std = df_ratings_pivot_.copy()
	# standardize 
	for i in X:
		df_ratings_pivot_std[i] = preprocessing.scale(df_ratings_pivot_std[i])
	# pca : modify dimension from N  ->  2 
	# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
	# need to find tuned pca dimension below 

	pca = PCA(n_components=2)
	pca.fit(df_ratings_pivot_std)
	ratings_pivot_std_pca =  pca.fit_transform(df_ratings_pivot_std)
	# fix extreme value (for clustering)
	ratings_pivot_std_pca_ = pd.DataFrame(ratings_pivot_std_pca)
	# set userID as index for output clustering group outcome
	ratings_pivot_std_pca_['userId'] = ratings_pivot_std_pca_.index+1
	ratings_pivot_std_pca_ = ratings_pivot_std_pca_.set_index('userId')
	ratings_pivot_std_pca_[0] = fix_extreme_value(ratings_pivot_std_pca_[0])
	ratings_pivot_std_pca_[1] = fix_extreme_value(ratings_pivot_std_pca_[1])
	print (ratings_pivot_std_pca_.max())
	
	#print (df_ratings_pivot_)
	#print (df_ratings_pivot_std)
	return df_ratings_pivot_, df_ratings_pivot_std, ratings_pivot_std_pca_


# -------------------------------------
# model 



def KNN_model(user_movie_metrix,df_ratings_pivot):
	# kmeans clustering 
	# can tune the KNN super-parameter below 
	kmean = cluster.KMeans(n_clusters=10, max_iter=300, random_state=4000)
	kmean.fit(user_movie_metrix)
	# add lebel to user table 
	df_ratings_pivot['group'] = kmean.labels_
	#df_train['group'] = kmean.labels_ 
	print ('*'*10)
	print ('*** Cluster output : ')
	print (df_ratings_pivot)
	print ('*** User group  : ')
	print (df_ratings_pivot.group.value_counts())
	print ('*'*10)
	return df_ratings_pivot





# -------------------------------------


if __name__ == '__main__':
	df_ratings = get_data()
	# get user-movie matrix
	df_ratings_pivot_, df_ratings_pivot_std, ratings_pivot_std_pca_ = get_user_movie_metrix(df_ratings)
	# KNN modeling  
	df_ratings_pivot_group = KNN_model(ratings_pivot_std_pca_,df_ratings_pivot_)
	# top 10 movies in all group (mean rating)
	for group_ in list(set(df_ratings_pivot_group.group)):
	    print ('---------- recommend movie : ----------')
	    print ('group = ', group_)
	    print ('movie id ,  rating')
	    print (df_ratings_pivot_group[df_ratings_pivot_group.group==group_]\
	                          .iloc[:,:-1]\
	                          .mean(axis=0)\
	                          .sort_values(ascending=False)\
	                          .head(10))

	"""

	### todo : filter outler / refine df_ratings_pivot_std (user-movie matrix)
	
	"""
                     






