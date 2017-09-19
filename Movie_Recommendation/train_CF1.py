# python 3 

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from feature import *

#===============================================
# help function 


def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
                                        size=10, 
                                        replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]
        
    # Test and training are truly disjoint
    assert(np.all((train * test) == 0)) 
    return train, test



def fast_similarity(ratings, kind='user', epsilon=1e-9):
    # epsilon -> small number for handling dived-by-zero errors
    if kind == 'user':
        sim = ratings.dot(ratings.T) + epsilon
    elif kind == 'item':
        sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

    
def predict_fast_simple(ratings, similarity, kind='user'):
    if kind == 'user':
        return similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif kind == 'item':
        return ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])



def predict_topk(ratings, similarity, kind='user', k=40):
    pred = np.zeros(ratings.shape)
    if kind == 'user':
        for i in range(ratings.shape[0]):
            top_k_users = [np.argsort(similarity[:,i])[:-k-1:-1]]
            for j in range(ratings.shape[1]):
                pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users]) 
                pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
    if kind == 'item':
        for j in range(ratings.shape[1]):
            top_k_items = [np.argsort(similarity[:,j])[:-k-1:-1]]
            for i in range(ratings.shape[0]):
                pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T) 
                pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items]))        
    
    return pred



def get_mse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)


def top_k_movies(similarity, mapper, movie_idx, k=10):
    return [mapper[x] for x in np.argsort(similarity[movie_idx,:])[:-k-1:-1]]
    

#===============================================


# load data 
def main(idx):
	# given movie name, predict other simiar movie names 
	users, ratings, items, ratings_base, ratings_test = load_data()
	ratings = ratings.drop('Unnamed: 0', 1)
	ratings.columns = ['user_id', 'item_id', 'rating', 'timestamp']
	n_users = ratings.user_id.unique().shape[0]
	n_items = ratings.item_id.unique().shape[0]
	ratings_ = np.zeros((n_users, n_items))
	for row in ratings.itertuples():
	    ### dispense "rating" from every user to every movie to matrix 
	    ### row[3] : rating 
	    ### ratings_[row[1]-1, row[2]-1] : position in matrix 
	    ### i.e. a[1,2] ->  element in matrix a in row 1, column 2 
	    ratings_[row[1]-1, row[2]-1] = row[3]
	# split train, test set 
	train, test = train_test_split(ratings_)
	# get similarity
	user_similarity = fast_similarity(train, kind='user')
	item_similarity = fast_similarity(train, kind='item')
	# predict 
	predict_fast_simple(train, user_similarity, kind='user')

	# load movie data 
	idx_to_movie_ = {}     
	with open('data/u.item', 'r',encoding = "ISO-8859-1") as f:
	    for line in f.readlines():
	        info = line.split('|')
	        idx_to_movie_[int(info[0])-1] = info[1]
	movies = top_k_movies(item_similarity, idx_to_movie_, idx)
	print ('* Movie name :', movies[0])
	print ('')
	print ('* Similar movie name :', movies[1:])
	print ('----------------------------')
	return movies 


def user_similar(userid_range):
	users, ratings, items, ratings_base, ratings_test = load_data()
	ratings = ratings.drop('Unnamed: 0', 1)
	ratings.columns = ['user_id', 'item_id', 'rating', 'timestamp']
	n_users = ratings.user_id.unique().shape[0]
	n_items = ratings.item_id.unique().shape[0]
	ratings_ = np.zeros((n_users, n_items))
	for row in ratings.itertuples(): 
	    ratings_[row[1]-1, row[2]-1] = row[3]
	train, test = train_test_split(ratings_)
	user_similarity = fast_similarity(train, kind='user')
	item_similarity = fast_similarity(train, kind='item')
	df_user_similarity = pd.DataFrame(user_similarity)
	for userid in range(userid_range):
		user_similar = pd.DataFrame(df_user_similarity.iloc[userid].sort_values(ascending=False).head(5)).reset_index()
		user_similar.columns = ['userid','similarity']
		print (user_similar)
		print ('----------------------------')


if __name__ == '__main__':
	#for k in range(5):
	#	main(k)
	user_similar(50)










