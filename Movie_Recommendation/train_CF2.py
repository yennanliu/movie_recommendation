# python 3 

import numpy as np
import pandas as pd

from feature import *

# help function 


def get_error(Q, X, Y, W):
    return np.sum((W * (Q - np.dot(X, Y)))**2)

def print_recommendations(W, Q, Q_hat, movie_titles):
#def print_recommendations(W=W, Q=Q, Q_hat=Q_hat, movie_titles=movie_titles):
	#Q_hat -= np.min(Q_hat)
    #Q_hat[Q_hat < 1] *= 5
    Q_hat -= np.min(Q_hat)
    Q_hat *= float(5) / np.max(Q_hat)
    movie_ids = np.argmax(Q_hat - 5 * W, axis=1)
    for jj, movie_id in zip(range(m), movie_ids):
        #if Q_hat[jj, movie_id] < 0.1: continue
        print('User {} liked {}\n'.format(jj + 1, ', '.join([movie_titles[ii] for ii, qq in enumerate(Q[jj]) if qq > 3])))
        print('User {} did not like {}\n'.format(jj + 1, ', '.join([movie_titles[ii] for ii, qq in enumerate(Q[jj]) if qq < 3 and qq != 0])))
        print('\n User {} recommended movie is {} - with predicted rating: {}'.format(
                    jj + 1, movie_titles[movie_id], Q_hat[jj, movie_id]))
        print('\n' + 100 *  '-' + '\n')


def prepare():
	users, ratings, items, ratings_base, ratings_test = load_data()
	ratings = ratings.drop('Unnamed: 0', 1)
	ratings.columns = ['user_id', 'item_id', 'rating', 'timestamp']
	# get movie list 
	movie_titles = list(items['movie title'])
	rp = pd.pivot_table(data=ratings,columns=['item_id'],index=['user_id'],values='rating')
	rp = rp.fillna(0); # Replace NaN
	Q = rp.values
	W = Q>0.5
	W[W == True] = 1
	W[W == False] = 0
	W = W.astype(np.float64, copy=False)
	# set up parameters 
	lambda_ = 0.1
	n_factors = 100
	m, n = Q.shape
	n_iterations = 20
	X = 5 * np.random.rand(m, n_factors) 
	Y = 5 * np.random.rand(n_factors, n)
	errors = []
	for ii in range(n_iterations):
	    X = np.linalg.solve(np.dot(Y, Y.T) + lambda_ * np.eye(n_factors), 
	                        np.dot(Y, Q.T)).T
	    Y = np.linalg.solve(np.dot(X.T, X) + lambda_ * np.eye(n_factors),
	                        np.dot(X.T, Q))
	    if ii % 100 == 0:
	        print('{}th iteration is completed'.format(ii))
	    errors.append(get_error(Q, X, Y, W))
	Q_hat = np.dot(X, Y)
	print('Error of rated movies: {}'.format(get_error(Q, X, Y, W)))
	weighted_errors = []
	for ii in range(n_iterations):
	    for u, Wu in enumerate(W):
	        X[u] = np.linalg.solve(np.dot(Y, np.dot(np.diag(Wu), Y.T)) + lambda_ * np.eye(n_factors),
	                               np.dot(Y, np.dot(np.diag(Wu), Q[u].T))).T
	    for i, Wi in enumerate(W.T):
	        Y[:,i] = np.linalg.solve(np.dot(X.T, np.dot(np.diag(Wi), X)) + lambda_ * np.eye(n_factors),
	                                 np.dot(X.T, np.dot(np.diag(Wi), Q[:, i])))
	    weighted_errors.append(get_error(Q, X, Y, W))
	    print('{}th iteration is completed'.format(ii))
	weighted_Q_hat = np.dot(X,Y)
	return W,Q,Q_hat,movie_titles


def run():
	W,Q,Q_hat,movie_titles = prepare()
	print_recommendations(W,Q,Q_hat,movie_titles)


if __name__ == '__main__':
	run()





