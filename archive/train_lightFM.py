# python 3 


import numpy as np
import pandas as pd
from lightfm.datasets import fetch_movielens
from lightfm import LightFM
from lightfm.evaluation import precision_at_k

from feature import *


def tune():
	# import data 
	data = fetch_movielens(min_rating=5.0)

	alpha_ = [1e-05,1.5e-05,2e-05]
	epochs_ = [50,60,70]
	num_components_ = [30,32,34]

	for alpha in alpha_:
		for epochs in epochs_:
			for num_components in num_components_:
				warp_model = LightFM(no_components=num_components,
				                    loss='warp',
				                    learning_schedule='adagrad',
				                    max_sampled=100,
				                    user_alpha=alpha,
				                    item_alpha=alpha)
				warp_model.fit(data['train'], epochs=epochs, num_threads=2)
				print ('alpha = ', alpha, \
					   'epochs = ', epochs, \
					   'num_components = ', num_components )
				print("Train precision: %.2f" % precision_at_k(warp_model, data['train'], k=5).mean())
				print("Test precision: %.2f" % precision_at_k(warp_model, data['test'], k=5).mean())



def train():
	# import data 
	data = fetch_movielens(min_rating=5.0)
	# https://lyst.github.io/lightfm/docs/datasets.html
	# min_rating (float, optional) – Minimum rating to include in the interaction matrix.
	# set model super parameter 
	alpha = 1e-05
	epochs = 70
	num_components = 32

	# https://lyst.github.io/lightfm/docs/lightfm.html
	# learning_schedule (string, optional) – one of (‘adagrad’, ‘adadelta’).

	warp_model = LightFM(no_components=num_components,
	                    loss='warp',
	                    learning_schedule='adagrad',
	                    max_sampled=100,
	                    user_alpha=alpha,
	                    item_alpha=alpha)
	# fitting data 
	warp_model.fit(data['train'], epochs=30, num_threads=2)
	# precision on train, test data 
	print("Train precision: %.2f" % precision_at_k(warp_model, data['train'], k=5).mean())
	print("Test precision: %.2f" % precision_at_k(warp_model, data['test'], k=5).mean())
	model = warp_model
	return model, data 




def sample_recommendation(model, data, user_ids):
    

    n_users, n_items = data['train'].shape

    for user_id in user_ids:
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        
        scores = model.predict(user_id, np.arange(n_items))
        top_items = data['item_labels'][np.argsort(-scores)]
        
        print("User %s" % user_id)
        print("     Known positives:")
        
        for x in known_positives[:3]:
            print("        %s" % x)

        print("     Recommended:")
        
        for x in top_items[:3]:
            print("        %s" % x)


if __name__ == '__main__':
	#model, data  = train()
	#sample_recommendation(model, data, [0,1,2,3,4,13, 25, 450])
	tune()











