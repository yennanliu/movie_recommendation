# python 3 
# ML 
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
# ops 
from time import time
import time
import math
import os 
import pandas as pd 
import numpy as np 


# import SparkContext
from pyspark import SparkContext
sc =SparkContext()



# help functions 
# ----------------------

# Preprocess 

def get_data_preview():
	datasets_path = '/Users/yennanliu/movie_recommendation/datasets/'
	small_ratings_file = os.path.join(datasets_path, 'ml-latest-small', 'ratings.csv')
	small_movies_file = os.path.join(datasets_path, 'ml-latest-small', 'movies.csv')
	print (" ----------------------- ")
	time.sleep(5)
	df_rating=pd.read_csv(small_ratings_file)
	df_movie=pd.read_csv(small_movies_file)
	print (df_rating.head(3))
	print (df_movie.head(3))
	print (" ----------------------- ")
	time.sleep(5)





def get_data(full_dataset=False):

	datasets_path = '/Users/yennanliu/movie_recommendation/datasets/'
	if full_dataset==False:
		#------  rating small dataset  ------#
		# userid, movieid, rating, timestamp 
		small_ratings_file = os.path.join(datasets_path, 'ml-latest-small', 'ratings.csv')
		small_ratings_raw_data = sc.textFile(small_ratings_file)
		# get heater 
		small_ratings_raw_data_header = small_ratings_raw_data.take(1)[0]
		# filter out header 
		# only get 1st, 2rd, and 3rd columns
		# fix dtype to float  
		small_ratings_data = small_ratings_raw_data.filter(lambda line: line!=small_ratings_raw_data_header)\
		.map(lambda line: line.split(",")).map(lambda tokens: (float(tokens[0]),float(tokens[1]),float(tokens[2]))).cache()
		#------ movie dataset ------#
		# movieid, name 
		small_movies_file = os.path.join(datasets_path, 'ml-latest-small', 'movies.csv')
		small_movies_raw_data = sc.textFile(small_movies_file)
		small_movies_raw_data_header = small_movies_raw_data.take(1)[0]
		# filter out header 
		# fix dtype to float  
		small_movies_data = small_movies_raw_data.filter(lambda line: line!=small_movies_raw_data_header)\
		.map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1],tokens[2])).cache()
		return small_ratings_data, small_movies_data

	elif full_dataset==True:
		#------ rating completed dataset  ------# 
		complete_ratings_file = os.path.join(datasets_path, 'ml-latest', 'ratings.csv')
		complete_ratings_raw_data = sc.textFile(complete_ratings_file)
		complete_ratings_raw_data_header = complete_ratings_raw_data.take(1)[0]
		# filter out header 
		complete_ratings_data = complete_ratings_raw_data.filter(lambda line: line!=complete_ratings_raw_data_header)\
		    .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2]))).cache()
		#------ movie completed dataset  ------# 
		complete_movies_file = os.path.join(datasets_path, 'ml-latest', 'movies.csv')
		complete_movies_raw_data = sc.textFile(complete_movies_file)
		complete_movies_raw_data_header = complete_movies_raw_data.take(1)[0]
		# filter out header 
		complete_movies_data = complete_movies_raw_data.filter(lambda line: line!=complete_movies_raw_data_header)\
		    .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),tokens[1],tokens[2])).cache()
		complete_movies_titles = complete_movies_data.map(lambda x: (int(x[0]),x[1]))
		return complete_ratings_data, complete_movies_data

 


def train_test_split(dataset):
	# split data into train (60%), validate (20%), and test (20%)
	training_RDD, validation_RDD, test_RDD = dataset.randomSplit([6, 2, 2])
	validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
	test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))
	return training_RDD, validation_RDD, test_RDD, validation_for_predict_RDD, test_for_predict_RDD



def get_counts_and_averages(ID_and_ratings_tuple):
    nratings = len(ID_and_ratings_tuple[1])
    return ID_and_ratings_tuple[0], (nratings, float(sum(x for x in ID_and_ratings_tuple[1]))/nratings)



# ML 


def ALS_model(training_RDD,validation_RDD,validation_for_predict_RDD):
	# ------------- 
	# super parameters
	err=0
	min_error = float('inf')
	parameter = {}
	parameter['seed'] = 30
	parameter['iterations'] = 10
	parameter['regularization_parameter'] = 0.1
	parameter['ranks'] = [4, 8, 12]
	parameter['errors'] = [0, 0, 0]
	parameter['tolerance'] = 0.02
	# -------------
	# train the model over super parameters sets 
	for rank in parameter['ranks']:
		model = ALS.train(training_RDD, rank, seed=parameter['seed'], iterations=parameter['iterations'],lambda_=parameter['regularization_parameter'])
		predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
		# join real rating and predicted rating 
		rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
		# predicted rating error (mean square error)
		error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
		parameter['errors'][err] = error
		err += 1
		# --- fix here for python 3 --- #
		print ('For rank %s the RMSE is %s' % (rank, error))
		if error < min_error:
			min_error = error
			best_rank = rank
	# --- fix here for python 3 --- #
	print ('The best model was trained with rank %s' % best_rank)

	return model, predictions, rates_and_preds, min_error,best_rank, parameter


def ALS_model_predict(model,test_for_predict_RDD,test_RDD):
	#model = ALS.train(training_RDD, best_rank, seed=parameter['seed'], iterations=parameter['iterations'],lambda_=parameter['regularization_parameter'])
	predictions = model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
	rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
	error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
	print ('For testing data the RMSE is %s' % (error))




# ----------------------


if __name__ == '__main__':
	# dataset preview 
	get_data_preview()
	# get data 
	small_ratings_data,small_movies_data = get_data()
	# train, test split 
	training_RDD, validation_RDD, test_RDD, validation_for_predict_RDD, test_for_predict_RDD = train_test_split(small_ratings_data)
	# ------------ Model Training  ------------ #
	# train ALS model 
	model, predictions, rates_and_preds, min_error,best_rank, parameter = ALS_model(training_RDD,validation_RDD,validation_for_predict_RDD)
	# predict with trained ALS model 
	print ('************')
	ALS_model_predict(model,test_for_predict_RDD,test_RDD)
	print ('************')
	#complete_ratings_data, complete_movies_data = get_data(full_dataset=True)
	#print(complete_ratings_data.take(3))







