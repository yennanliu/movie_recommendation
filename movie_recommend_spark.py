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
def get_data():
    # rating small dataset 
    # userid, movieid, rating, timestamp 
    datasets_path = '/Users/yennanliu/movie_recommendation/datasets/'
    small_ratings_file = os.path.join(datasets_path, 'ml-latest-small', 'ratings.csv')
    small_ratings_raw_data = sc.textFile(small_ratings_file)
    # get heater 
    small_ratings_raw_data_header = small_ratings_raw_data.take(1)[0]
    # filter out header 
    # only get 1st, 2rd, and 3rd columns
    small_ratings_data = small_ratings_raw_data.filter(lambda line: line!=small_ratings_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1],tokens[2])).cache()
    # movie dataset 
    # movieid, name 
    small_movies_file = os.path.join(datasets_path, 'ml-latest-small', 'movies.csv')
    small_movies_raw_data = sc.textFile(small_movies_file)
    small_movies_raw_data_header = small_movies_raw_data.take(1)[0]
    # filter out header 
    small_movies_data = small_movies_raw_data.filter(lambda line: line!=small_movies_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1])).cache()
    
    return small_ratings_data, small_movies_data


def train_test_split(dataset):
    # split data into train (60%), validate (20%), and test (20%)
    training_RDD, validation_RDD, test_RDD = dataset.randomSplit([6, 2, 2])
    validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
    test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))
    return training_RDD, validation_RDD, test_RDD, validation_for_predict_RDD, test_for_predict_RDD

# ML 


def ALS_model(training_RDD,validation_RDD,validation_for_predict_RDD):
    # super parameters 
    seed = 30
    iterations = 10
    regularization_parameter = 0.1
    ranks = [4, 8, 12]
    errors = [0, 0, 0]
    err = 0
    tolerance = 0.02
    # minor setting 
    min_error = float('inf')
    best_rank = -1
    best_iteration = -1
    for rank in ranks:
        model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,
                          lambda_=regularization_parameter)
        predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
        rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
        error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
        errors[err] = error
        err += 1
        # --- fix here for python 3 --- #
        print ('For rank %s the RMSE is %s' % (rank, error))
        if error < min_error:
            min_error = error
            best_rank = rank
    # --- fix here for python 3 --- #
    print ('The best model was trained with rank %s' % best_rank)



# ----------------------


if __name__ == '__main__':
	# dataset preview 
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
	# ------------ Model Training  ------------ #
	# get data 
	small_ratings_data,small_movies_data = get_data()
	# train, test split 
	training_RDD, validation_RDD, test_RDD, validation_for_predict_RDD, test_for_predict_RDD = train_test_split(small_ratings_data)
	# run ALS model 
	ALS_model(training_RDD,validation_RDD,validation_for_predict_RDD)




