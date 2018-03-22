# python 3 


# import library 
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS
from time import time
import math
import os 


sc =SparkContext()
sc

# --- modify here for user customize case --- # 

datasets_path = '/Users/yennanliu/movie_recommendation/datasets/'


# help function 
def get_counts_and_averages(ID_and_ratings_tuple):
    nratings = len(ID_and_ratings_tuple[1])
    return ID_and_ratings_tuple[0], (nratings, float(sum(x for x in ID_and_ratings_tuple[1]))/nratings)



def main():
	# get small dataset 
	small_ratings_file = os.path.join(datasets_path, 'ml-latest-small', 'ratings.csv')
	small_ratings_raw_data = sc.textFile(small_ratings_file)
	small_ratings_raw_data_header = small_ratings_raw_data.take(1)[0]
	small_ratings_data = small_ratings_raw_data.filter(lambda line: line!=small_ratings_raw_data_header)\
	    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1],tokens[2])).cache()

	# join with another big dataset    
	small_movies_file = os.path.join(datasets_path, 'ml-latest-small', 'movies.csv')
	small_movies_raw_data = sc.textFile(small_movies_file)
	small_movies_raw_data_header = small_movies_raw_data.take(1)[0]
	small_movies_data = small_movies_raw_data.filter(lambda line: line!=small_movies_raw_data_header)\
	    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1])).cache()
	training_RDD, validation_RDD, test_RDD = small_ratings_data.randomSplit([6, 2, 2])
	validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
	test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

	seed = 30
	iterations = 10
	regularization_parameter = 0.1
	ranks = [4, 8, 12]
	errors = [0, 0, 0]
	err = 0
	tolerance = 0.02

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

	# model
	print ('----------- model ----------------')
	model = ALS.train(training_RDD, best_rank, seed=seed, iterations=iterations,
	                      lambda_=regularization_parameter)
	predictions = model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
	rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
	error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
	print ('For testing data the RMSE is %s' % (error))

	# Load the complete dataset file
	complete_ratings_file = os.path.join(datasets_path, 'ml-latest', 'ratings.csv')
	complete_ratings_raw_data = sc.textFile(complete_ratings_file)
	complete_ratings_raw_data_header = complete_ratings_raw_data.take(1)[0]
	# Parse
	complete_ratings_data = complete_ratings_raw_data.filter(lambda line: line!=complete_ratings_raw_data_header)\
	    .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2]))).cache()

	print ("There are %s recommendations in the complete dataset" % (complete_ratings_data.count()))

	# split train / test
	training_RDD, test_RDD = complete_ratings_data.randomSplit([7, 3], seed=30)
	complete_model = ALS.train(training_RDD, best_rank, seed=seed, 
	                           iterations=iterations, lambda_=regularization_parameter)
	# predict test data 
	test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))
	predictions = complete_model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
	rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
	error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
	print ('For testing data the RMSE is %s' % (error))
	complete_movies_file = os.path.join(datasets_path, 'ml-latest', 'movies.csv')
	complete_movies_raw_data = sc.textFile(complete_movies_file)
	complete_movies_raw_data_header = complete_movies_raw_data.take(1)[0]
	# Parse
	complete_movies_data = complete_movies_raw_data.filter(lambda line: line!=complete_movies_raw_data_header)\
	    .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),tokens[1],tokens[2])).cache()
	complete_movies_titles = complete_movies_data.map(lambda x: (int(x[0]),x[1]))
	print ("There are %s movies in the complete dataset" % (complete_movies_titles.count()))

	movie_ID_with_ratings_RDD = (complete_ratings_data.map(lambda x: (x[1], x[2])).groupByKey())
	movie_ID_with_avg_ratings_RDD = movie_ID_with_ratings_RDD.map(get_counts_and_averages)
	movie_rating_counts_RDD = movie_ID_with_avg_ratings_RDD.map(lambda x: (x[0], x[1][0]))

	new_user_ID = 0

	# The format of each line is (userID, movieID, rating)
	new_user_ratings = [
	     (0,260,9), # Star Wars (1977)
	     (0,1,8), # Toy Story (1995)
	     (0,16,7), # Casino (1995)
	     (0,25,8), # Leaving Las Vegas (1995)
	     (0,32,9), # Twelve Monkeys (a.k.a. 12 Monkeys) (1995)
	     (0,335,4), # Flintstones, The (1994)
	     (0,379,3), # Timecop (1994)
	     (0,296,7), # Pulp Fiction (1994)
	     (0,858,10) , # Godfather, The (1972)
	     (0,50,8) # Usual Suspects, The (1995)
	    ]
	new_user_ratings_RDD = sc.parallelize(new_user_ratings)
	print ('New user ratings: %s' % new_user_ratings_RDD.take(10))

	complete_data_with_new_ratings_RDD = complete_ratings_data.union(new_user_ratings_RDD)

	t0 = time()
	new_ratings_model = ALS.train(complete_data_with_new_ratings_RDD, best_rank, seed=seed, 
	                              iterations=iterations, lambda_=regularization_parameter)
	tt = time() - t0
	print ("New model trained in %s seconds" % round(tt,3))
	new_user_ratings_ids = map(lambda x: x[1], new_user_ratings) # get just movie IDs
	# keep just those not on the ID list (thanks Lei Li for spotting the error!)
	new_user_unrated_movies_RDD = (complete_movies_data.filter(lambda x: x[0] not in new_user_ratings_ids).map(lambda x: (new_user_ID, x[0])))
	# Use the input RDD, new_user_unrated_movies_RDD, with new_ratings_model.predictAll() to predict new ratings for the movies
	new_user_recommendations_RDD = new_ratings_model.predictAll(new_user_unrated_movies_RDD)
	# Transform new_user_recommendations_RDD into pairs of the form (Movie ID, Predicted Rating)
	new_user_recommendations_rating_RDD = new_user_recommendations_RDD.map(lambda x: (x.product, x.rating))
	new_user_recommendations_rating_title_and_count_RDD = \
	    new_user_recommendations_rating_RDD.join(complete_movies_titles).join(movie_rating_counts_RDD)
	new_user_recommendations_rating_title_and_count_RDD.take(3)
	new_user_recommendations_rating_title_and_count_RDD = \
	    new_user_recommendations_rating_title_and_count_RDD.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))

	top_movies = new_user_recommendations_rating_title_and_count_RDD.filter(lambda r: r[2]>=25).takeOrdered(25, key=lambda x: -x[1])
	print ('TOP recommended movies (with more than 25 reviews):\n%s' %
	        '\n'.join(map(str, top_movies)))

if __name__ == '__main__':
	main()



# ---------------------------

# run the spark python script via command line 
# $export SPARK_HOME=/Users/yennanliu/spark
# $export PATH=$SPARK_HOME/bin:$PATH
# $spark-submit Movie_Recommendation_Pyspark.py

# ---------------------------








