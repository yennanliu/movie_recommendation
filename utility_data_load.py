# python 3 
# python 
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

# pyspark 
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
		small_movies_titles = small_movies_data.map(lambda x: (int(x[0]),x[1]))
		return small_ratings_data, small_movies_data, small_movies_titles

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
		return complete_ratings_data, complete_movies_data, complete_movies_titles
