
# python 2.7 
# https://www.analyticsvidhya.com/blog/2016/06/quick-guide-build-recommendation-engine-python/




import graphlab
from feature import * 



def train_predict():
	# load data 
	users, ratings, items, ratings_base, ratings_test = load_data()	
	train_data = graphlab.SFrame(ratings_base)
	test_data = graphlab.SFrame(ratings_test)

	# train 
	popularity_model = graphlab.popularity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')
	#Get recommendations for first 5 users and print them
	#users = range(1,6) specifies user ID of first 5 users
	#k=5 specifies top 5 recommendations to be given

	# predict 
	popularity_recomm = popularity_model.recommend(users=range(1,20),k=10)
	popularity_recomm.print_rows(num_rows=250)
	ratings_base.groupby(by='movie_id')['rating'].mean().sort_values(ascending=False).head(20)


if __name__ == '__main__':
	train_predict()



















