# python 3 

"""
# ref 
# https://github.com/llSourcell/recommender_live/blob/master/Song%20Recommender_Python.ipynb
# https://towardsdatascience.com/how-to-build-a-simple-song-recommender-296fcbc8c85



* Recommend model based on popular movie 
* Note that the output of this model is SAME TO ALL USERID 
* Output : 

userId :  6
---------
      userId  movieId  score  Rank
264        6      296    267   1.0
318        6      356    267   2.0
282        6      318    249   3.0
519        6      593    233   4.0
230        6      260    230   5.0
423        6      480    215   6.0
2005       6     2571    213   7.0
0          6        1    209   8.0
1001       6     1270    194   9.0
529        6      608    192  10.0
---------
userId :  16
---------
      userId  movieId  score  Rank
264       16      296    267   1.0
318       16      356    267   2.0
282       16      318    249   3.0
519       16      593    233   4.0
230       16      260    230   5.0
423       16      480    215   6.0
2005      16     2571    213   7.0
0         16        1    209   8.0
1001      16     1270    194   9.0
529       16      608    192  10.0
---------




"""

import numpy as np
import pandas as pd 
import time
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib


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



# -------------------------------------


# ML 

class popularity_recommender():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.popularity_recommendations = None
        
    #Create the popularity based recommender system model
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

        #Get a count of user_ids for each unique song as recommendation score
        train_data_grouped = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()
        train_data_grouped.rename(columns = {'userId': 'score'},inplace=True)
    
        #Sort the songs based upon recommendation score
        train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending = [0,1])
    
        #Generate a recommendation rank based upon score
        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')
        
        #Get the top 10 recommendations
        self.popularity_recommendations = train_data_sort.head(10)

    #Use the popularity based recommender system model to
    #make recommendations
    def recommend(self, user_id):    
        user_recommendations = self.popularity_recommendations
        #print (user_recommendations)
        
        #Add user_id column for which the recommendations are being generated
        user_recommendations['userId'] = user_id
        print ('userId : ', user_id)
    
        #Bring user_id column to the front
        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols]
        print ('---------')
        print (user_recommendations)
        print ('---------')
        return user_recommendations


# -------------------------------------


if __name__ == '__main__':
    df_ratings = get_data()
    users = df_ratings.userId.unique()
    train_data, test_data = get_train_test_data(df_ratings)
    ### ML ###
    model = popularity_recommender()
    model.create(train_data, 'userId', 'movieId')
    # recommend for user  6
    user_id = users[5]
    model.recommend(user_id)
    # recommend for user 16 
    user_id = users[15]
    model.recommend(user_id)








