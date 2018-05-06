
# python3 

import pandas as pd 
from sklearn.cross_validation import train_test_split
import numpy as np
import time
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



# ML




#Class for Item similarity based Recommender System model
class item_similarity_recommender_py():
    def __init__(self):
        self.train_data = None
        self.userId = None
        self.movieId = None
        self.cooccurence_matrix = None
        self.movies_dict = None
        self.rev_movies_dict = None
        self.item_similarity_recommendations = None
        
    #Get unique items (songs) corresponding to a given user
    def get_user_items(self, user):
        user_data = self.train_data[self.train_data[self.userId] == user]
        user_items = list(user_data[self.movieId].unique())
        
        return user_items
        
    #Get unique users for a given item (song)
    def get_item_users(self, item):
        item_data = self.train_data[self.train_data[self.movieId] == item]
        item_users = set(item_data[self.userId].unique())
            
        return item_users
        
    #Get unique items (songs) in the training data
    def get_all_items_train_data(self):
        all_items = list(self.train_data[self.movieId].unique())
            
        return all_items
    

        
    #Construct cooccurence matrix
    def construct_cooccurence_matrix(self, user_movies, all_movies):
            
        ####################################
        #Get users for all songs in user_songs.
        ####################################
        user_movies_users = []        
        for i in range(0, len(user_movies)):
            user_movies_users.append(self.get_item_users(user_movies[i]))
            
        ###############################################
        #Initialize the item cooccurence matrix of size 
        #len(user_songs) X len(songs)
        ###############################################
        cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_movies), len(all_movies))), float)
           
        #############################################################
        #Calculate similarity between user songs and all unique songs
        #in the training data
        #############################################################
        for i in range(0,len(all_movies)):
            #Calculate unique listeners (users) of song (item) i
            movie_i_data = self.train_data[self.train_data[self.movieId] == all_movies[i]]
            users_i = set(movie_i_data[self.userId].unique())
            
            for j in range(0,len(user_movies)):       
                    
                #Get unique listeners (users) of song (item) j
                users_j = user_movies_users[j]
                    
                #Calculate intersection of listeners of songs i and j
                users_intersection = users_i.intersection(users_j)
                
                #Calculate cooccurence_matrix[i,j] as Jaccard Index
                if len(users_intersection) != 0:
                    #Calculate union of listeners of songs i and j
                    users_union = users_i.union(users_j)
                    
                    cooccurence_matrix[j,i] = float(len(users_intersection))/float(len(users_union))
                else:
                    cooccurence_matrix[j,i] = 0
                    
        
        return cooccurence_matrix
    
    
    #Use the cooccurence matrix to make top recommendations
    def generate_top_recommendations(self, user, cooccurence_matrix, all_movies, user_movies):
        print("Non zero values in cooccurence_matrix :%d" % np.count_nonzero(cooccurence_matrix))
        
        #Calculate a weighted average of the scores in cooccurence matrix for all user songs.
        user_sim_scores = cooccurence_matrix.sum(axis=0)/float(cooccurence_matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()
 
        #Sort the indices of user_sim_scores based upon their value
        #Also maintain the corresponding score
        sort_index = sorted(((e,i) for i,e in enumerate(list(user_sim_scores))), reverse=True)
    
        #Create a dataframe from the following
        #columns = ['user_id', 'song', 'score', 'rank']
        columns = ['userId', 'movieId' ,'rating' ,'view_count']
        #index = np.arange(1) # array of numbers for the number of samples
        df = pd.DataFrame(columns=columns)
         
        #Fill the dataframe with top 10 item based recommendations
        rank = 1 
        for i in range(0,len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_movies[sort_index[i][1]] not in user_movies and rank <= 10:
                df.loc[len(df)]=[user,all_movies[sort_index[i][1]],sort_index[i][0],rank]
                rank = rank+1
        
        #Handle the case where there are no recommendations
        if df.shape[0] == 0:
            print("The current user has no songs for training the item similarity based recommendation model.")
            return -1
        else:
            return df
    

    #Create the item similarity based recommender system model
    def create(self, train_data, userId, movieId):
        self.train_data = train_data
        self.userId = userId
        self.movieId = movieId

    #Use the item similarity based recommender system model to
    #make recommendations
    def recommend(self, user):
        
        ########################################
        #A. Get all unique songs for this user
        ########################################
        user_movies = self.get_user_items(user)   
        print ('------------')
        print (user_movies)
        print ('------------')
            
        #print("No. of unique movies for the user: %d" % len(user_movies))
        
        ######################################################
        #B. Get all unique items (songs) in the training data
        ######################################################
        all_movies = self.get_all_items_train_data()
        
        print("no. of unique movies in the training set: %d" % len(all_movies))
         
        ###############################################
        #C. Construct item cooccurence matrix of size 
        #len(user_songs) X len(songs)
        ###############################################
        cooccurence_matrix = self.construct_cooccurence_matrix(user_movies, all_movies)
        
        #######################################################
        #D. Use the cooccurence matrix to make recommendations
        #######################################################
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_movies, user_movies)
        print (df_recommendations)       
        return df_recommendations
    
    #Get similar items to given items
    def get_similar_items(self, item_list):
        
        user_movies = item_list
        
        ######################################################
        #B. Get all unique items (songs) in the training data
        ######################################################
        all_songs = self.get_all_items_train_data()
        
        print("no. of unique songs in the training set: %d" % len(all_movies))
         
        ###############################################
        #C. Construct item cooccurence matrix of size 
        #len(user_songs) X len(songs)
        ###############################################
        cooccurence_matrix = self.construct_cooccurence_matrix(user_movies, all_movies)
        
        #######################################################
        #D. Use the cooccurence matrix to make recommendations
        #######################################################
        user = ""
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_movies, user_movies)
        print (df_recommendations)
        return df_recommendations




# -------------------------------------


if __name__ == '__main__':
	# data preprocess
	df_ratings = get_data()
	movie_grouped = data_preprocess(df_ratings)
	print (movie_grouped)
	train_data, test_data = get_train_test_data(df_ratings)
	# modeling  
	is_model = item_similarity_recommender_py()
	is_model.create(train_data, 'userId', 'movieId')
	#Print the songs for the user in training data
	user_id = 30
	user_items = is_model.get_user_items(user_id)
	#
	print("------------------------------------------------------------------------------------")
	print("Training data movies for the user userid: %s:" % user_id)
	print("------------------------------------------------------------------------------------")

	for user_item in user_items:
	    print(user_item)

	print("----------------------------------------------------------------------")
	print("Recommendation process going on:")
	print("----------------------------------------------------------------------")

	#Recommend songs for the user using personalized model
	is_model.recommend(user_id)




