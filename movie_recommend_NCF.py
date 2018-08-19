# python3 

"""
* 1) Source 

- A) Get the idea from 
- https://github.com/hexiangnan/neural_collaborative_filtering
- https://arxiv.org/abs/1708.05031

- B) modify the code from 
- https://github.com/khanhnamle1994/movielens
- https://github.com/vaslnk/Spotify-Song-Recommendation-ML

* 2) Terms  

- User / Item latent vectors 
- latent means not directly observable.
- For a movie, its latent features determine the amount of action, romance, story-line, a famous actor, etc. 
- https://stats.stackexchange.com/questions/108059/meaning-of-latent-features

"""

# OPS 
import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.sparse as sp
from time import time
import multiprocessing as mp
import sys
import math
import argparse
# ML 
from keras.models import Sequential
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import keras
from keras import backend as K
from keras import initializers
from keras.models import Sequential, Model, load_model, save_model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten
from keras.optimizers import Adam
from keras.regularizers import l2


# -------------------------------------------------------------------------
# help function 

def get_data():
    route='datasets/ml-latest-small/'
    #df_movie = pd.read_csv(route +'movies.csv')
    df_ratings = pd.read_csv(route +'ratings.csv')
    return df_ratings


# Function to predict the ratings given User ID and Movie ID
def predict_rating(userId, movieId,model_):
    return model_.rate(userId - 1, movieId - 1)


def get_train_instances(train, num_negatives):
    playlist_input, item_input, labels = [],[],[]
    num_playlists = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        playlist_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in xrange(num_negatives):
            j = np.random.randint(num_items)
            while train.has_key((u, j)):
                j = np.random.randint(num_items)
            playlist_input.append(u)
            item_input.append(j)
            labels.append(0)
    return playlist_input, item_input, labels


# -------------------------------------------------------------------------
# model 


class NCF_model_V1(Sequential):

    # The constructor for the class
    def __init__(self, n_users, m_items, k_factors, **kwargs):
        # U is the embedding layer that creates an User by latent factors matrix.
        # If the intput is a user_id, U returns the latent factor vector for that user.
        U = Sequential()
        U.add(Embedding(n_users, k_factors, input_length=1))
        U.add(Reshape((k_factors,)))

        # M is the embedding layer that creates a Movie by latent factors matrix.
        # If the input is a movie_id, M returns the latent factor vector for that movie.
        M = Sequential()
        M.add(Embedding(m_items, k_factors, input_length=1))
        M.add(Reshape((k_factors,)))

        super(NCF_model, self).__init__(**kwargs)
        
        # The Merge layer takes the dot product of user and movie latent factor vectors to return the corresponding rating.
        self.add(Merge([U, M], mode='dot', dot_axes=1))

    # The rate function to predict user's rating of unrated items
    def rate(self, userId, movieId):
        return self.predict([np.array([userId]), np.array([movieId])])[0][0]



class NCF_model_V2(Sequential):

    def __init__(self, n_users, m_items, k_factors, **kwargs):
        #initializers.normal(shape, scale=0.01, name=name)
        # U is the embedding layer that creates an User by latent factors matrix.
        # If the intput is a user_id, U returns the latent factor vector for that user.
        U = Sequential()
        U.add(Embedding(n_users, k_factors, input_length=1))
        U.add(Reshape((k_factors,)))

        # M is the embedding layer that creates a Movie by latent factors matrix.
        # If the input is a movie_id, M returns the latent factor vector for that movie.
        M = Sequential()
        M.add(Embedding(m_items, k_factors, input_length=1))
        M.add(Reshape((k_factors,)))

        super(NCF_model_V2, self).__init__(**kwargs)
        
        # The Merge layer takes the dot product of user and movie latent factor vectors to return the corresponding rating.
        self.add(Merge([U, M], mode='dot', dot_axes=1))
        self.add(Dense(1, activation='sigmoid',init='lecun_uniform'))


    def rate(self, userId, movieId):
        return self.predict([np.array([userId]), np.array([movieId])])[0][0]




class NCF_model_V3(Sequential):

    def __init__(self, n_users, m_items, k_factors, **kwargs):
        #initializers.normal(shape, scale=0.01, name=name)
        # U is the embedding layer that creates an User by latent factors matrix.
        # If the intput is a user_id, U returns the latent factor vector for that user.
        U = Sequential()
        U.add(Embedding(n_users, k_factors, input_length=1))
        U.add(Reshape((k_factors,)))

        # M is the embedding layer that creates a Movie by latent factors matrix.
        # If the input is a movie_id, M returns the latent factor vector for that movie.
        M = Sequential()
        M.add(Embedding(m_items, k_factors, input_length=1))
        M.add(Reshape((k_factors,)))

        super(NCF_model_V3, self).__init__(**kwargs)
        
        # The Merge layer takes the dot product of user and movie latent factor vectors to return the corresponding rating.
        self.add(Merge([U, M], mode='dot', dot_axes=1))
        self.add(Dense(1, activation='sigmoid',init='lecun_uniform'))


    def rate(self, userId, movieId):
        return self.predict([np.array([userId]), np.array([movieId])])[0][0]



class NCF_model_V4():

    def __init__(self,n_users, m_items, latent_dim,regs):
        self.n_users = n_users
        self.m_items = m_items
        self.latent_dim = latent_dim
        self.regs = regs
        print ('n_users :' , n_users)
        print ('m_items :' , m_items)
        print ('latent_dim :' , latent_dim)

    def init_normal(self,shape, name=None):
        name=None
        scale=0.01
        #return initializers.normal()

    def get_model(self):
        # Input variables
        user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
        item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
        print ('init_normal :', self.init_normal)
        print ('n_users :', self.n_users)
        MF_Embedding_user = Embedding(input_dim = self.n_users, output_dim = self.latent_dim, name = 'user_embedding',
                                       W_regularizer = l2(self.regs[0]), input_length=1)
        MF_Embedding_Item = Embedding(input_dim = self.m_items, output_dim = self.latent_dim, name = 'item_embedding',
                                      W_regularizer = l2(self.regs[1]), input_length=1)   
        
        # Crucial to flatten an embedding vector!
        user_latent = Flatten()(MF_Embedding_user(user_input))
        item_latent = Flatten()(MF_Embedding_Item(item_input))
        
        # Element-wise product of playlist and item embeddings 
        predict_vector = merge([user_latent, item_latent], mode = 'mul')
        
        # Final prediction layer
        prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name = 'prediction')(predict_vector)     
        model = Model(input=[user_input, item_input], output=prediction)
        return model


# -------------------------------------------------------------------------
# execute the processes 

def run():
    df_ratings = get_data()
    # Define constants
    # The number of dimensional embeddings for movies and users
    K_FACTORS = 100 
    # A random test user (user_id = 2000)
    TEST_USER = 2000 
    # parameter for model V3 
    regs = [0,0]
    max_userid = max(df_ratings.userId)
    max_movieid  =  max(df_ratings.movieId)
    Users = df_ratings.head(1000).userId.values
    Movies = df_ratings.head(1000).movieId.values
    Ratings = df_ratings.head(1000).rating.values

    # --------------  MODELING  --------------

    # ----- model 1  -----
    #model = NCF_model_V1(max_userid, max_movieid, K_FACTORS)
    # ----- model 2  -----
    #model = NCF_model_V2(max_userid, max_movieid, K_FACTORS)
    # ----- model 3  -----
    #model = NCF_model_V3(max_userid, max_movieid, K_FACTORS)
    
    # ------------------------- for MODEL V4   -------------------------
    # ----- model 4  -----
    model = NCF_model_V4(max_userid, max_movieid, K_FACTORS,regs).get_model()

    # Compile the model using MSE as the loss function and the AdaMax learning algorithm
    model.compile(loss='mse', optimizer='adamax')
    # Train model
    for epoch in range(5):
        # Generate training instances
        playlist_input, item_input, labels = get_train_instances(df_ratings, num_negatives=4)

        # Training
        hist = model.fit([np.array(playlist_input), np.array(item_input)], #input
                         np.array(labels), # labels 
                         validation_split=0.20, batch_size=batch_size, nb_epoch=1, verbose=0, shuffle=True)
        print(hist.history)
    # ------------------------- for MODEL V4   -------------------------







    # Callbacks monitor the validation loss
    # Save the model weights each time the validation loss has improved
    callbacks = [EarlyStopping('val_loss', patience=2), ModelCheckpoint('weights.h5', save_best_only=True)]
    # --------------  Use 30 epochs, 90% training data, 10% validation data   --------------
    history = model.fit([Users, Movies], Ratings, nb_epoch=5, validation_split=.1, verbose=2, callbacks=callbacks)
    history.history
    # Show the best validation RMSE
    min_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
    print ('Minimum RMSE at epoch', '{:d}'.format(idx+1), '=', '{:.4f}'.format(math.sqrt(min_val_loss)))
    # --------------  Use the pre-trained model  (dev)  --------------
    #trained_model = CFModel(max_userid, max_movieid, K_FACTORS)
    # Load weights
    #trained_model.load_weights('weights.h5')
    # -------------- predict  --------------
    TEST_USER = 12
    print ('Users  : ', Users)
    #Users[Users['userId'] == TEST_USER]
    user_ratings = df_ratings[df_ratings['userId'] == TEST_USER][['userId', 'movieId', 'rating']]
    user_ratings['prediction'] = user_ratings.apply(lambda x: predict_rating(TEST_USER, x['movieId'],model), axis=1)
    #user_ratings.sort_values(by='rating', 
    #                     ascending=False).merge(movies, 
    #                                            on='movieId', 
    #                                            how='inner', 
    #                                            suffixes=['_u', '_m']).head(20)
    print (user_ratings.sort_values(by='rating', ascending=False))









# -------------------------------------------------------------------------

if __name__ == '__main__':
    run()






