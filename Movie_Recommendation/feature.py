import csv
import math
import sys
import numpy as np
import pandas as pd 


def init_data():
    #Reading users file:
    u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv('data/u.user', sep='|', names=u_cols,
     encoding='latin-1')

    #Reading ratings file:
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('data/u.data', sep='\t', names=r_cols,
     encoding='latin-1')

    #Reading items file:
    i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
     'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    items = pd.read_csv('data/u.item', sep='|', names=i_cols,
     encoding='latin-1')
    # save to csv 
    users.to_csv('users.csv')
    ratings.to_csv('ratings.csv')
    items.to_csv('items.csv')
    
def load_csv():
    users = pd.read_csv('users.csv')
    ratings =  pd.read_csv('ratings.csv')
    items =  pd.read_csv('items.csv')
    return users, ratings,items 
    



