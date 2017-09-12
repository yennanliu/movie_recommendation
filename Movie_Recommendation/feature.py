import csv
import math
import sys
import numpy as np
import pandas as pd
import datetime


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
    
def load_train_test_data():
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings_base = pd.read_csv('data/ua.base', sep='\t', names=r_cols, encoding='latin-1')
    ratings_test = pd.read_csv('data/ua.test', sep='\t', names=r_cols, encoding='latin-1')
    ratings_base.to_csv('ratings_base.csv')
    ratings_test.to_csv('ratings_test.csv')

    
def load_data():
    users = pd.read_csv('users.csv')
    ratings =  pd.read_csv('ratings.csv')
    items =  pd.read_csv('items.csv')
    ratings_base = pd.read_csv('ratings_base.csv')
    ratings_test = pd.read_csv('ratings_test.csv')
    return users, ratings, items, ratings_base, ratings_test
    

def unix2datetime(x):
    temp = datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d')
    return temp








