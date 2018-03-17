# movie_recommendation

	- `Building a light movie push system based on user rating log & item data`

### Intro


- Collaborative filtering (CF)
	- ubased_predict : User Based CF
	- mbased_predict : Item Based CF
	- similarityall_predict : All User Based CF with much smaller time complexity


- KNN 
	- dev 


- DL 
	- dev 


### File structure 
```
yennanliu@yennanliude-MacBook-Pro:~/movie_recommendation$  tree --si
.
├── [187k]  MovieLens_EDA.ipynb 
├── [ 24k]  Movie_Recommendation_Pyspark.ipynb : pyspark ML nb 
├── [2.0k]  README.md
├── [ 160]  datasets : dataset for pyspark ML training  
│   ├── [ 288]  ml-latest
│   │   ├── [9.8k]  README.txt
│   │   ├── [345M]  genome-scores.csv
│   │   ├── [ 18k]  genome-tags.csv
│   │   ├── [989k]  links.csv
│   │   ├── [2.3M]  movies.csv
│   │   ├── [710M]  ratings.csv
│   │   └── [ 27M]  tags.csv
│   └── [ 224]  ml-latest-small
│       ├── [8.4k]  README.txt
│       ├── [183k]  links.csv
│       ├── [458k]  movies.csv
│       ├── [2.4M]  ratings.csv
│       └── [ 42k]  tags.csv
├── [1.9k]  feature.py
├── [1.5k]  install_pyspark.sh : help script install local pyspark 
├── [4.9k]  train_CF1.py  : train via "user similarity", ( dataset:  /data)
├── [3.1k]  train_CF2.py  : train via "CF", (dataset :  /train_data)
├── [ 929]  train_Graphlab.py : train via "Graphlab" library 
├── [ 224]  train_data        : dataset for train_CF2.py        
│   ├── [245k]  items.csv
│   ├── [2.6M]  ratings.csv
│   ├── [2.3M]  ratings_base.csv
│   ├── [233k]  ratings_test.csv
│   └── [ 26k]  users.csv
└── [2.8k]  train_lightFM.py : train via "lightFM" library 

7 directories, 51 files
```


### Quick start 


```bash 
# get the repo 
$ git clone https://github.com/yennanliu/movie_recommendation.git
# back to the root directory 
$ cd ~
# install pyspark
$ bash /Users/<your_user_name>/movie_recommendation/install_pyspark.sh

```
### Ref 

- Movie recommendation 
	- https://github.com/jadianes/spark-movie-lens
	- https://github.com/nchah/movielens-recommender
	- https://github.com/narenkmanoharan/Movie-Recommender-System
	- http://ampcamp.berkeley.edu/5/exercises/movie-recommendation-with-mllib.html
	- https://github.com/srajangarg/movie-recommendation-algorithms
	- https://www.kaggle.com/rounakbanik/the-movies-dataset/kernels
	- https://blog.statsbot.co/recommendation-system-algorithms-ba67f39ac9a3
	- http://dataaspirant.com/2015/05/25/collaborative-filtering-recommendation-engine-implementation-in-python/
	- https://www.datacamp.com/community/tutorials/recommender-systems-python
	- https://beckernick.github.io/matrix-factorization-recommender/


- Song recommendation
	- https://towardsdatascience.com/how-to-build-a-simple-song-recommender-296fcbc8c85






