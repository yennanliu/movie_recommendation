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
yennanliu@yennanliude-MacBook-Pro:~/movie_recommendation$ tree --si
.
├── [2.8k]  README.md
├── [ 384]  archive
│   ├── [ 24k]  M_Recommend_Pyspark_demo.ipynb : demo nb of pyspark ML movie recommend model
│   ├── [8.0k]  M_Recommend_Pyspark_demo.py    : M_Recommend_Pyspark_demo.ipynb in python script 
│   ├── [ 832]  data                           : dataset for train_CF1.py
│   ├── [1.9k]  feature.py                     : python script extract data features (dev)
│   ├── [4.9k]  train_CF1.py                   : train via "user similarity", ( dataset:  archive/data)
│   ├── [3.1k]  train_CF2.py                   : train via "CF", (dataset :  archive/train_data)
│   ├── [ 929]  train_Graphlab.py              : train via "Graphlab" library
│   ├── [ 224]  train_data                     : dataset for train_CF2.py   
│   └── [2.8k]  train_lightFM.py               : train via "lightFM" library
├── [ 160]  datasets                           : dataset for movie_recommend_spark.p and M_Recommend_Pyspark_demo.py
├── [1.6k]  install_pyspark.sh                 : help script install local pyspark 
├── [ 11k]  movie_recommend_spark.ipynb        : nb step by step demo for movie_recommend_spark.p
├── [4.1k]  movie_recommend_spark.py           : movie recommend via pyspark ML library  
└── [  96]  notebook
    └── [187k]  MovieLens_EDA.ipynb            : EDA nb 

7 directories, 53 files
```


### Quick start 


```bash 
# get the repo 
$ git clone https://github.com/yennanliu/movie_recommendation.git
# back to the root directory 
$ cd ~
# install pyspark
$ bash /Users/<your_user_name>/movie_recommendation/install_pyspark.sh
# declare env variables  
$ export SPARK_HOME=/Users/<your_user_name>/spark
$ export PATH=$SPARK_HOME/bin:$PATH
# run the pyspark script 
$ spark-submit  movie_recommend_spark.py 
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






