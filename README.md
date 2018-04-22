
# movie_recommendation

<img src="./images/model_demo.gif" width="600" height="279">


## INTRO 
Build a light movie recommend system via user rating, movie data, and other meta data via CF, KNN, and DL models. The output can be dump csv/DB, APIs or web APPs. 
Main idea of this prove-of-concept project is implementing ML to pratical question and making it production. 

> There will be 4 different ways building the different recommend systems.
  Current plan is : Build CF model via Pyspark, Popularity model via Numpy, KNN model via scikit-learn, and the RNN model via Tensforflow/Keras. 

Please check the theory intro, step-by-step notebook, and quick start start below.




### THEORY 


- Collaborative filtering (CF)
	- ubased_predict : User Based CF
	- mbased_predict : Item Based CF
	- similarityall_predict : All User Based CF with much smaller time complexity
- Popularity 
	- dev 
- KNN 
	- dev 
- DL 
	- dev 



### PROCESS  
- dev 



### FILE STRUCTURE 
```
yennanliu@yennanliude-MacBook-Pro:~/movie_recommendation$ tree --si
.
├── [2.8k]  README.md
├── [ 384]  archive
│   ├── [ 24k]  M_Recommend_Pyspark_demo.ipynb : demo nb of pyspark ML movie recommend model
│   ├── [8.0k]  M_Recommend_Pyspark_demo.py    : M_Recommend_Pyspark_demo.ipynb in python script 
├── [ 160]  datasets                           : dataset for movie_recommend_spark.py and  M_Recommend_Pyspark_demo.py
├── [1.6k]  install_pyspark.sh                 : help script install local pyspark 
├── [ 11k]  movie_recommend_spark_ALS.ipynb        : nb step by step demo for movie_recommend_spark_ALS.py
├── [4.1k]  movie_recommend_spark_ALS.py           : movie recommend via pyspark ML library  
└── [  96]  notebook
    └── [187k]  MovieLens_EDA.ipynb            : EDA nb 

7 directories, 53 files
```



### QUICK START 


```bash 
# get the repo 
$ git clone https://github.com/yennanliu/movie_recommendation.git
# back to the root directory 
$ cd ~
# install pyspark
$ bash /Users/$USER/movie_recommendation/install_pyspark.sh
# declare env variables  
$ export SPARK_HOME=/Users/$USER/spark && export PATH=$SPARK_HOME/bin:$PATH
# run the pyspark model train script 
$ spark-submit  movie_recommend_spark.py 
```
```bash

# output 
For rank 4 the RMSE is 0.9432575570983046
For rank 8 the RMSE is 0.9566157499964845
For rank 12 the RMSE is 0.9521388924465031
The best model was trained with rank 4
************
For testing data the RMSE is 0.9491107183690944
************
[(2.0, 107), (97328.0, 1), (4.0, 13)]
random movid id :  [3613 4927 8845 1508 5692]
-------------------
Please rate following 5 random movies as new user teste interest : 
-------------------
movie_id : 3613
movie_name : Things Change (1988)
 * What is your rating? 3
-> Your rating for Things Change (1988) is : 3.0
movie_id : 4927
movie_name : "Last Wave
 * What is your rating? 1
-> Your rating for "Last Wave is : 1.0
insert movie_id : 8845
movie_id not exist
 * What is your rating? 0
-> Your rating for None is : 0.0
movie_id : 1508
movie_name : Traveller (1997)
 * What is your rating? 3
-> Your rating for Traveller (1997) is : 3.0
insert movie_id : 5692
movie_id not exist
 * What is your rating? 2
-> Your rating for None is : 2.0
New user ratings: [(9997, 3613, 3.0), (9997, 4927, 1.0), (9997, 8845, 0.0), (9997, 1508, 3.0), (9997, 5692, 2.0)]
[(1.0, 31.0, 2.5), (1.0, 1029.0, 3.0), (1.0, 1061.0, 3.0), (1.0, 1129.0, 2.0), (1.0, 1172.0, 4.0), (1.0, 1263.0, 2.0), (1.0, 1287.0, 2.0), (1.0, 1293.0, 2.0), (1.0, 1339.0, 3.5), (1.0, 1343.0, 2.0)]
<pyspark.mllib.recommendation.MatrixFactorizationModel object at 0x10f4b1240>
=======================
[Rating(user=9997, product=267, rating=1.9431035590658032), Rating(user=9997, product=18, rating=2.4471404575434224), Rating(user=9997, product=227, rating=1.8898669807166826), Rating(user=9997, product=639, rating=1.2313836204250688), Rating(user=9997, product=630, rating=2.0651897033288247), Rating(user=9997, product=248, rating=0.9056995408584969), Rating(user=9997, product=183, rating=1.096099378407863), Rating(user=9997, product=62, rating=2.3965661520727375), Rating(user=9997, product=318, rating=2.693287049630902), Rating(user=9997, product=6, rating=2.403548622949053)]
=======================
=======================
TOP recommended movies (with more than 25 reviews):
('Forrest Gump (1994)', 2.740902212733893, 341)
('Braveheart (1995)', 2.7270995452301943, 228)
('"Shawshank Redemption', 2.693287049630902, 311)
("Schindler's List (1993)", 2.6615360145071953, 244)
('Much Ado About Nothing (1993)', 2.646212665422727, 60)
('Welcome to the Dollhouse (1995)', 2.578621020576323, 30)
('Philadelphia (1993)', 2.5727445302939564, 86)
.....
```


### REFERENCE 

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






