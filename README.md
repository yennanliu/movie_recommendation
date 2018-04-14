
movie_recommendation
============
Build a light movie recommend system via user rating, movie data, and other meta data via CF, KNN, and DL models. The output can be dump csv/DB, APIs or web APPs. 
Main idea of this prove-of-concept project is implementing ML to pratical question and making it production. 

> There will be 3 different ways building the different recommend systems.
  Current plan is : Build CF model via Pyspark, build KNN model via scikit-learnm, and the
  RNN model via Tensforflow/Keras. 

Please check the theory intro, step-by-step notebook, and quick start start below.


### INTRO


- Collaborative filtering (CF)
	- ubased_predict : User Based CF
	- mbased_predict : Item Based CF
	- similarityall_predict : All User Based CF with much smaller time complexity
- KNN 
	- dev 
- DL 
	- dev 




### Process 
- dev 


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
For rank 4 the RMSE is 0.9359471213361842
For rank 8 the RMSE is 0.945083657995206
For rank 12 the RMSE is 0.947288651524303
The best model was trained with rank 4
************
For testing data the RMSE is 0.9505235687561104
************
[(2.0, 107), (97328.0, 1), (4.0, 13)]
New user ratings: [(0, 260, 9), (0, 1, 8), (0, 16, 7), (0, 25, 8), (0, 32, 9), (0, 335, 4), (0, 379, 3), (0, 296, 7), (0, 858, 10), (0, 50, 8)]
[(1.0, 31.0, 2.5), (1.0, 1029.0, 3.0), (1.0, 1061.0, 3.0), (1.0, 1129.0, 2.0), (1.0, 1172.0, 4.0), (1.0, 1263.0, 2.0), (1.0, 1287.0, 2.0), (1.0, 1293.0, 2.0), (1.0, 1339.0, 3.5), (1.0, 1343.0, 2.0)]
<pyspark.mllib.recommendation.MatrixFactorizationModel object at 0x10c6b2c18>
=======================
[Rating(user=0, product=496, rating=5.916224600729824), Rating(user=0, product=349, rating=6.165838514210409), Rating(user=0, product=353, rating=6.1812863730122825), Rating(user=0, product=271, rating=4.8335955795266425), Rating(user=0, product=335, rating=4.15584619159136), Rating(user=0, product=312, rating=5.773801618133824), Rating(user=0, product=96, rating=1.6675354378462282), Rating(user=0, product=425, rating=6.551042158690938), Rating(user=0, product=251, rating=4.749453786985258), Rating(user=0, product=455, rating=4.441779083807681)]
=======================
=======================
TOP recommended movies (with more than 25 reviews):
('"Postman', 8.799329003993375, 45)
('Shadowlands (1993)', 8.522045319247562, 25)
('"Madness of King George', 8.517315636677042, 39)
('Shallow Grave (1994)', 8.460473843101195, 38)
('"Shawshank Redemption', 8.392621787267679, 311)
('"Usual Suspects', 8.34299873957546, 201)
("Schindler's List (1993)", 8.327829872104124, 244)
('"Remains of the Day', 8.325397737582897, 46)
('Three Colors: Red (Trois couleurs: Rouge) (1994)', 8.325049982909732, 32)
('Fargo (1996)', 8.314538759182922, 224)
('In the Name of the Father (1993)', 8.278309549089208, 31)
('Pulp Fiction (1994)', 8.25876240525476, 324)
('Hoop Dreams (1994)', 8.254244496540839, 61)
('Searching for Bobby Fischer (1993)', 8.137494762727776, 45)
('Three Colors: Blue (Trois couleurs: Bleu) (1993)', 8.124997685535352, 31)
('"City of Lost Children', 8.06900109014378, 40)
('Mystery Science Theater 3000: The Movie (1996)', 8.055024356863075, 33)
('Star Wars: Episode IV - A New Hope (1977)', 8.048430843995435, 291)
('"Hudsucker Proxy', 8.031615721738133, 49)
('Heavenly Creatures (1994)', 8.008045456962229, 43)
('Léon: The Professional (a.k.a. The Professional) (Léon) (1994)', 8.004693631107212, 132)
('Like Water for Chocolate (Como agua para chocolate) (1992)', 7.985160321089869, 62)
('Taxi Driver (1976)', 7.955982328737049, 118)
('Sense and Sensibility (1995)', 7.945264458703306, 86)
('"Piano', 7.846991400658464, 78)
=======================
```



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
├── [ 11k]  movie_recommend_spark.ipynb        : nb step by step demo for movie_recommend_spark.py
├── [4.1k]  movie_recommend_spark.py           : movie recommend via pyspark ML library  
└── [  96]  notebook
    └── [187k]  MovieLens_EDA.ipynb            : EDA nb 

7 directories, 53 files
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






