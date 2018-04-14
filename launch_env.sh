#!/bin/sh

# launch spark env 
source activate pyspark_
# declare env variables 
export SPARK_HOME=/Users/$USER/spark 
export PATH=$SPARK_HOME/bin:$PATH
# ready for running spark script via command line  
# spark-submit movie_recommend_spark.py 
#spark-submit  movie_recommend_spark.py
 