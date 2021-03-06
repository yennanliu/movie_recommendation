/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//package org.apache.spark.examples.ml;

/*
credit 
https://github.com/apache/spark/blob/master/examples/src/main/java/org/apache/spark/examples/ml/JavaALSExample.java
*/

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

// $example on$
import java.io.Serializable;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
// $example off$

public class movie_recommend_ALS {

  // $example on$
  public static class Rating implements Serializable {
    private int userId;
    private int movieId;
    private float rating;
    private long timestamp;

    public Rating() {}

    public Rating(int userId, int movieId, float rating, long timestamp) {
      this.userId = userId;
      this.movieId = movieId;
      this.rating = rating;
      this.timestamp = timestamp;
    }

    public int getUserId() {
      return userId;
    }

    public int getMovieId() {
      return movieId;
    }

    public float getRating() {
      return rating;
    }

    public long getTimestamp() {
      return timestamp;
    }

    public static Rating parseRating(String str) {
      String[] fields = str.split(",");
      if (fields.length != 4) {
        throw new IllegalArgumentException("Each line must contain 4 fields");
      }
      int userId = Integer.parseInt(fields[0]);
      int movieId = Integer.parseInt(fields[1]);
      float rating = Float.parseFloat(fields[2]);
      long timestamp = Long.parseLong(fields[3]);
      return new Rating(userId, movieId, rating, timestamp);
    }
  }
  // $example off$

  public static void main(String[] args) {
    SparkSession spark = SparkSession
      .builder()
      .appName("movie_recommend_ALS")
      .config("spark.master", "local")
      .getOrCreate();

    // $example on$
    String PATH = "/Users/yennanliu/movie_recommendation/datasets/ml-latest-small/ratings.csv";
    //JavaRDD<Rating> ratingsRDD = spark.read().textFile(PATH).javaRDD()
    Dataset<Row> ratings = spark.read().option("header", "true").csv(PATH);
    Dataset<Row> ratings_ = ratings.withColumn("userId", ratings.col("userId").cast("double"))
                                   .withColumn("movieId", ratings.col("movieId").cast("double"))
                                   .withColumn("rating", ratings.col("rating").cast("double")); 

    System.out.println("----------------- print dataset ----------------- ");
    ratings_.show(); 
    System.out.println("----------------- print dataset ----------------- ");
    Dataset<Row>[] splits = ratings_.randomSplit(new double[]{0.8, 0.2});
    Dataset<Row> training = splits[0];
    Dataset<Row> test = splits[1];

    // Build the recommendation model using ALS on the training data
    ALS als = new ALS()
      .setMaxIter(5)
      .setRegParam(0.01)
      .setUserCol("userId")
      .setItemCol("movieId")
      .setRatingCol("rating");
    ALSModel model = als.fit(training);

    // Evaluate the model by computing the RMSE on the test data
    // Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    model.setColdStartStrategy("drop");
    Dataset<Row> predictions = model.transform(test);

    RegressionEvaluator evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction");
    Double rmse = evaluator.evaluate(predictions);
    System.out.println("Root-mean-square error = " + rmse);

    // Generate top 10 movie recommendations for each user
    Dataset<Row> userRecs = model.recommendForAllUsers(10);
    // Generate top 10 user recommendations for each movie
    Dataset<Row> movieRecs = model.recommendForAllItems(10);

    // Generate top 10 movie recommendations for a specified set of users
    Dataset<Row> users = ratings_.select(als.getUserCol()).distinct().limit(15);
    Dataset<Row> userSubsetRecs = model.recommendForUserSubset(users, 10);
    // Generate top 10 user recommendations for a specified set of movies
    Dataset<Row> movies = ratings_.select(als.getItemCol()).distinct().limit(15);
    Dataset<Row> movieSubSetRecs = model.recommendForItemSubset(movies, 10);
    // $example off$
    userRecs.show();
    movieRecs.show();
    userSubsetRecs.show();
    movieSubSetRecs.show();

    spark.stop();
  }
}