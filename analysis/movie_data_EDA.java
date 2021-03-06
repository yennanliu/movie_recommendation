import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.*;

import static org.apache.spark.sql.functions.avg;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.max;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import scala.Tuple2;

import org.apache.commons.lang.StringUtils;
import java.util.Map;

// UDF 
import commons.Utils;
import commons.AvgCount;


public class movie_data_EDA {

	private static final String  RATING_MIDPOINT = "rating";
  private static final String TIMESTAMP_MIDPOINT = "timestamp";

	public static void main(String[] args) throws Exception {

        Logger.getLogger("org").setLevel(Level.ERROR);


//-----------------------------------------------------------------------------------------------
        System.out.println("******************* PART 1) SPARK SQL ******************* ");
        SparkSession session = SparkSession.builder().appName("MovieDataSurvey").master("local[1]").getOrCreate();

        DataFrameReader dataFrameReader = session.read();
        Dataset<Row> responses = dataFrameReader.option("header","true").csv("../datasets/ml-latest-small/ratings.csv");
        Dataset<Row> movies = dataFrameReader.option("header","true").csv("../datasets/ml-latest-small/movies.csv");

        System.out.println("=== Print out schema ===");
        responses.printSchema();
        movies.printSchema();

        System.out.println("=== Print 20 records of responses table ===");
        responses.show(20);
        movies.show(20);

        System.out.println("=== Print the movieId and rating columns of gender table ===");
        responses.select(col("movieId"),  col("rating")).show();

        System.out.println("=== Print records where the userId is equal 1 ===");
        responses.filter(col("userId").equalTo("1")).show();

        System.out.println("=== Print the count of userId ===");
        RelationalGroupedDataset groupedDataset = responses.groupBy(col("userId"));
        groupedDataset.count().show();

        System.out.println("=== Cast the salary mid point and age mid point to integer ===");
        Dataset<Row> castedResponse = responses.withColumn(RATING_MIDPOINT, col(RATING_MIDPOINT).cast("integer"))
                                               .withColumn(TIMESTAMP_MIDPOINT, col(TIMESTAMP_MIDPOINT).cast("integer"));

        System.out.println("=== Print out casted schema ===");
        castedResponse.printSchema();


//-----------------------------------------------------------------------------------------------
        System.out.println("******************* PART 2) SPARK RDD ******************* ");
        System.out.println("=== Spark flat map ===");

        Logger.getLogger("org").setLevel(Level.ERROR);
        SparkConf conf = new SparkConf().setAppName("movies").setMaster("local").set("spark.driver.allowMultipleContexts", "true");
        JavaSparkContext sc = new JavaSparkContext(conf);
        JavaRDD<String> ratings = sc.textFile("../datasets/ml-latest-small/ratings.csv");

        System.out.println(ratings.take(30));

        System.out.println("=== movie-rating RDD string ===");

        JavaRDD < String > movie_id_ratings = ratings.map(line -> {
                            String[] splits = line.split(Utils.COMMA_DELIMITER);
                            return StringUtils.join(new String[] {
                            splits[1], splits[2] }, ",");
                        }
                );

        System.out.println(movie_id_ratings.take(30));

        System.out.println("=== pair RDD ===");

        JavaPairRDD<String, String> moviePairRDD = ratings.mapToPair(getMovieAndRatingPair());

        System.out.println(moviePairRDD.take(30));

        System.out.println("=== PairRDD reduce by key ===");

        JavaRDD<String> cleanedLines = ratings.filter(line -> !line.contains("movieId"));

        JavaPairRDD<String, AvgCount> movieratingPairRdd = cleanedLines.mapToPair(
        line -> new Tuple2<>(line.split(",")[1],
                new AvgCount(1, Double.parseDouble(line.split(",")[2]))));


        System.out.println(movieratingPairRdd.take(30));


        JavaPairRDD<String, AvgCount> movieratingTotal = movieratingPairRdd.reduceByKey(
         (x, y) -> new AvgCount(x.getCount() + y.getCount(), x.getTotal() + y.getTotal()));

        System.out.println("movieratingTotal: ");
        for (Map.Entry<String, AvgCount> movieratingTotalPair : movieratingTotal.collectAsMap().entrySet()) {
            System.out.println(movieratingTotalPair.getKey() + " : " + movieratingTotalPair.getValue());
        }

        JavaPairRDD<String, Double> movieratingAvg = movieratingTotal.mapValues(avgCount -> avgCount.getTotal()/avgCount.getCount());
        
        System.out.println("housePriceAvg: ");
        for (Map.Entry<String, Double> movieratingAvgPair : movieratingAvg.collectAsMap().entrySet()) {
            System.out.println(movieratingAvgPair.getKey() + " : " + movieratingAvgPair.getValue());
        }



	}

    private static PairFunction<String, String, String> getMovieAndRatingPair() {
    return (PairFunction<String, String, String>) line -> new Tuple2<>(line.split(Utils.COMMA_DELIMITER)[1],
                                                                           line.split(Utils.COMMA_DELIMITER)[2]);
    }

}




