import pandas as pd
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Construct SparkSession instance
# (Combines SparkContext, SQLContext and HiveContext)
spark = SparkSession.builder.master("local").appName("CS5052") \
    .config("spark.hadoop.conf-key", "spark.hadoop.conf-value") \
    .getOrCreate()

df_movies = None
df_ratings = None


# Set the dataset to the small directory
def setDataSetToSmall():
    global df_movies
    df_movies = spark.read \
        .options(header=True) \
        .csv("ml-latest-small/movies.csv")
    global df_ratings
    df_ratings = spark.read \
        .options(header=True, inferSchema=True) \
        .csv("ml-latest-small/ratings.csv")


# Set the dataset to the large directory
def setDataSetToLarge():
    global df_movies
    df_movies = spark.read \
        .options(header=True) \
        .csv("ml-latest/movies.csv")
    global df_ratings
    df_ratings = spark.read \
        .options(header=True, inferSchema=True) \
        .csv("ml-latest/ratings.csv")


def setDataSet(useSmallDataSet):
    if (useSmallDataSet == True):
        setDataSetToSmall()
    else:
        setDataSetToLarge()


# Get movie data
def movies_id(id, useSmallDataSet):
    """Takes a movie ID and returns the average rating and number of users who have watched the movie."""
    setDataSet(useSmallDataSet)
    innerJoin = df_movies.join(df_ratings, on=['movieID'], how='inner')
    filtered_innerJoin = innerJoin.where(innerJoin["movieID"] == id)
    return filtered_innerJoin.groupBy("movieID").agg(F.first('title').alias('Movie Title'),
                                                     F.round(F.mean('rating'), 2).alias('Average Rating'),
                                                     F.count('userID').alias('User Count'))


# Get movie data
def movies_title(title, useSmallDataSet):
    """Takes a movie ID and returns the average rating and number of users who have watched the movie."""
    setDataSet(useSmallDataSet)
    innerJoin = df_movies.join(df_ratings, on=['movieID'], how='inner')
    filtered_innerJoin = innerJoin.where(innerJoin["title"] == title)
    return filtered_innerJoin.groupBy("movieID").agg(F.first('title').alias('Movie Title'),
                                                     F.round(F.mean('rating'), 2).alias('Average Rating'),
                                                     F.count('userID').alias('User Count'))


def genre_movies(genre, useSmallDataSet):
    """Takes a genre and outputs all movies in that genre."""
    setDataSet(useSmallDataSet)
    genres = list(map(str, genre.split(",")))
    movies = df_movies
    genre_list = movies.select(F.explode(F.split(movies['genres'], "[|]")).alias("Genres"), movies['title'])
    filtered_genre_list = genre_list.where(genre_list["Genres"].isin(genres))
    return filtered_genre_list.orderBy("Genres")


def movies_year(year, useSmallDataSet):
    """Takes a year and returns all movies released that year."""
    setDataSet(useSmallDataSet)
    return df_movies.filter(df_movies.title.contains(year)) \
        .select(F.col('title').alias('Title'))


def highest_rating(n, useSmallDataSet):
    setDataSet(useSmallDataSet)
    n = int(n)
    """Outputs n highest-rated movies."""
    df_ratings_grouped = df_ratings.groupby(['movieId']) \
        .agg(F.mean(df_ratings.rating).alias('avg_rating'))

    return df_ratings_grouped.alias('r') \
        .join(df_movies.alias('m'), F.col('r.movieId') == F.col('m.movieId')) \
        .orderBy(F.col('avg_rating').desc()) \
        .select(F.col('m.title').alias('Title'), F.col('avg_rating').alias('Rating')).limit(n)


def highest_views(n, useSmallDataSet):
    setDataSet(useSmallDataSet)
    n = int(n)
    """Outputs n most-watched movies."""
    df_ratings_grouped = df_ratings.groupby(['movieId']) \
        .agg(F.count(df_ratings.userId).alias('views'))

    return df_ratings_grouped.alias('r') \
        .join(df_movies.alias('m'), F.col('r.movieId') == F.col('m.movieId')) \
        .orderBy(F.col('views').desc()) \
        .select(F.col('m.title').alias('Title'), F.col('views').alias('Views')).limit(n)


def user_history(user_id, useSmallDataSet):
    """Takes a user ID and outputs statistics in the form of number movies
    watched (rated) by that user and the number of distinct genres represented
    by those movies as well as a table of the movies and their genres.
    """
    setDataSet(useSmallDataSet)
    innerJoin = df_movies.join(df_ratings, on=['movieID'], how='inner')
    filtered_innerJoin = innerJoin.where(innerJoin["userID"] == user_id)
    user_movies_watched = filtered_innerJoin.groupBy("userID").agg(
        F.countDistinct('movieID').alias('Count of Movies Watched'))

    innerJoinGenres = df_movies.join(df_ratings, on=['movieID'], how='inner')
    filtered_innerJoinGenres = innerJoinGenres.where(innerJoinGenres["userID"] == user_id)
    genre_list = filtered_innerJoinGenres.select(filtered_innerJoinGenres['userID'],
                                                 F.explode(F.split(filtered_innerJoinGenres['genres'], "[|]")).alias(
                                                     "Genres"))
    user_genre_list = genre_list.groupBy("userID").agg(F.countDistinct('Genres').alias('Count of Genres Watched'))

    combined_list = user_movies_watched.join(user_genre_list,
                                             user_movies_watched['userID'] == user_genre_list['userID'], 'inner')
    return combined_list


def movies_watched(user_id, useSmallDataSet):
    setDataSet(useSmallDataSet)
    """Takes a user ID and outputs movies watched (rated) by that user."""
    user_ids = list(map(int, user_id.split(",")))
    innerJoin = df_movies.join(df_ratings, on=['movieID'], how='inner')
    filtered_innerJoin = innerJoin.where(innerJoin["userID"].isin(user_ids))
    movies_list = filtered_innerJoin.select(filtered_innerJoin["userID"], filtered_innerJoin["title"])
    return movies_list.orderBy("userID")


def genre_list(user_id, useSmallDataSet):
    setDataSet(useSmallDataSet)
    innerJoin = df_movies.join(df_ratings, on=['movieID'], how='inner')
    filtered_innerJoin = innerJoin.where(innerJoin["userID"] == user_id)
    genre_list = filtered_innerJoin.select(filtered_innerJoin['rating'],
                                           F.explode(F.split(filtered_innerJoin['genres'], "[|]")).alias(
                                               "Genres")).groupBy("genres").agg(F.sum("rating").alias("Sum of Ratings"))
    # genre_list.orderBy('Sum of Ratings', ascending=False).limit(1).show()
    return genre_list.orderBy('Sum of Ratings', ascending=False)


def genre_list_multiple_users(user_id, useSmallDataSet):
    setDataSet(useSmallDataSet)
    user_ids = list(map(int, user_id.split(",")))
    innerJoin = df_movies.join(df_ratings, on=['movieID'], how='inner')
    filtered_innerJoin = innerJoin.where(innerJoin["userID"].isin(user_ids))
    genre_list = filtered_innerJoin.select(filtered_innerJoin['userID'], filtered_innerJoin['rating'],
                                           F.explode(F.split(filtered_innerJoin['genres'], "[|]")).alias(
                                               "Genres")).groupBy("genres").agg(F.sum("rating").alias("Sum of Ratings"))
    return genre_list.orderBy('Sum of Ratings', ascending=False).limit(1)


def user_comparison(user_a, user_b, useSmallDataSet):
    """Takes two users and outputs the percentage overlap between their watch
    history.
    """
    setDataSet(useSmallDataSet)
    movies_a = []
    for rating in df_ratings \
            .filter("userId == '" + user_a + "'") \
            .select("movieId").collect():
        movies_a.append(rating.movieId)

    movies_b = []
    for rating in df_ratings \
            .filter("userId == '" + user_b + "'") \
            .select("movieId").collect():
        movies_b.append(rating.movieId)

    # Number of movies watched by both users
    num_both_watched = len(set(movies_a).intersection(movies_b))
    # Take average no. total movies watched between the two users as base
    base = (len(movies_a) + len(movies_b)) / 2
    # Overlap as percentage
    overlap = round((num_both_watched / base) * 100, 3)

    print("\nOverlap: {}%\n".format(overlap))


def cluster():
    # Load ratings.csv using pandas
    df = pd.read_csv("ml-latest-small/ratings.csv")

    # Get distinct users from original DataFrame...
    users = pd.DataFrame(data=df.userId.unique().flatten())

    # Pivot using pandas in order to define a feature vector for each user
    df_pivot = df.pivot(values='rating', index='userId', columns='movieId')

    # String list of input columns to be passed to VectorAssembler
    list_cols = [str(i) for i in list(df_pivot)]

    # Convert from pandas to PySpark DataFrame and replace NaNs with zeros
    spark_df = spark.createDataFrame(df_pivot).na.fill(0)

    # Transform into single feature vector per user
    vec_assembler = VectorAssembler(inputCols=list_cols, outputCol='features')
    new_df = vec_assembler.transform(spark_df)

    # Fit KMeans with 5 clusters
    k_means = KMeans(k=5)
    model = k_means.fit(new_df.select('features'))

    # Include cluster assignments and...
    transformed = model.transform(new_df)
    # ...convert to pandas
    pred_pd = transformed.select('prediction').toPandas()

    # Output concatenated pandas DataFrame
    df_concat = pd.concat([users, pred_pd], axis=1)
    df_concat.rename(columns={0: 'userId', 'prediction': 'Cluster'}, inplace=True)
    print(df_concat.to_string(index=False))


def user_recommendation(user_id, n, useSmallDataSet):
    """Takes a user ID and integer n and outputs the recommendation model's n
    top recommendations for the user, sorted by the 'Recommendation Score', the
    model's predicted target variable. Note that the Recommendation Score does
    not correspond to the MovieLens 5-star rating scale and that, if the model
    is retrained with every query, recommendations will vary.
    """
    setDataSet(useSmallDataSet)

    # Split data into training and test set
    (training, test) = df_ratings.randomSplit([0.8, 0.2])

    # Build model using alternating least squares (ALS) on training set
    als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId",
              ratingCol="rating", coldStartStrategy="drop")
    model = als.fit(training)

    # Evaluate model by root-mean-square error (RMSE) on test set
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    '''Uncomment below to output RMSE'''
    # print("RMSE: " + str(rmse))

    user_recs = model.recommendForAllUsers(n)
    user_recs.filter("userId == '" + user_id + "'") \
        .select('userId',
                F.explode('recommendations').alias('recs'),
                F.col('recs.movieId'),
                F.col('recs.rating')).alias('r') \
        .join(df_movies.alias('m'), F.col('r.movieId') == F.col('m.movieId')) \
        .select(F.col('title').alias('Title'),
                F.col('r.recs.rating').alias('Recommendation Score')) \
        .show(n=n, truncate=False)
