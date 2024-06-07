from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, split, explode, regexp_extract, avg

spark = SparkSession.builder.appName("NetflixAnalysis").getOrCreate()

netflix_df = (
    spark.read.option("header", "true")
    .option("quote", '"')
    .option("escape", '"')
    .csv("datasets/netflix_titles.csv", encoding="latin1")
)

movies_df = netflix_df.filter(col("type") == "Movie")

directors_count = (
    movies_df.filter(col("director").isNotNull())
    .groupBy("director")
    .agg(count("title").alias("movie_count"))
    .orderBy(col("movie_count").desc())
)

print("Les réalisateurs les plus prolifiques et leur nombre de films respectifs :")
directors_count.show()

country_df = netflix_df.filter(col("country").isNotNull()).withColumn(
    "country", explode(split(col("country"), ", "))
)

country_count = country_df.groupBy("country").agg(count("show_id").alias("count"))

total_count = netflix_df.filter(col("country").isNotNull()).count()
country_percentage = country_count.withColumn(
    "percentage", (col("count") / total_count) * 100
).orderBy(col("percentage").desc())

print("Pourcentages des films/séries produits par pays :")
country_percentage.show()

duration_df = movies_df.filter(col("duration").isNotNull()).withColumn(
    "duration_min", regexp_extract(col("duration"), r"(\d+)", 1).cast("int")
)

average_duration = duration_df.select(avg("duration_min")).collect()[0][0]

longest_movie = (
    duration_df.orderBy(col("duration_min").desc())
    .select("title", "duration_min")
    .first()
)
shortest_movie = (
    duration_df.orderBy(col("duration_min").asc())
    .select("title", "duration_min")
    .first()
)

print("Durée moyenne des films :", average_duration)
print("Film le plus long :", longest_movie)
print("Film le plus court :", shortest_movie)

duration_df = duration_df.filter(col("release_year").isNotNull()).withColumn(
    "interval", (col("release_year") - (col("release_year") % 2)).cast("int")
)

interval_avg_duration = (
    duration_df.groupBy("interval")
    .agg(avg("duration_min").alias("avg_duration"))
    .orderBy(col("interval").desc())
)

print("Durée moyenne des films par intervalles de 2 ans :")
interval_avg_duration.show()

actor_df = movies_df.filter(
    col("cast").isNotNull() & col("director").isNotNull()
).withColumn("actor", explode(split(col("cast"), ", ")))

director_actor_count = (
    actor_df.groupBy("director", "actor")
    .agg(count("title").alias("movie_count"))
    .orderBy(col("movie_count").desc())
)

print("Le duo réalisateur-acteur le plus prolifique :")
director_actor_count.show()
