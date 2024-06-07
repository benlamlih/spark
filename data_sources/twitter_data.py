from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, lower, date_format, lit, concat

spark = SparkSession.builder.appName("TrumpTweetAnalysis").getOrCreate()

trump_tweets = spark.read.option("header", "true").csv(
    "datasets/trump_insult_tweets_2014_to_2021.csv", encoding="latin1"
)

trump_tweets = trump_tweets.withColumn("date", col("date").cast("date"))

trump_tweets = trump_tweets.withColumn("tweet", lower(col("tweet")))

target_counts = (
    trump_tweets.filter(col("target").isNotNull())
    .groupBy("target")
    .agg(count("*").alias("count"))
    .orderBy(col("count").desc())
    .limit(7)
)

total_targets = trump_tweets.filter(col("target").isNotNull()).count()
target_percentages = target_counts.withColumn(
    "percentage", (col("count") / total_targets) * 100
)

print(
    "Les 7 comptes que Donald Trump insulte le plus (en nombres et en pourcentages) :"
)
target_percentages.show()

insult_counts = (
    trump_tweets.filter(col("insult").isNotNull())
    .groupBy("insult")
    .agg(count("*").alias("count"))
    .orderBy(col("count").desc())
)

total_insults = trump_tweets.filter(col("insult").isNotNull()).count()
insult_percentages = insult_counts.withColumn(
    "percentage", (col("count") / total_insults) * 100
)

print("Les insultes que Donald Trump utilise le plus (en nombres et en pourcentages) :")
insult_percentages.show()

biden_insults = (
    trump_tweets.filter((col("target") == "joe-biden") & col("insult").isNotNull())
    .groupBy("insult")
    .agg(count("*").alias("count"))
    .orderBy(col("count").desc())
    .limit(1)
)

print("L'insulte la plus utilisée pour Joe Biden :")
biden_insults.show()

mexico_count = trump_tweets.filter(col("tweet").contains("mexico")).count()
china_count = trump_tweets.filter(col("tweet").contains("china")).count()
coronavirus_count = trump_tweets.filter(col("tweet").contains("coronavirus")).count()

print("Nombre de tweets contenant 'Mexico':", mexico_count)
print("Nombre de tweets contenant 'China':", china_count)
print("Nombre de tweets contenant 'coronavirus':", coronavirus_count)

trump_tweets = trump_tweets.filter(col("date").isNotNull())

trump_tweets = trump_tweets.withColumn(
    "period", concat(date_format(col("date"), "yyyy-MM"), lit("-01"))
)

tweets_by_period = (
    trump_tweets.groupBy("period").agg(count("*").alias("count")).orderBy("period")
)

print("Nombre de tweets par période de 6 mois :")
tweets_by_period.show()
