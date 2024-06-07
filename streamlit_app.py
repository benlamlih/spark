import streamlit as st
import math
from pyspark.sql import SparkSession
import pandas as pd
from pyspark.sql.functions import (
    col,
    count,
    split,
    explode,
    regexp_extract,
    avg,
    lower,
    date_format,
    lit,
    concat,
    when,
    count,
    regexp_replace,
    countDistinct,
    sum,
    format_number,
)

# Configurer la page Streamlit
st.set_page_config(
    page_title="Analyse des Données",
    page_icon="📊",
    layout="wide",
)


# Créer ou récupérer la session Spark
@st.cache_resource
def get_spark_session():
    return SparkSession.builder.appName("AnalysisApp").getOrCreate()


spark = get_spark_session()


# Charger et traiter les données Netflix
@st.cache_data
def analyze_netflix():
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

    country_df = netflix_df.filter(col("country").isNotNull()).withColumn(
        "country", explode(split(col("country"), ", "))
    )

    country_count = country_df.groupBy("country").agg(count("show_id").alias("count"))

    total_count = netflix_df.filter(col("country").isNotNull()).count()
    country_percentage = country_count.withColumn(
        "percentage", (col("count") / total_count) * 100
    ).orderBy(col("percentage").desc())

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

    duration_df = duration_df.filter(col("release_year").isNotNull()).withColumn(
        "interval", (col("release_year") - (col("release_year") % 2)).cast("int")
    )

    interval_avg_duration = (
        duration_df.groupBy("interval")
        .agg(avg("duration_min").alias("avg_duration"))
        .orderBy(col("interval").desc())
    )

    actor_df = movies_df.filter(
        col("cast").isNotNull() & col("director").isNotNull()
    ).withColumn("actor", explode(split(col("cast"), ", ")))

    director_actor_count = (
        actor_df.groupBy("director", "actor")
        .agg(count("title").alias("movie_count"))
        .orderBy(col("movie_count").desc())
    )

    directors_count_pd = directors_count.toPandas()
    country_percentage_pd = country_percentage.toPandas()
    interval_avg_duration_pd = interval_avg_duration.toPandas()
    director_actor_count_pd = director_actor_count.toPandas()

    return (
        directors_count_pd,
        country_percentage_pd,
        average_duration,
        longest_movie,
        shortest_movie,
        interval_avg_duration_pd,
        director_actor_count_pd,
    )


# Charger et traiter les données Twitter
@st.cache_data
def analyze_trump_tweets():
    trump_tweets = (
        spark.read.option("header", "true")
        .option("quote", '"')
        .option("escape", '"')
        .csv("datasets/trump_insult_tweets_2014_to_2021.csv", encoding="latin1")
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

    biden_insults = (
        trump_tweets.filter((col("target") == "joe-biden") & col("insult").isNotNull())
        .groupBy("insult")
        .agg(count("*").alias("count"))
        .orderBy(col("count").desc())
        .limit(1)
    )

    mexico_count = trump_tweets.filter(col("tweet").contains("mexico")).count()
    china_count = trump_tweets.filter(col("tweet").contains("china")).count()
    coronavirus_count = trump_tweets.filter(
        col("tweet").contains("coronavirus")
    ).count()

    trump_tweets = trump_tweets.filter(col("date").isNotNull())

    trump_tweets = trump_tweets.withColumn(
        "period", concat(date_format(col("date"), "yyyy-MM"), lit("-01"))
    )

    tweets_by_period = (
        trump_tweets.groupBy("period").agg(count("*").alias("count")).orderBy("period")
    )

    # Convert to pandas DataFrames
    target_percentages_pd = target_percentages.toPandas()
    insult_percentages_pd = insult_percentages.toPandas()
    biden_insults_pd = biden_insults.toPandas()
    tweets_by_period_pd = tweets_by_period.toPandas()

    return (
        target_percentages_pd,
        insult_percentages_pd,
        biden_insults_pd,
        mexico_count,
        china_count,
        coronavirus_count,
        tweets_by_period_pd,
    )


# Analyser les données
(
    directors_count,
    country_percentage,
    average_duration,
    longest_movie,
    shortest_movie,
    interval_avg_duration,
    director_actor_count,
) = analyze_netflix()
(
    target_percentages,
    insult_percentages,
    biden_insults,
    mexico_count,
    china_count,
    coronavirus_count,
    tweets_by_period,
) = analyze_trump_tweets()

st.title("📊 Analyse des Données Netflix, Tweets de Donald Trump et Airbnb")
st.subheader("Mohammed Benlamlih - 2024")

# Personnalisation des couleurs pour les graphiques
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

st.header("🎬 Analyse Netflix")
st.subheader("Les réalisateurs les plus prolifiques")
st.dataframe(directors_count)

st.bar_chart(
    directors_count.set_index("director")["movie_count"],
    use_container_width=True,
    height=300,
)

st.subheader("Pourcentages des films/séries produits par pays")
st.dataframe(country_percentage)

st.bar_chart(
    country_percentage.set_index("country")["percentage"],
    use_container_width=True,
    height=300,
)

st.subheader("Durée des films")

st.metric(
    label="Durée moyenne des films",
    value="{:.2f} minutes".format(average_duration),
)

st.metric(
    label="Film le plus long",
    value="{} avec {} minutes".format(
        longest_movie["title"], longest_movie["duration_min"]
    ),
)

st.metric(
    label="Film le plus court",
    value="{} avec {} minutes".format(
        shortest_movie["title"], shortest_movie["duration_min"]
    ),
)

st.subheader("Durée moyenne des films par intervalles de 2 ans")
st.line_chart(
    interval_avg_duration.set_index("interval")["avg_duration"],
    use_container_width=True,
    height=300,
)

st.subheader("Le duo réalisateur-acteur le plus prolifique")
st.dataframe(director_actor_count.head(10))

st.header("🐦 Analyse des Tweets de Donald Trump")
st.subheader("Les 7 comptes que Donald Trump insulte le plus")
st.dataframe(target_percentages)

st.bar_chart(
    target_percentages.set_index("target")["percentage"],
    use_container_width=True,
    height=300,
)

st.subheader("Les insultes que Donald Trump utilise le plus")
st.dataframe(insult_percentages.head(20))

st.bar_chart(
    insult_percentages.set_index("insult")["percentage"],
    use_container_width=True,
    height=300,
)

st.subheader("L'insulte la plus utilisée pour Joe Biden")
st.dataframe(biden_insults)

st.subheader("Comptage de mots spécifiques dans les tweets")

st.metric(
    label="Nombre de tweets contenant 'Mexico':",
    value=mexico_count,
)

st.metric(
    label="Nombre de tweets contenant 'China':",
    value=china_count,
)


st.metric(
    label="Nombre de tweets contenant 'coronavirus':",
    value=coronavirus_count,
)

st.subheader("Nombre de tweets par période de 6 mois")
st.line_chart(
    tweets_by_period.set_index("period")["count"], use_container_width=True, height=300
)


# Airbnb Analysis
@st.cache_data
def analyze_airbnb():
    file_path = "datasets/listings.csv"
    df = (
        spark.read.option("header", "true")
        .option("multiLine", "true")
        .option("escape", '"')
        .option("encoding", "UTF-8")
        .csv(file_path)
    )

    total_listings = df.count()

    room_type_counts = df.groupBy("room_type").count()
    room_type_analysis = room_type_counts.withColumn(
        "Percentage", (col("count") / total_listings) * 100
    )

    room_type_analysis = room_type_analysis.select(
        "room_type",
        format_number("Percentage", 1).alias("Percentage"),
        col("count").alias("Count"),
    )

    average_length_of_stay = round(
        df.agg(avg(col("minimum_nights_avg_ntm"))).first()[0], 2
    )

    df = df.withColumn("estimated_bookings", col("number_of_reviews_ltm") / 0.50)

    df = df.withColumn(
        "estimated_nights_booked",
        col("estimated_bookings") * col("minimum_nights_avg_ntm"),
    )

    average_nights_booked = df.select(
        avg("estimated_nights_booked").alias("average_nights_booked")
    ).first()[0]

    df = df.withColumn("price", regexp_replace(col("price"), r",", ""))
    df = df.withColumn("price", regexp_replace(col("price"), r"\$", ""))
    df = df.withColumn(
        "price",
        when(col("price").isNull() | (col("price") == ""), lit(0)).otherwise(
            col("price")
        ),
    )
    df = df.withColumn("price", col("price").cast("float"))

    total_price = df.agg(sum("price").alias("total_price")).collect()[0]["total_price"]
    average_price = total_price / total_listings

    average_income = average_price * average_nights_booked

    df = df.withColumn(
        "estimated_income", col("estimated_nights_booked") * col("price")
    )
    df = df.withColumn("minimum_nights", col("minimum_nights").cast("int"))

    df = df.withColumn(
        "minimum_nights",
        when(col("minimum_nights").isNull(), lit(1)).otherwise(col("minimum_nights")),
    )

    long_term_rentals_count = df.filter(col("maximum_minimum_nights") > 30).count()
    short_term_rentals_count = total_listings - long_term_rentals_count

    short_term_rentals_percent = (short_term_rentals_count / total_listings) * 100
    long_term_rentals_percent = (long_term_rentals_count / total_listings) * 100

    multi_listing_hosts = df.filter(col("host_listings_count") > 1)

    host_location_group = multi_listing_hosts.groupBy(
        "host_id", "latitude", "longitude"
    ).agg(countDistinct("id").alias("unique_listings"))

    single_listers_at_same_location = host_location_group.filter(
        col("unique_listings") == 1
    )

    df = df.withColumn("host_listings_count", col("host_listings_count").cast("int"))
    df = df.withColumn("latitude", col("latitude").cast("double"))
    df = df.withColumn("longitude", col("longitude").cast("double"))

    df = df.withColumn(
        "host_listings_count",
        when(col("host_listings_count").isNull(), lit(-1)).otherwise(
            col("host_listings_count")
        ),
    )

    single_listers_count = single_listers_at_same_location.count()
    single_listings_percent = (single_listers_count / total_listings) * 100

    multi_listings_count = total_listings - single_listers_count
    multi_listings_percent = (multi_listings_count / total_listings) * 100

    # Convert necessary columns to numeric
    df = df.withColumn(
        "calculated_host_listings_count",
        col("calculated_host_listings_count").cast("int"),
    )
    df = df.withColumn(
        "calculated_host_listings_count_entire_homes",
        col("calculated_host_listings_count_entire_homes").cast("int"),
    )
    df = df.withColumn(
        "calculated_host_listings_count_private_rooms",
        col("calculated_host_listings_count_private_rooms").cast("int"),
    )
    df = df.withColumn(
        "calculated_host_listings_count_shared_rooms",
        col("calculated_host_listings_count_shared_rooms").cast("int"),
    )

    # Remove duplicates based on host_id
    unique_hosts_df = df.dropDuplicates(["host_id"])

    # Select relevant columns
    host_summary = unique_hosts_df.select(
        "host_name",
        "calculated_host_listings_count_entire_homes",
        "calculated_host_listings_count_private_rooms",
        "calculated_host_listings_count_shared_rooms",
        "calculated_host_listings_count",
    ).orderBy("calculated_host_listings_count", ascending=False)

    return (
        total_listings,
        room_type_analysis.toPandas(),
        average_nights_booked,
        average_price,
        average_income,
        short_term_rentals_count,
        short_term_rentals_percent,
        long_term_rentals_count,
        long_term_rentals_percent,
        multi_listings_count,
        multi_listings_percent,
        single_listers_count,
        single_listings_percent,
        host_summary.toPandas(),
    )


# Load and analyze Airbnb data
(
    total_listings,
    room_type_analysis,
    average_nights_booked,
    average_price,
    average_income,
    short_term_rentals_count,
    short_term_rentals_percent,
    long_term_rentals_count,
    long_term_rentals_percent,
    multi_listings_count,
    multi_listings_percent,
    single_listers_count,
    single_listings_percent,
    host_summary,
) = analyze_airbnb()

# Airbnb Data Analysis
st.title("🏡 Airbnb Data Analysis")


st.header("London")
st.metric(
    label="Total Listings",
    value=total_listings,
)

st.header("Room Type")

room_type_analysis = room_type_analysis.sort_values(by="Count", ascending=False)
col1, col2 = st.columns([1, 2])

# Display bar chart in the first column
with col1:
    st.bar_chart(
        room_type_analysis.set_index("room_type")["Count"], use_container_width=True
    )

with col2:
    for index, row in room_type_analysis.iterrows():
        room_type = row["room_type"]
        count = row["Count"]
        percentage = row["Percentage"]
        st.metric(
            label=f"{room_type}",
            value=f"{count} ({percentage}%)",
        )

st.header("Activity")

st.write(
    "The minimum stay, price and number of reviews have been used to estimate the number of nights booked and the income for each listing, for the last 12 months.\n"
    "Is the home, apartment or room rented frequently and displacing units of housing and residents? "
    "Does the income from Airbnb incentivize short-term rentals vs long-term housing?"
)

st.metric(label="Average nights booked", value=f"{average_nights_booked:.1f}")
st.metric(label="Average price/night", value=f"£{math.ceil(average_price)}")
st.metric(label="Average income", value=f"£{average_income:.2f}")

st.header("Short-Term Rentals")
# Because why not?
st.write(
    "The housing policies of cities and towns can be restrictive of short-term rentals, to protect housing for residents.\n"
    "By looking at the 'minimum nights' setting for listings, we can see if the market has shifted to longer-term stays. "
    "Was it to avoid regulations, or in response to changes in travel demands?\n\n"
    "In some cases, Airbnb has moved large numbers of their listings to longer-stays to avoid short-term rental regulations and accountability."
)

st.metric(
    label="Short-term rentals",
    value=f"{short_term_rentals_count} ({short_term_rentals_percent:.1f}%) short-term rentals",
)
st.metric(
    label="Long-term rentals",
    value=f"{long_term_rentals_count} ({long_term_rentals_percent:.1f}%) longer-term rentals",
)


st.header("Listings per Host")
st.write(
    "Some Airbnb hosts have multiple listings.\n"
    "A host may list separate rooms in the same apartment, or multiple apartments or homes available in their entirety.\n\n"
    "Hosts with multiple listings are more likely to be running a business, are unlikely to be living in the property, "
    "and in violation of most short-term rental laws designed to protect residential housing."
)

st.metric(
    label="Single-listings",
    value=f" {multi_listings_count} ({multi_listings_percent:.1f}%)",
)
st.metric(
    label="Multi-listings",
    value=f"{single_listers_count} ({single_listings_percent:.1f}%)",
)


# Show dataframes for details
st.subheader("Host Summary")
st.dataframe(host_summary)
spark.stop()
