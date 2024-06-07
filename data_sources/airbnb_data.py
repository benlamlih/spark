from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    when,
    lit,
    format_number,
    avg,
    regexp_replace,
    countDistinct,
    sum,
)
import math


spark = SparkSession.builder.appName("AirbnbListings").getOrCreate()

file_path = "datasets/listings.csv"
df = (
    spark.read.option("header", "true")
    .option("multiLine", "true")
    .option("escape", '"')
    .option("encoding", "UTF-8")
    .csv(file_path)
)

total_listings = df.count()
print(f"Total number of listings: {total_listings}")

room_type_counts = df.groupBy("room_type").count()
room_type_analysis = room_type_counts.withColumn(
    "Percentage", (col("count") / total_listings) * 100
)

room_type_analysis = room_type_analysis.select(
    "room_type",
    format_number("Percentage", 1).alias("Percentage"),
    col("count").alias("Count"),
)

room_type_analysis.show(truncate=False)

average_length_of_stay = round(df.agg(avg(col("minimum_nights_avg_ntm"))).first()[0], 2)

df = df.withColumn("estimated_bookings", col("number_of_reviews_ltm") / 0.50)

df = df.withColumn(
    "estimated_nights_booked",
    col("estimated_bookings") * col("minimum_nights_avg_ntm"),
)

average_nights_booked = df.select(
    avg("estimated_nights_booked").alias("average_nights_booked")
)

average_nights_booked.show()

df = df.withColumn("price", regexp_replace(col("price"), r",", ""))
df = df.withColumn("price", regexp_replace(col("price"), r"\$", ""))
df = df.withColumn(
    "price",
    when(col("price").isNull() | (col("price") == ""), lit(0)).otherwise(col("price")),
)
df = df.withColumn("price", col("price").cast("float"))

total_price = df.agg(sum("price").alias("total_price")).collect()[0]["total_price"]
average_price = total_price / total_listings

print(f"Average price: {math.ceil(average_price)}")

print(f"Average Income: {math.ceil(average_price * average_nights_booked.first()[0])}£")

df = df.withColumn("estimated_income", col("estimated_nights_booked") * col("price"))
df = df.withColumn("minimum_nights", col("minimum_nights").cast("int"))

df = df.withColumn(
    "minimum_nights",
    when(col("minimum_nights").isNull(), lit(1)).otherwise(col("minimum_nights")),
)

# if maximum_minimum_nights is greater than 30 means that the maximum (minimum nights) is greater than 30
# so it is a long term rentals
long_term_rentals_count = df.filter(col("maximum_minimum_nights") > 30).count()
short_term_rentals_count = total_listings - long_term_rentals_count

short_term_rentals_percent = (short_term_rentals_count / total_listings) * 100
long_term_rentals_percent = (long_term_rentals_count / total_listings) * 100

print(
    f"Short-term rentals: {short_term_rentals_count} ({short_term_rentals_percent:.2f}%)"
)
print(
    f"Long-term rentals: {long_term_rentals_count} ({long_term_rentals_percent:.2f}%)"
)


multi_listing_hosts = df.filter(col("host_listings_count") > 1)

# If the host has more than 1 listing at the same location, it is considered a single lister
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

multi = total_listings - single_listers_count
multi_listings_percent = ((multi) / total_listings) * 100

print(f"Single-listings: {multi} ({multi_listings_percent:.2f}%)")
print(f"Multi-listings: {single_listers_count} ({single_listings_percent:.2f}%)")


# Convert necessary columns to numeric
df = df.withColumn(
    "calculated_host_listings_count", col("calculated_host_listings_count").cast("int")
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

host_summary.show(truncate=False)
spark.stop()
