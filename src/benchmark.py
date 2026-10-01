from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
import time


DATA_PATH = "hdfs://namenode:9000/uber/raw/fhvhv_tripdata_2021-01.parquet"

FEATURE_COLUMNS = [
    "trip_miles",
    "PULocationID",
    "DOLocationID",
    "pickup_hour",
    "pickup_weekday",
]


def main():
    spark = (
        SparkSession.builder
        .appName("GBT Performance Benchmark")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(DATA_PATH)

    df = (
        df
        .withColumn("pickup_hour", hour("pickup_datetime"))
        .withColumn("pickup_weekday", dayofweek("pickup_datetime"))
        .select(*FEATURE_COLUMNS, "trip_time")
        .dropna()
    )

    df = df.filter(
        (col("trip_miles") >= 0.5)
        & (col("trip_miles") <= 100)
        & (col("trip_time") >= 60)
        & (col("trip_time") <= 4 * 60 * 60)
    )

    df = df.limit(2_000_000)

    assembler = VectorAssembler(
        inputCols=FEATURE_COLUMNS,
        outputCol="features"
    )

    data = assembler.transform(df).cache()

    # Force Spark to materialize/cache the data BEFORE timing ML training.
    count = data.count()
    print(f"\nRecords: {count:,}")

    # Random Forest
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="trip_time",
        numTrees=50,
        maxDepth=10,
        seed=42
    )

    print("\nTraining Random Forest...")

    start = time.time()
    rf.fit(data)
    rf_time = time.time() - start


    # Gradient-Boosted Trees
    gbt = GBTRegressor(
        featuresCol="features",
        labelCol="trip_time",
        maxIter=50,
        maxDepth=5,
        stepSize=0.1,
        seed=42
    )

    print("\nTraining GBT...")

    start = time.time()
    gbt.fit(data)
    gbt_time = time.time() - start


    # Results
    print("\n==============================")
    print("ML PERFORMANCE BENCHMARK")
    print("==============================")
    print(f"Records:       {count:,}")
    print(f"Random Forest: {rf_time:.2f} seconds")
    print(f"GBT:           {gbt_time:.2f} seconds")
    print("==============================")
    data.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()