from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import (
    RandomForestRegressionModel,
    GBTRegressionModel,
)
from pyspark.ml.evaluation import RegressionEvaluator
import time


DATA_PATH = "hdfs://namenode:9000/uber/raw/fhvhv_tripdata_2021-02.parquet"

RF_MODEL_PATH = "hdfs://namenode:9000/uber/models/random_forest"
GBT_MODEL_PATH = "hdfs://namenode:9000/uber/models/gradient_boosted_trees"

OUTPUT_PATH = "hdfs://namenode:9000/uber/predictions"

FEATURE_COLUMNS = [
    "trip_miles",
    "PULocationID",
    "DOLocationID",
    "pickup_hour",
    "pickup_weekday",
]


def evaluate(name, predictions):
    rmse = RegressionEvaluator(
        labelCol="trip_time",
        predictionCol="prediction",
        metricName="rmse",
    ).evaluate(predictions)

    mae = RegressionEvaluator(
        labelCol="trip_time",
        predictionCol="prediction",
        metricName="mae",
    ).evaluate(predictions)

    r2 = RegressionEvaluator(
        labelCol="trip_time",
        predictionCol="prediction",
        metricName="r2",
    ).evaluate(predictions)

    print(f"\n=== {name} ===")
    print(f"RMSE: {rmse:.2f} seconds")
    print(f"MAE:  {mae:.2f} seconds")
    print(f"R²:   {r2:.4f}")

    return rmse, mae, r2


def main():
    spark = (
        SparkSession.builder
        .appName("NYC HVFHV Trip Duration Prediction")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")


    # 1. Load February data
    print("Loading February data...")

    df = spark.read.parquet(DATA_PATH)


    # 2. Same feature engineering as training
    df = (
        df
        .withColumn("pickup_hour", hour("pickup_datetime"))
        .withColumn("pickup_weekday", dayofweek("pickup_datetime"))
        .select(
            *FEATURE_COLUMNS,
            "trip_time",
        )
        .dropna()
    )

    # Same cleaning rules as training
    df = df.filter(
        (col("trip_miles") >= 0.5)
        & (col("trip_miles") <= 100)
        & (col("trip_time") >= 60)
        & (col("trip_time") <= 4 * 60 * 60)
    )

    # Keep prediction experiment manageable
    df = df.limit(500_000)

    print(f"Prediction records: {df.count():,}")


    # 3. Assemble features
    assembler = VectorAssembler(
        inputCols=FEATURE_COLUMNS,
        outputCol="features",
    )

    data = assembler.transform(df)

    # 4. Load models from HDFS
    print("\nLoading models from HDFS...")

    rf_model = RandomForestRegressionModel.load(RF_MODEL_PATH)
    gbt_model = GBTRegressionModel.load(GBT_MODEL_PATH)

    print("Models loaded.")


    # 5. Random Forest predictions
    print("\nRunning Random Forest predictions...")

    start = time.time()
    rf_predictions = rf_model.transform(data)
    rf_predictions.cache()

    rf_metrics = evaluate(
        "RANDOM FOREST - FEBRUARY",
        rf_predictions,
    )

    rf_time = time.time() - start


    # 6. GBT predictions
    print("\nRunning GBT predictions...")

    start = time.time()
    gbt_predictions = gbt_model.transform(data)
    gbt_predictions.cache()

    gbt_metrics = evaluate(
        "GBT - FEBRUARY",
        gbt_predictions,
    )

    gbt_time = time.time() - start


    # 7. Show predictions
    print("\n=== GBT SAMPLE PREDICTIONS ===")

    gbt_predictions.select(
        "trip_miles",
        "PULocationID",
        "DOLocationID",
        "pickup_hour",
        "pickup_weekday",
        "trip_time",
        "prediction",
    ).show(10, truncate=False)


    # 8. Comparison
    print("\n=== FEBRUARY MODEL COMPARISON ===")

    print(
        f"{'Model':<20}"
        f"{'RMSE':>12}"
        f"{'MAE':>12}"
        f"{'R²':>12}"
        f"{'Prediction':>15}"
    )

    print("-" * 71)

    print(
        f"{'Random Forest':<20}"
        f"{rf_metrics[0]:>12.2f}"
        f"{rf_metrics[1]:>12.2f}"
        f"{rf_metrics[2]:>12.4f}"
        f"{rf_time:>14.2f}s"
    )

    print(
        f"{'GBT':<20}"
        f"{gbt_metrics[0]:>12.2f}"
        f"{gbt_metrics[1]:>12.2f}"
        f"{gbt_metrics[2]:>12.4f}"
        f"{gbt_time:>14.2f}s"
    )


    # 9. Save GBT predictions to HDFS
    print("\nSaving predictions...")

    output = gbt_predictions.select(
        "trip_miles",
        "PULocationID",
        "DOLocationID",
        "pickup_hour",
        "pickup_weekday",
        "trip_time",
        "prediction",
    )

    output.write \
        .mode("overwrite") \
        .option("header", True) \
        .csv(f"{OUTPUT_PATH}/gbt_february")

    print(
        f"Predictions saved to "
        f"{OUTPUT_PATH}/gbt_february"
    )

    rf_predictions.unpersist()
    gbt_predictions.unpersist()

    spark.stop()


if __name__ == "__main__":
    main()