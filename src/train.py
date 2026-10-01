from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import time


DATA_PATH = "hdfs://namenode:9000/uber/raw/fhvhv_tripdata_2021-01.parquet"

RF_MODEL_PATH = "hdfs://namenode:9000/uber/models/random_forest"
GBT_MODEL_PATH = "hdfs://namenode:9000/uber/models/gradient_boosted_trees"

FEATURE_COLUMNS = [
    "trip_miles",
    "PULocationID",
    "DOLocationID",
    "pickup_hour",
    "pickup_weekday",
]


def metrics(predictions):
    rmse = RegressionEvaluator(
        labelCol="trip_time",
        predictionCol="prediction",
        metricName="rmse"
    ).evaluate(predictions)

    mae = RegressionEvaluator(
        labelCol="trip_time",
        predictionCol="prediction",
        metricName="mae"
    ).evaluate(predictions)

    r2 = RegressionEvaluator(
        labelCol="trip_time",
        predictionCol="prediction",
        metricName="r2"
    ).evaluate(predictions)

    return rmse, mae, r2


def main():
    spark = (
        SparkSession.builder
        .appName("NYC HVFHV Model Validation")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")


    # Load + prepare data
    print("Loading data...")

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

    df = df.limit(500_000)

    print(f"Records used: {df.count():,}")


    # Train / validation / test
    train_df, validation_df, test_df = df.randomSplit(
        [0.70, 0.15, 0.15],
        seed=42
    )

    print(f"Training:   {train_df.count():,}")
    print(f"Validation: {validation_df.count():,}")
    print(f"Testing:    {test_df.count():,}")

    assembler = VectorAssembler(
        inputCols=FEATURE_COLUMNS,
        outputCol="features"
    )

    train_data = assembler.transform(train_df).cache()
    validation_data = assembler.transform(validation_df).cache()
    test_data = assembler.transform(test_df).cache()

    # Random Forest parameter experiments
    rf_configs = [
        {"numTrees": 20, "maxDepth": 5},
        {"numTrees": 50, "maxDepth": 10},
        {"numTrees": 100, "maxDepth": 10},
    ]

    best_rf_model = None
    best_rf_rmse = float("inf")
    best_rf_config = None

    print("\n=== RANDOM FOREST VALIDATION ===")

    for config in rf_configs:
        model = RandomForestRegressor(
            featuresCol="features",
            labelCol="trip_time",
            numTrees=config["numTrees"],
            maxDepth=config["maxDepth"],
            seed=42
        )

        start = time.time()
        fitted = model.fit(train_data)
        training_time = time.time() - start

        predictions = fitted.transform(validation_data)
        rmse, mae, r2 = metrics(predictions)

        print(
            f"trees={config['numTrees']:<3} "
            f"depth={config['maxDepth']:<2} | "
            f"RMSE={rmse:.2f} | "
            f"MAE={mae:.2f} | "
            f"R²={r2:.4f} | "
            f"time={training_time:.2f}s"
        )

        if rmse < best_rf_rmse:
            best_rf_rmse = rmse
            best_rf_model = fitted
            best_rf_config = config

    print(f"\nBest RF configuration: {best_rf_config}")


    # GBT parameter experiments
    gbt_configs = [
        {"maxIter": 10, "maxDepth": 5},
        {"maxIter": 30, "maxDepth": 5},
        {"maxIter": 50, "maxDepth": 5},
    ]

    best_gbt_model = None
    best_gbt_rmse = float("inf")
    best_gbt_config = None

    print("\n=== GBT VALIDATION ===")

    for config in gbt_configs:
        model = GBTRegressor(
            featuresCol="features",
            labelCol="trip_time",
            maxIter=config["maxIter"],
            maxDepth=config["maxDepth"],
            stepSize=0.1,
            seed=42
        )

        start = time.time()
        fitted = model.fit(train_data)
        training_time = time.time() - start

        predictions = fitted.transform(validation_data)
        rmse, mae, r2 = metrics(predictions)

        print(
            f"iterations={config['maxIter']:<3} "
            f"depth={config['maxDepth']} | "
            f"RMSE={rmse:.2f} | "
            f"MAE={mae:.2f} | "
            f"R²={r2:.4f} | "
            f"time={training_time:.2f}s"
        )

        if rmse < best_gbt_rmse:
            best_gbt_rmse = rmse
            best_gbt_model = fitted
            best_gbt_config = config

    print(f"\nBest GBT configuration: {best_gbt_config}")


    # Final test evaluation
    print("\n=== FINAL TEST RESULTS ===")

    rf_predictions = best_rf_model.transform(test_data)
    rf_rmse, rf_mae, rf_r2 = metrics(rf_predictions)

    gbt_predictions = best_gbt_model.transform(test_data)
    gbt_rmse, gbt_mae, gbt_r2 = metrics(gbt_predictions)

    print(
        f"Random Forest | "
        f"RMSE={rf_rmse:.2f} | "
        f"MAE={rf_mae:.2f} | "
        f"R²={rf_r2:.4f}"
    )

    print(
        f"GBT           | "
        f"RMSE={gbt_rmse:.2f} | "
        f"MAE={gbt_mae:.2f} | "
        f"R²={gbt_r2:.4f}"
    )


    # Save selected models
    print("\nSaving best models to HDFS...")

    best_rf_model.write().overwrite().save(RF_MODEL_PATH)
    best_gbt_model.write().overwrite().save(GBT_MODEL_PATH)

    print(f"RF:  {RF_MODEL_PATH}")
    print(f"GBT: {GBT_MODEL_PATH}")

    train_data.unpersist()
    validation_data.unpersist()
    test_data.unpersist()

    spark.stop()


if __name__ == "__main__":
    main()