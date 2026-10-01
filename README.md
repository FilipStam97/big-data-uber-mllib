# Big Data Systems – Project 3

Machine learning analysis of the NYC High Volume For-Hire Vehicle (HVFHV) dataset using Apache Spark MLlib.

The project trains and evaluates **Random Forest** and **Gradient-Boosted Trees (GBT)** regression models for predicting trip duration.

## Technologies

- Apache Spark 4.0.2
- PySpark / MLlib
- Hadoop HDFS
- Docker Compose
- Python
- Pandas
- Matplotlib

## Architecture

The Docker environment contains:

- 1 Spark Master
- 2 Spark Workers
- 1 Hadoop NameNode
- 1 Hadoop DataNode
- 1 Spark application container

Spark applications connect to:

```text
spark://spark-master:7077
```

HDFS is available internally at:

```text
hdfs://namenode:9000
```

## Dataset

NYC High Volume For-Hire Vehicle Trip Records (2021).

Training and model selection use January 2021 data.

February 2021 is used as unseen temporal data for final prediction analysis.

### Target

The regression target is:

```text
trip_time
```

Trip duration is measured in seconds.

### Features

The following features are used:

```text
trip_miles
PULocationID
DOLocationID
pickup_hour
pickup_weekday
```

`pickup_hour` and `pickup_weekday` are derived from `pickup_datetime`.

`dropoff_datetime` is intentionally not used because it would introduce target leakage.

## Data Cleaning

Trips are retained when:

```text
0.5 <= trip_miles <= 100
60 <= trip_time <= 14400
```

This removes extremely short/invalid trips and extreme duration/distance values.

For model development, 500,000 cleaned January records are used.

The dataset is split into:

```text
70% training
15% validation
15% testing
```

with a fixed random seed.

## Model Training

### Random Forest

Tested configurations:

| Trees | Max Depth | Validation RMSE | Validation MAE | Validation R² | Training Time |
|---:|---:|---:|---:|---:|---:|
| 20 | 5 | 291.22 | 200.69 | 0.7043 | 6.81 s |
| 50 | 10 | 255.14 | 164.64 | 0.7730 | 20.87 s |
| 100 | 10 | **254.69** | **164.40** | **0.7738** | 52.09 s |

Selected configuration:

```text
100 trees
maxDepth = 10
```

Increasing from 50 to 100 trees produced only a small accuracy improvement while substantially increasing training time.

### Gradient-Boosted Trees

| Iterations | Max Depth | Validation RMSE | Validation MAE | Validation R² | Training Time |
|---:|---:|---:|---:|---:|---:|
| 10 | 5 | 255.44 | 164.65 | 0.7725 | 6.62 s |
| 30 | 5 | 249.65 | 159.70 | 0.7827 | 17.05 s |
| 50 | 5 | **247.73** | **158.00** | **0.7860** | 26.30 s |

Selected configuration:

```text
maxIter = 50
maxDepth = 5
stepSize = 0.1
```

## Final Test Results

The selected models were evaluated once on the untouched January test set.

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Random Forest | 245.29 s | 163.45 s | 0.7859 |
| GBT | **238.32 s** | **157.26 s** | **0.7979** |

GBT achieved better results across all three evaluation metrics.

## Unseen February Data

The saved models were also applied to February data, which was not used during model training or parameter selection.

Initial February evaluation showed lower performance than the random January test split, demonstrating the difficulty of temporal generalization.

Prediction results are exported to HDFS as CSV and analyzed using Python and Matplotlib.

## Visualizations

Generated plots:

```text
results/plots/actual_vs_predicted.png
results/plots/error_distribution.png
results/plots/error_by_hour.png
```

The analysis shows:

- Most predictions follow the actual trip duration reasonably closely.
- Large errors are concentrated primarily among unusually long trips.
- Long trips tend to be underpredicted.
- Prediction error varies by time of day and is higher around midday.

## Spark Performance Experiment

Performance was tested using different Spark cluster configurations.

### 500,000 records

Final model configurations were used for the initial benchmark.

| Model | 1 Worker / 2 Cores | 2 Workers / 4 Cores |
|---|---:|---:|
| Random Forest (100 trees) | 151.27 s | 156.06 s |
| GBT (50 iterations) | 35.42 s | 34.61 s |

### 2,000,000 records

For the larger scaling experiment, Random Forest was reduced to 50 trees because the 100-tree model exceeded the practical resources of the local Docker cluster.

| Model | 1 Worker / 2 Cores | 2 Workers / 4 Cores |
|---|---:|---:|
| Random Forest (50 trees) | **135.12 s** | 139.58 s |
| GBT (50 iterations) | **90.80 s** | 95.13 s |

Adding a second worker did not significantly improve training performance for these workloads.

The additional Spark scheduling, communication, and synchronization overhead offset the additional computational resources in this small local Docker cluster.

Experiments with larger datasets and the 100-tree Random Forest also demonstrated substantially higher resource requirements.

## Running the Project

### Start the environment

```powershell
docker compose up -d
```

Check containers:

```powershell
docker ps
```

### Upload dataset to HDFS

```powershell
docker exec p3-namenode hdfs dfs -mkdir -p /uber/raw
```

January:

```powershell
docker exec p3-namenode hdfs dfs -put /data/raw/fhvhv_tripdata_2021-01.parquet /uber/raw/
```

February:

```powershell
docker exec p3-namenode hdfs dfs -put /data/raw/fhvhv_tripdata_2021-02.parquet /uber/raw/
```

Check files:

```powershell
docker exec p3-namenode hdfs dfs -ls /uber/raw
```

### Train Models

```powershell
docker exec p3-spark-app /opt/spark/bin/spark-submit --master spark://spark-master:7077 --conf spark.driver.host=spark-app /opt/app/train.py
```

The selected models are saved to:

```text
/uber/models/random_forest
/uber/models/gradient_boosted_trees
```

### Run Predictions

```powershell
docker exec p3-spark-app /opt/spark/bin/spark-submit --master spark://spark-master:7077 --conf spark.driver.host=spark-app /opt/app/predict.py
```

Predictions are written to:

```text
/uber/predictions/gbt_february
```

### Download Prediction CSV

```powershell
docker exec p3-namenode hdfs dfs -getmerge /uber/predictions/gbt_february /tmp/gbt_february.csv
```

```powershell
docker cp p3-namenode:/tmp/gbt_february.csv results/predictions/gbt_february.csv
```

### Generate Visualizations

Run locally:

```powershell
python src/visualize.py
```

### Performance Benchmark

```powershell
docker exec p3-spark-app /opt/spark/bin/spark-submit --master spark://spark-master:7077 --conf spark.driver.host=spark-app /opt/app/benchmark.py
```

To benchmark with one worker:

```powershell
docker stop p3-spark-worker-2
```

Run the benchmark and then restore the worker:

```powershell
docker start p3-spark-worker-2
```

## Project Structure

```text
.
├── data/
│   ├── raw/
│   └── lookup/
├── docker/
│   ├── app/
│   │   └── Dockerfile
│   └── hadoop/
│       ├── core-site.xml
│       └── hdfs-site.xml
├── results/
│   ├── predictions/
│   └── plots/
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── visualize.py
│   └── benchmark.py
├── docker-compose.yml
└── README.md
```

## Conclusion

Both Random Forest and Gradient-Boosted Trees were successfully trained and evaluated using Spark MLlib.

GBT produced the best overall prediction accuracy while requiring less training time than the final Random Forest configuration.

Testing on unseen February data showed reduced accuracy compared with the random January test split, highlighting temporal generalization challenges.

Spark cluster experiments also showed that increasing the number of workers does not necessarily improve performance for workloads where distributed execution overhead is significant.