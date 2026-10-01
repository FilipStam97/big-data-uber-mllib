import pandas as pd
import matplotlib.pyplot as plt


INPUT_PATH = "results/predictions/gbt_february.csv"
OUTPUT_DIR = "results/plots"


def main():
    df = pd.read_csv(INPUT_PATH)

    # Prediction error
    df["error"] = df["prediction"] - df["trip_time"]

    # 1. Actual vs predicted duration
    sample = df.sample(
        min(5000, len(df)),
        random_state=42
    )

    plt.figure(figsize=(8, 6))

    plt.scatter(
        sample["trip_time"] / 60,
        sample["prediction"] / 60,
        alpha=0.25,
        s=10
    )

    max_value = max(
        sample["trip_time"].max(),
        sample["prediction"].max()
    ) / 60

    plt.plot(
        [0, max_value],
        [0, max_value],
        linestyle="--"
    )

    plt.xlabel("Actual trip duration (minutes)")
    plt.ylabel("Predicted trip duration (minutes)")
    plt.title("Actual vs Predicted Trip Duration")

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/actual_vs_predicted.png",
        dpi=150
    )
    plt.close()


    # 2. Prediction error distribution
    plt.figure(figsize=(8, 6))

    errors_minutes = df["error"] / 60

    plt.hist(
        errors_minutes,
        bins=80,
        range=(-30, 30)
    )

    plt.axvline(
        0,
        linestyle="--"
    )

    plt.xlabel("Prediction error (minutes)")
    plt.ylabel("Number of trips")
    plt.title("GBT Prediction Error Distribution")

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/error_distribution.png",
        dpi=150
    )
    plt.close()

    # 3. Average error by pickup hour
    hourly = (
        df
        .assign(abs_error=df["error"].abs())
        .groupby("pickup_hour")["abs_error"]
        .mean()
        / 60
    )

    plt.figure(figsize=(8, 6))

    plt.bar(
        hourly.index,
        hourly.values
    )

    plt.xlabel("Pickup hour")
    plt.ylabel("Mean absolute error (minutes)")
    plt.title("Prediction Error by Time of Day")

    plt.xticks(range(24))

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/error_by_hour.png",
        dpi=150
    )
    plt.close()

    print("Created:")
    print("  actual_vs_predicted.png")
    print("  error_distribution.png")
    print("  error_by_hour.png")


if __name__ == "__main__":
    main()