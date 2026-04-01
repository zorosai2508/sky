import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_data():
    path = os.path.join(BASE_DIR, "data/processed/base_dataset.csv")
    return pd.read_csv(path)


def feature_engineering(df):
    df = df.copy()

    df["price_per_km"] = df["price"] / (df["distance"] + 1)
    df["speed"] = df["distance"] / (df["duration"] + 1)

    df["value_score"] = df["distance"] / (df["price"] + 1)

    df["price_norm"] = (df["price"] - df["price"].min()) / (df["price"].max() - df["price"].min())
    df["delay_norm"] = (df["delay"] - df["delay"].min()) / (df["delay"].max() - df["delay"].min())

    df["utility_score"] = (
        0.4 * df["price_norm"] +
        0.3 * df["delay_norm"] -
        0.3 * df["sentiment_score"] +
        np.random.normal(0, 0.05, len(df))
    )

    return df


def save_data(df):
    path = os.path.join(BASE_DIR, "data/processed/final_dataset.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print("✅ Final dataset saved!")


if __name__ == "__main__":
    df = load_data()
    df = feature_engineering(df)
    save_data(df)

    print("🎉 Feature engineering completed!")
