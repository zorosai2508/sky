import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data/processed/final_dataset.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data/processed/xgb_predictions.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models/xgb_model.json")


# =========================
# LOAD DATA
# =========================
def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    return df


# =========================
# PREPARE FEATURES
# =========================

def prepare_data(df):
    df = df.copy()

    features = [
        "price_norm",
        "delay_norm",
        "duration",
        "distance",
        "sentiment_score"
    ]

    target = "utility_score"

    X = df[features]
    y = df[target]

    return X, y, df




# =========================
# TRAIN MODEL
# =========================
def train_model(X_train, y_train):
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model


# =========================
# EVALUATE MODEL
# =========================

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    print("\n📊 TRAIN PERFORMANCE")
    print("R2  :", r2_score(y_train, train_pred))
    print("MAE :", mean_absolute_error(y_train, train_pred))
    print("MSE :", mean_squared_error(y_train, train_pred))

    print("\n📊 TEST PERFORMANCE")
    print("R2  :", r2_score(y_test, test_pred))
    print("MAE :", mean_absolute_error(y_test, test_pred))
    print("MSE :", mean_squared_error(y_test, test_pred))






# =========================
# MAIN PIPELINE
# =========================
if __name__ == "__main__":
    print("🚀 Starting XGBoost baseline model...")

    # Load data
    df = load_data()

    # Prepare features
    X, y, df = prepare_data(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = train_model(X_train, y_train)

    # Predict test data
    y_pred = model.predict(X_test)

    # Evaluate
    evaluate_model(model, X_train, y_train, X_test, y_test)

    # Predict full dataset
    df["predicted_score"] = model.predict(X)

    # Save predictions
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save_model(MODEL_PATH)

    print("\n✅ Predictions saved at:", OUTPUT_PATH)
    print("✅ Model saved at:", MODEL_PATH)
    print("🎉 Baseline model completed!")

