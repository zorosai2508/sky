import pandas as pd
import os
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data/raw/Tweets.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data/processed/sentiment_scores.csv")


def load_data():
    return pd.read_csv(INPUT_PATH)


def compute_sentiment(df):
    sia = SentimentIntensityAnalyzer()

    df["sentiment_score"] = df["text"].apply(
        lambda x: sia.polarity_scores(str(x))["compound"]
    )

    return df


def aggregate_sentiment(df):
    sentiment = df.groupby("airline")["sentiment_score"].mean().reset_index()
    sentiment["airline"] = sentiment["airline"].str.lower().str.strip()
    return sentiment


def save_data(sentiment):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    sentiment.to_csv(OUTPUT_PATH, index=False)
    print("✅ Sentiment scores saved!")


if __name__ == "__main__":
    df = load_data()
    df = compute_sentiment(df)
    sentiment = aggregate_sentiment(df)
    save_data(sentiment)

    print("🎉 NLP Sentiment Model Completed!")
