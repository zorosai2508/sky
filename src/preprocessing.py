import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_data():
    flights = pd.read_csv(os.path.join(BASE_DIR, "data/raw/flight.csv"), low_memory=False)
    airlines = pd.read_csv(os.path.join(BASE_DIR, "data/raw/airlines.csv"))
    airports = pd.read_csv(os.path.join(BASE_DIR, "data/raw/airports.csv"))

    return flights, airlines, airports


def preprocess_flights(flights):
    flights = flights[[
        'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
        'DEPARTURE_DELAY', 'ELAPSED_TIME', 'DISTANCE'
    ]].copy()

    flights = flights.rename(columns={
        'AIRLINE': 'airline',
        'ORIGIN_AIRPORT': 'source',
        'DESTINATION_AIRPORT': 'destination',
        'DEPARTURE_DELAY': 'delay',
        'ELAPSED_TIME': 'duration',
        'DISTANCE': 'distance'
    })

    flights = flights.dropna()
    flights = flights[(flights['delay'] > -20) & (flights['delay'] < 300)]

    return flights


def merge_airlines(flights, airlines):
    airlines = airlines.rename(columns={
        'IATA_CODE': 'airline',
        'AIRLINE': 'airline_name'
    })

    flights = flights.merge(airlines, on='airline', how='left')

    flights["airline_name"] = flights["airline_name"].str.lower().str.strip()

    return flights


# 🔥 FIXED MAPPING
def map_airline_names(flights):
    mapping = {
        "united air lines inc.": "united",
        "delta air lines inc.": "delta",
        "american airlines inc.": "american",
        "southwest airlines co.": "southwest",
        "us airways inc.": "us airways",
        "jetblue airways": "jetblue",
        "virgin america": "virgin america"
    }

    flights["airline_name"] = flights["airline_name"].map(mapping)

    return flights


def merge_airports(flights, airports):
    airports = airports[['IATA_CODE', 'AIRPORT', 'CITY']]

    src = airports.rename(columns={
        'IATA_CODE': 'source',
        'AIRPORT': 'source_airport',
        'CITY': 'source_city'
    })

    dst = airports.rename(columns={
        'IATA_CODE': 'destination',
        'AIRPORT': 'destination_airport',
        'CITY': 'destination_city'
    })

    flights = flights.merge(src, on='source', how='left')
    flights = flights.merge(dst, on='destination', how='left')

    return flights


def load_sentiment():
    path = os.path.join(BASE_DIR, "data/processed/sentiment_scores.csv")
    sentiment = pd.read_csv(path)
    sentiment["airline"] = sentiment["airline"].str.lower().str.strip()
    return sentiment


def merge_sentiment(flights, sentiment):
    flights = flights.merge(
        sentiment,
        left_on="airline_name",
        right_on="airline",
        how="left"
    )

    flights["sentiment_score"] = flights["sentiment_score"].fillna(0)

    return flights


def create_price(flights):
    flights["price"] = (
        flights["distance"] * 0.3 +
        flights["duration"] * 0.8 +
        flights["delay"] * 1.5 +
        100
    )
    return flights


def save_data(df):
    path = os.path.join(BASE_DIR, "data/processed/base_dataset.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print("✅ Base dataset saved!")


if __name__ == "__main__":
    flights, airlines, airports = load_data()

    flights = preprocess_flights(flights)
    flights = merge_airlines(flights, airlines)
    flights = map_airline_names(flights)  # 🔥 IMPORTANT FIX
    flights = merge_airports(flights, airports)

    sentiment = load_sentiment()
    flights = merge_sentiment(flights, sentiment)

    flights = create_price(flights)

    save_data(flights)
