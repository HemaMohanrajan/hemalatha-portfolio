from src.data_loader import load_data
from src.features.build_features import build_features

DATA_PATH = "data/raw/PJME_hourly.csv"
TARGET = "value"

def main():
    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Building features...")
    df = build_features(df, TARGET)

    # Add required Feast columns
    df['energy_id'] = 1
    df['event_timestamp'] = df['datetime']

    print("Saving feature data...")
    df.to_parquet("data/feature_data.parquet", index=False)

    print("Feature data saved!")

if __name__ == "__main__":
    main()