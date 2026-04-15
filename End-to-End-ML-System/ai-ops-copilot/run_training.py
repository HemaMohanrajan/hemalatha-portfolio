import sys
import os
from src.data_loader import load_data
from src.features.build_features import build_features
from src.models.train import train_model


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "")))

DATA_PATH = "data/raw/PJME_hourly.csv"
TARGET = "value"

def main():
    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Building features...")
    df = build_features(df, TARGET)

    print("Training model...")
    train_model(df, TARGET)

    print("Done!")

if __name__ == "__main__":
    main()