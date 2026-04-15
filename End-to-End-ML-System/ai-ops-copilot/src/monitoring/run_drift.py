import pandas as pd
from src.monitoring.drift import run_drift_detection

# Load reference (training data)
reference_df = pd.read_parquet("data/feature_data.parquet")

# Simulate new data (current)
current_df = reference_df.sample(frac=0.2).copy()

# OPTIONAL: simulate drift
current_df["value"] = current_df["value"] * 1.2

run_drift_detection(reference_df, current_df)