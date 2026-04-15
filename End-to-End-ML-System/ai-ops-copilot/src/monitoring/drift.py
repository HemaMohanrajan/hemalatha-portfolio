import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import os

def run_drift_detection(reference_df: pd.DataFrame, current_df: pd.DataFrame):

    drift_results = {}

    numeric_cols = reference_df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        stat, p_value = ks_2samp(reference_df[col], current_df[col])

        drift_results[col] = {
            "p_value": float(p_value),
            "drift_detected": bool(p_value < 0.05)
        }

    # Save report
    os.makedirs("reports", exist_ok=True)

    with open("reports/drift_report.txt", "w") as f:
        for col, result in drift_results.items():
            f.write(f"{col}: {result}\n")

    print("Custom drift report generated!")