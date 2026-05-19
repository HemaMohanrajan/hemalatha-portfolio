import json
from pathlib import Path
import pandas as pd

def load_sec_sections(sec_dir: Path, max_files=50):
    records = []

    for path in list(sec_dir.glob("*.json"))[:max_files]:
        with open(path, "r") as f:
            data = json.load(f)

        for key, value in data.items():
            if key.lower().startswith("item_") and isinstance(value, str):
                records.append({
                    "cik": data.get("cik"),
                    "company_name": data.get("company"),
                    "item": key,
                    "raw_text": value
                })

    return pd.DataFrame(records)