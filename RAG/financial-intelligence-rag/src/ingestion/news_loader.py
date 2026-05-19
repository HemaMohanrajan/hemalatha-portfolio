import pandas as pd
from pathlib import Path

def load_news_data(news_dir: Path) -> pd.DataFrame:
    cnbc = pd.read_csv(news_dir / "cnbc_headlines.csv")
    cnbc = cnbc.rename(columns={
        "Headlines": "headline",
        "Time": "time",
        "Description": "description"
    })
    cnbc["source"] = "cnbc"

    guardian = pd.read_csv(news_dir / "guardian_headlines.csv")
    guardian = guardian.rename(columns={
        "Headlines": "headline",
        "Time": "time"
    })
    guardian["description"] = ""
    guardian["source"] = "guardian"

    reuters = pd.read_csv(news_dir / "reuters_headlines.csv")
    reuters = reuters.rename(columns={
        "Headlines": "headline",
        "Time": "time",
        "Description": "description"
    })
    reuters["source"] = "reuters"

    return pd.concat([cnbc, guardian, reuters], ignore_index=True)