import pandas as pd

def clean_news_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["headline"]).copy()
    df["text"] = df["headline"].astype(str) + ". " + df["description"].astype(str)
    df = df[df["text"].str.len() > 20]
    return df.reset_index(drop=True)