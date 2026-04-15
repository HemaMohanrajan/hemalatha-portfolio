import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df.columns = [col.lower() for col in df.columns]
    df.rename(columns={"pjme_mw": "value"}, inplace=True)

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')

    # Remove duplicates
    df = df.drop_duplicates()

    return df