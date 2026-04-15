import numpy as np
import holidays

def create_time_features(df):
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['dayofyear'] = df['datetime'].dt.dayofyear

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    return df

def create_lag_features(df, target):
    df['lag_1'] = df[target].shift(1)
    df['lag_24'] = df[target].shift(24)
    df['lag_168'] = df[target].shift(168)
    return df

def create_rolling_features(df, target):
    df['rolling_mean_24'] = df[target].rolling(24).mean()
    df['rolling_std_24'] = df[target].rolling(24).std()
    return df


def add_holiday_feature(df):
    us_holidays = holidays.US()
    df['is_holiday'] = df['datetime'].dt.normalize().isin(us_holidays)
    return df

def build_features(df, target):
    df = create_time_features(df)
    df = create_lag_features(df, target)
    df = create_rolling_features(df, target)
    df = add_holiday_feature(df)

    df = df.dropna()

    return df