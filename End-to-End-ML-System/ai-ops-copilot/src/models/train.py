import pandas as pd
import xgboost as xgb
import joblib
import os
from sklearn.metrics import mean_absolute_error

def train_model(df, target):

    split_date = df['datetime'].max() - pd.DateOffset(years=1)

    train_df = df[df['datetime'] < split_date]
    test_df = df[df['datetime'] >= split_date]

    X_train = train_df.drop(columns=[target, 'datetime'])
    y_train = train_df[target]

    X_test = test_df.drop(columns=[target, 'datetime'])
    y_test = test_df[target]

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)

    print(f"MAE: {mae}")
    
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/model.pkl")

    return model