from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Energy Forecast API")

# Load model
model = joblib.load("models/model.pkl")


# Input schema
class InputData(BaseModel):
    hour: int
    dayofweek: int
    month: int
    dayofyear: int
    hour_sin: float
    hour_cos: float
    lag_1: float
    lag_24: float
    lag_168: float
    rolling_mean_24: float
    rolling_std_24: float
    is_holiday: bool


@app.get("/")
def home():
    return {"message": "API is running 🚀"}


@app.post("/predict")
def predict(data: InputData):

    features = np.array([[
        data.hour,
        data.dayofweek,
        data.month,
        data.dayofyear,
        data.hour_sin,
        data.hour_cos,
        data.lag_1,
        data.lag_24,
        data.lag_168,
        data.rolling_mean_24,
        data.rolling_std_24,
        int(data.is_holiday)
    ]])

    prediction = model.predict(features)

    return {"prediction": float(prediction[0])}