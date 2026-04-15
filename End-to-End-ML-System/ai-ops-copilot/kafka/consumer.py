from kafka import KafkaConsumer
import json
import joblib
import numpy as np
import pandas as pd

model = joblib.load("models/model.pkl")

consumer = KafkaConsumer(
    "energy_topic",
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

data_buffer = []

for message in consumer:
    data = message.value
    print("Received:", data)

    # Store last values
    data_buffer.append(data['value'])

    if len(data_buffer) < 168:
        continue

    # Create simple features (minimal for demo)
    lag_1 = data_buffer[-1]
    lag_24 = data_buffer[-24]
    lag_168 = data_buffer[-168]

    rolling_mean = np.mean(data_buffer[-24:])
    rolling_std = np.std(data_buffer[-24:])

    dt = pd.to_datetime(data['datetime'])

    features = np.array([[
        dt.hour,
        dt.dayofweek,
        dt.month,
        dt.dayofyear,
        np.sin(2 * np.pi * dt.hour / 24),
        np.cos(2 * np.pi * dt.hour / 24),
        lag_1,
        lag_24,
        lag_168,
        rolling_mean,
        rolling_std,
        0
    ]])

    prediction = model.predict(features)

    print("Prediction:", prediction[0])