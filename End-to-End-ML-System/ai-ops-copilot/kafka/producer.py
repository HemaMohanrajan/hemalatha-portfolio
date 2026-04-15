from kafka import KafkaProducer
import json
import time
import pandas as pd

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Load dataset
df = pd.read_csv("data/raw/PJME_hourly.csv")
df.columns = [col.lower() for col in df.columns]
df.rename(columns={"pjme_mw": "value"}, inplace=True)

df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime')


for _, row in df.iterrows():
    message = {
        "datetime": str(row['datetime']),
        "value": row['value']
    }

    producer.send("energy_topic", message)
    print("Sent:", message)

    time.sleep(1)  # simulate real-time