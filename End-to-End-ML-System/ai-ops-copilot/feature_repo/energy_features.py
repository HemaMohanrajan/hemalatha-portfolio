from feast import Entity, FeatureView, Field
from feast.types import Float32, Int64
from feast import FileSource

import pandas as pd

# Entity (key)
energy = Entity(name="energy_id", join_keys=["energy_id"])

# Data source
data_source = FileSource(
    path="../data/feature_data.parquet",
    timestamp_field="event_timestamp"
)

# Feature View
energy_features = FeatureView(
    name="energy_features",
    entities=[energy],
    ttl=None,
    schema=[
        Field(name="lag_1", dtype=Float32),
        Field(name="lag_24", dtype=Float32),
        Field(name="lag_168", dtype=Float32),
        Field(name="rolling_mean_24", dtype=Float32),
        Field(name="rolling_std_24", dtype=Float32),
        Field(name="hour", dtype=Int64),
        Field(name="dayofweek", dtype=Int64),
        Field(name="month", dtype=Int64),
    ],
    source=data_source,
)