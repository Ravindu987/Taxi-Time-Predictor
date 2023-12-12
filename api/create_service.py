import sys

sys.path.append("/media/ravindu/Ravindu/Projects/Taxi Time Predictor/Training/src")

import bentoml
import numpy as np
import pandas as pd

from bentoml.io import JSON, NumpyNdarray
from omegaconf import DictConfig
from hydra import initialize, compose
from pydantic import BaseModel

from preprocess import (
    feature_addition,
    onehot_categorical,
    scale,
    drop_columns,
)

with initialize(config_path="../config", version_base="1.1"):
    config = compose(config_name="main")


class Travel_Record(BaseModel):
    vendor_id: int
    pickup_datetime: str
    passenger_count: int
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    store_and_fwd_flag: str


def preprocess_data(data: pd.DataFrame, config: DictConfig):
    """
    Preprocess the data
    """
    data = feature_addition(data)
    data = onehot_categorical(data, config.features.onehot_categorical)
    data = drop_columns(data, config.features.drop_columns)
    # data = scale(data, config.features.scale_columns)

    return data


def handle_missing_columns(df):
    """
    Handle the missing columns
    """
    missing_columns = set(config.features.onehot_encoded) - set(df.columns)
    for col in missing_columns:
        df[col] = 0
    return df


model = bentoml.xgboost.get(f"{config.model.name}:latest").to_runner()

service = bentoml.Service("predict_travel_time", runners=[model])


@service.api(input=JSON(pydantic_model=Travel_Record), output=NumpyNdarray())
def predict(record: Travel_Record) -> np.ndarray:
    df = pd.DataFrame(record.dict(), index=[0])
    processed = preprocess_data(df, config)
    processed = handle_missing_columns(processed)
    predictions = model.run(processed)
    return predictions
