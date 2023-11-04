import pandas as pd
import numpy as np
import hydra

from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from hydra.utils import to_absolute_path as abspath


def get_data(path: str) -> pd.DataFrame:
    """
    Load the data from the path
    """
    data = pd.read_csv(path)
    return data


def feature_addition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features to the dataframe
    """

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["pickup_month"] = df["pickup_datetime"].dt.month
    df["pickup_date"] = df["pickup_datetime"].dt.day
    df["pickup_hour"] = df["pickup_datetime"].dt.hour

    df["distance_latitude"] = abs(df["dropoff_latitude"] - df["pickup_latitude"])
    df["distance_longitude"] = abs(df["dropoff_longitude"] - df["pickup_longitude"])
    df["distance"] = np.sqrt(
        np.square(df["distance_latitude"]) + np.square(df["distance_longitude"])
    )

    return df


def onehot_categorical(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Onehot encode the categorical features
    """
    df = pd.get_dummies(df, columns=columns)
    return df


def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Drop the columns from the dataframe
    """
    df = df.drop(columns=columns, axis=1, errors="ignore")
    return df


def remove_zero_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove the rows with zero distance
    """
    df = df[df["distance"] != 0]
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove the outliers from the dataframe
    """

    df["td"] = abs(df["trip_duration"] / df["distance"])

    Q1 = df["td"].quantile(0.25)
    Q3 = df["td"].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df["td"] >= lower_bound) & (df["td"] <= upper_bound)]

    df = df.drop(["td"], axis=1)

    return df


def scale(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Scale the columns
    """

    columns = [col for col in columns if col in df.columns]
    df[columns] = StandardScaler().fit_transform(df[columns])
    return df


def get_x_y(df: pd.DataFrame) -> tuple:
    """
    Get the x and y from the dataframe
    """
    x = df.drop("trip_duration", axis=1)
    y = df["trip_duration"]
    return x, y


def preprocess_data(data, config):
    """
    Preprocess the data
    """

    data = feature_addition(data)
    data = onehot_categorical(data, config.features.onehot_categorical)
    data = drop_columns(data, config.features.drop_columns)
    data = remove_zero_distance(data)
    data = remove_outliers(data)
    data = scale(data, config.features.scale_columns)

    return data


@hydra.main(config_path="../../config", config_name="main", version_base="1.1")
def process_data(config: DictConfig):
    data = get_data(abspath(config.data.path))

    data = preprocess_data(data, config)

    x, y = get_x_y(data)

    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    train_x.to_csv(abspath(config.processed.train_x.path), index=False)
    test_x.to_csv(abspath(config.processed.test_x.path), index=False)
    train_y.to_csv(abspath(config.processed.train_y.path), index=False)
    test_y.to_csv(abspath(config.processed.test_y.path), index=False)


if __name__ == "__main__":
    process_data()
