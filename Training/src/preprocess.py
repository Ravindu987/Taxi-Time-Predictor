import pandas as pd
import numpy as np
import hydra

from omegaconf import DictConfig
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from hydra.utils import to_absolute_path as abspath


def haversine_(lat1, lng1, lat2, lng2):
    """function to calculate haversine distance between two co-ordinates"""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


def bearing_array(lat1, lng1, lat2, lng2):
    """Calculates the angle or direction of 2 points with consideration of the roundness of earth."""
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(
        lng_delta_rad
    )
    return np.degrees(np.arctan2(y, x))


def get_data(path: str) -> pd.DataFrame:
    """
    Load the data from the path
    """
    data = pd.read_csv(path)
    return data


def filter_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the rows from the dataframe
    """

    df = df[~(df["passenger_count"] == 0)]
    df = df[(df["pickup_longitude"] > -74.1) & (df["pickup_longitude"] < -73.7)]
    df = df[(df["pickup_latitude"] > 40.6) & (df["pickup_latitude"] < 40.8)]
    df = df[(df["dropoff_longitude"] > -74.1) & (df["dropoff_longitude"] < -73.7)]
    df = df[(df["dropoff_latitude"] > 40.6) & (df["dropoff_latitude"] < 40.8)]
    return df


def feature_addition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features to the dataframe
    """

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["pickup_month"] = df["pickup_datetime"].dt.month
    df["pickup_date"] = df["pickup_datetime"].dt.day
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_weekday"] = df["pickup_datetime"].dt.weekday
    df["day_of_year"] = df["pickup_datetime"].dt.dayofyear

    coords = np.vstack(
        (
            df[["pickup_latitude", "pickup_longitude"]].values,
            df[["dropoff_latitude", "dropoff_longitude"]].values,
        )
    )

    pca = PCA().fit(coords)
    df["pickup_pca0"] = pca.transform(df[["pickup_latitude", "pickup_longitude"]])[:, 0]
    df["pickup_pca1"] = pca.transform(df[["pickup_latitude", "pickup_longitude"]])[:, 1]
    df["dropoff_pca0"] = pca.transform(df[["dropoff_latitude", "dropoff_longitude"]])[
        :, 0
    ]
    df["dropoff_pca1"] = pca.transform(df[["dropoff_latitude", "dropoff_longitude"]])[
        :, 1
    ]

    df.loc[:, "distance_haversine"] = haversine_(
        df.loc[:, "pickup_latitude"].values,
        df.loc[:, "pickup_longitude"].values,
        df.loc[:, "dropoff_latitude"].values,
        df.loc[:, "dropoff_longitude"].values,
    )

    df.loc[:, "direction"] = bearing_array(
        df.loc[:, "pickup_latitude"].values,
        df.loc[:, "pickup_longitude"].values,
        df.loc[:, "dropoff_latitude"].values,
        df.loc[:, "dropoff_longitude"].values,
    )

    df.loc[:, "center_latitude"] = (
        df.loc[:, "pickup_latitude"].values + df.loc[:, "dropoff_latitude"].values
    ) / 2
    df.loc[:, "center_longitude"] = (
        df.loc[:, "pickup_longitude"].values + df.loc[:, "dropoff_longitude"].values
    ) / 2

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
    df = df[df["distance_haversine"] != 0]
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove the outliers from the dataframe
    """

    df["td"] = abs(df["trip_duration"] / df["distance_haversine"])

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


def cluster(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cluster the columns
    """
    kmeans_pickup = MiniBatchKMeans(
        n_clusters=50, random_state=2, batch_size=10000, n_init=10
    ).fit(df[["pickup_longitude", "pickup_latitude"]])
    kmeans_drop = MiniBatchKMeans(
        n_clusters=50, random_state=2, batch_size=10000, n_init=10
    ).fit(df[["dropoff_longitude", "dropoff_latitude"]])

    # train_df
    df.loc[:, "pickup_cluster"] = kmeans_pickup.predict(
        df.loc[:, ["pickup_longitude", "pickup_latitude"]]
    )
    df.loc[:, "dropoff_cluster"] = kmeans_drop.predict(
        df.loc[:, ["dropoff_longitude", "dropoff_latitude"]]
    )

    centroid_pickups = pd.DataFrame(
        kmeans_pickup.cluster_centers_,
        columns=["centroid_pick_long", "centroid_pick_lat"],
    )
    centroid_dropoff = pd.DataFrame(
        kmeans_drop.cluster_centers_,
        columns=["centroid_drop_long", "centroid_drop_lat"],
    )

    # assign names to each cluster like 1,2,3,etc.
    centroid_pickups["pickup_cluster"] = centroid_pickups.index
    centroid_dropoff["dropoff_cluster"] = centroid_dropoff.index

    # train
    df = pd.merge(df, centroid_pickups, how="left", on=["pickup_cluster"])
    df = pd.merge(df, centroid_dropoff, how="left", on=["dropoff_cluster"])

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

    data = filter_rows(data)
    data = feature_addition(data)
    data = onehot_categorical(data, config.features.onehot_categorical)
    data = drop_columns(data, config.features.drop_columns)
    data = remove_zero_distance(data)
    data = remove_outliers(data)
    # data = scale(data, config.features.scale_columns)
    # data = cluster(data)

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
