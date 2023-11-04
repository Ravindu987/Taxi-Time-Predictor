import hydra
import joblib
import mlflow
import os
import pandas as pd

from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_squared_log_error,
)


def load_test_data(config: DictConfig):
    """
    Load the test data
    """
    test_x = pd.read_csv(abspath(config.processed.test_x.path))
    test_y = pd.read_csv(abspath(config.processed.test_y.path))
    return test_x, test_y


def load_model(config: DictConfig):
    """
    Load the model
    """
    model = joblib.load(abspath(config.model.model_path))
    return model


def predict(model, test_x: pd.DataFrame):
    """
    Predict the test data
    """
    predictions = model.predict(test_x)
    return predictions


def log_features(test_x: pd.DataFrame):
    """
    Log the features used for training
    """
    features = test_x.columns.tolist()
    mlflow.log_params({"features": features})


def log_parameters(model):
    """
    Log the parameters of the model
    """

    mlflow.log_params({"model_class": type(model).__name__})
    model_params = model.get_params()

    for arg, value in model_params.items():
        mlflow.log_params({arg: value})


@hydra.main(config_path="../../config", config_name="main", version_base="1.1")
def eval(config: DictConfig):
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    os.environ["MLFLOW_TRACKING_USERNAME"] = config.mlflow_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = config.mlflow_PASSWORD

    with mlflow.start_run():
        test_x, test_y = load_test_data(config)
        model = load_model(config)
        predictions = predict(model, test_x)

        mlflow.log_param("max_evals", config.model.max_evals)
        log_features(test_x)
        log_parameters(model)

        rmse = mean_squared_error(test_y, predictions, squared=False)
        mse = mean_squared_error(test_y, predictions)
        mae = mean_absolute_error(test_y, predictions)
        r2 = r2_score(test_y, predictions)
        rmsle = mean_squared_log_error(test_y, predictions)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("rmsle", rmsle)

        print(f"RMSE: {rmse}")
        print(f"MSE: {mse}")
        print(f"MAE: {mae}")
        print(f"R2: {r2}")
        print(f"RMSLE: {rmsle}")


if __name__ == "__main__":
    eval()
