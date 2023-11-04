import joblib
import pandas as pd
import hydra

from omegaconf import DictConfig
from hydra.utils import to_absolute_path as abspath

from preprocess import feature_addition, onehot_categorical, scale, drop_columns


def load_model(config: DictConfig):
    """
    Load the model
    """
    model = joblib.load(abspath(config.model.model_path))
    return model


def get_data(config: DictConfig):
    """
    Load the test data
    """
    test_x = pd.read_csv(abspath(config.data.test_path))
    return test_x


def preprocess_data(data: pd.DataFrame, config: DictConfig):
    """
    Preprocess the data
    """
    data = feature_addition(data)
    data = onehot_categorical(data, config.features.onehot_categorical)
    data = drop_columns(data, config.features.drop_columns)
    data = scale(data, config.features.scale_columns)

    return data


def predict(model, test_x: pd.DataFrame):
    """
    Predict the test data
    """
    predictions = model.predict(test_x)
    return predictions


def write_to_csv(predictions, id, path):
    """
    Write the predictions to a csv file
    """

    results = pd.DataFrame({"id": id, "trip_duration": predictions})

    results.to_csv(abspath(path), index=False)


@hydra.main(config_path="../../config", config_name="main", version_base="1.1")
def test(config: DictConfig):
    """
    Test the model
    """

    model = load_model(config)
    test_x = get_data(config)
    id = test_x["id"]
    print(test_x.shape)
    test_x = preprocess_data(test_x, config)
    predictions = predict(model, test_x)
    print(predictions.shape)
    write_to_csv(predictions, id, config.data.result_path)


if __name__ == "__main__":
    test()
