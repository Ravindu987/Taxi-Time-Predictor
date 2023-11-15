import bentoml
import hydra
import joblib

from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig


@hydra.main(config_path="../config", config_name="main", version_base="1.1")
def save_to_bentoml(config: DictConfig):
    """
    Save the model to bentoml
    """
    model = joblib.load(abspath(config.model.model_path))
    bentoml.picklable_model.save(config.model.name, model)


if __name__ == "__main__":
    save_to_bentoml()
