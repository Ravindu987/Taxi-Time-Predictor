import hydra
from preprocess import process_data
from train import train_model
from evaluate import eval

from omegaconf import DictConfig


@hydra.main(config_path="../../config", config_name="main", version_base="1.1")
def main(config: DictConfig):
    process_data(config)
    train_model(config)
    eval(config)
