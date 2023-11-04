import hydra
from preprocess import process_data
from train import train_model
from evaluate import eval

from omegaconf import DictConfig


@hydra.main(config_path="../../config", config_name="main", version_base="1.1")
def main(config: DictConfig):
    print("Preprocessing data...")
    process_data(config)
    print("Training model...")
    train_model(config)
    print("Evaluating model...")
    eval(config)


if __name__ == "__main__":
    main()
