import pandas as pd
import hydra
import numpy as np
import xgboost as xgb
import joblib

from omegaconf import DictConfig
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

from sklearn.metrics import mean_squared_error

from functools import partial
from typing import Callable


def load_data(conf: DictConfig) -> pd.DataFrame:
    """
    Load the data from the path
    """
    train_x = pd.read_csv(conf.processed.train_x)
    test_x = pd.read_csv(conf.processed.test_x)
    train_y = pd.read_csv(conf.processed.train_y)
    test_y = pd.read_csv(conf.processed.test_y)
    return train_x, test_x, train_y, test_y


def train(
    train_x: pd.DataFrame,
    train_y: pd.DataFrame,
    test_x: pd.DataFrame,
    test_y: pd.DataFrame,
    config: DictConfig,
    space: dict,
) -> float:
    """
    Train the model
    """
    model = xgb.XGBRegressor(
        use_label_encoder=config.model.use_label_encoder,
        objective=config.model.objective,
        n_estimators=int(space["n_estimators"]),
        max_depth=int(space["max_depth"]),
        gamma=space["gamma"],
        reg_alpha=int(space["reg_alpha"]),
        min_child_weight=int(space["min_child_weight"]),
        colsample_bytree=int(space["colsample_bytree"]),
        eval_metric=config.model.eval_metric,
        early_stopping_rounds=config.model.early_stopping_rounds,
        device=config.model.device,
    )

    model.fit(
        train_x,
        train_y,
        eval_set=[(test_x, test_y)],
        verbose=config.model.verbose,
    )

    predictions = model.predict(test_x)
    rmse = mean_squared_error(test_y, predictions, squared=False)

    return rmse


def hypertune(objective, space):
    """
    Hyperparameter tuning
    """

    trials = Trials()
    best_hyperparams = fmin(
        objective, space, algo=tpe.suggest, max_evals=100, trials=trials
    )

    best_model = trials.results[np.argmin([r["loss"] for r in trials.results])]["model"]
    return best_model


@hydra.main(config_path="../config", config_name="main", version_base="1.1")
def main(config: DictConfig):
    train_x, test_x, train_y, test_y = load_data(config)

    space = {
        "max_depth": hp.quniform("max_depth", **config.model.max_depth),
        "gamma": hp.uniform("gamma", **config.model.gamma),
        "reg_alpha": hp.quniform("reg_alpha", **config.model.reg_alpha),
        "reg_lambda": hp.uniform("reg_lambda", **config.model.reg_lambda),
        "colsample_bytree": hp.uniform(
            "colsample_bytree", **config.model.colsample_bytree
        ),
        "min_child_weight": hp.quniform(
            "min_child_weight", **config.model.min_child_weight
        ),
        "n_estimators": hp.quniform("n_estimators", **config.model.n_estimators),
        "seed": config.model.seed,
    }

    objective = partial(train, train_x, train_y, test_x, test_y, config)

    best_model = hypertune(objective, space)

    joblib.dump(best_model, config.model.model_path)


if __name__ == "__main__":
    main()
