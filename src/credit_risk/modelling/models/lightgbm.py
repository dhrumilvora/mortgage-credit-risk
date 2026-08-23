from __future__ import annotations
import logging
from time import perf_counter
from lightgbm import LGBMClassifier
from numpy.typing import ArrayLike
import pandas as pd

logger = logging.getLogger(__name__)


def train_lightgbm(
    X_train: ArrayLike, y_train: pd.Series, config: dict
) -> LGBMClassifier:
    "Train the configured Light-GBM Model"
    start = perf_counter()
    lightgbm_config = config["parameters"]["modelling"]["lightgbm"]
    model = LGBMClassifier(
        n_estimators=lightgbm_config["n_estimators"],
        max_depth=lightgbm_config["max_depth"],
        learning_rate=lightgbm_config["learning_rate"],
        num_leaves=lightgbm_config["num_leaves"],
        min_child_samples=lightgbm_config["min_child_samples"],
        subsample=lightgbm_config["subsample"],
        colsample_bytree=lightgbm_config["colsample_bytree"],
        reg_alpha=lightgbm_config["reg_alpha"],
        reg_lambda=lightgbm_config["reg_lambda"],
        objective=lightgbm_config["objective"],
        n_jobs=lightgbm_config["n_jobs"],
        scale_pos_weight=lightgbm_config["scale_pos_weight"],
        random_state=lightgbm_config["random_state"],
        verbosity=-1,
    )

    logger.info(
        "LightGBM training started: rows=%s features=%s event_rate=%.6f",
        f"{X_train.shape[0]:,}",
        X_train.shape[1],
        y_train.mean(),
    )

    model.fit(
        X_train,
        y_train,
    )

    logger.info(
        "LightGBM training completed: " "trees=%s duration_seconds=%.2f",
        lightgbm_config["n_estimators"],
        perf_counter() - start,
    )

    return model
