from __future__ import annotations
import logging
from time import perf_counter
from sklearn.ensemble import RandomForestClassifier
from numpy.typing import ArrayLike
import pandas as pd

logger = logging.getLogger(__name__)


def train_random_forest(
    X_train: ArrayLike, y_train: pd.Series, config: dict
) -> RandomForestClassifier:
    """Train Random Forest Classifier"""
    start = perf_counter()
    random_forest_config = config["parameters"]["modelling"]["random_forest"]
    model = RandomForestClassifier(
        n_estimators=random_forest_config["n_estimators"],
        max_depth=random_forest_config["max_depth"],
        min_samples_split=random_forest_config["min_samples_split"],
        min_samples_leaf=random_forest_config["min_samples_leaf"],
        max_features=random_forest_config["max_features"],
        bootstrap=random_forest_config["bootstrap"],
        class_weight=random_forest_config["class_weight"],
        random_state=random_forest_config["random_state"],
        n_jobs=random_forest_config["n_jobs"],
    )
    logger.info(
        "Random Forest training started: rows=%s features=%s event_rate=%.6f",
        f"{X_train.shape[0]:,}",
        X_train.shape[1],
        y_train.mean(),
    )

    model.fit(
        X_train,
        y_train,
    )

    logger.info(
        "Random Forest training completed: " "trees=%s duration_seconds=%.2f",
        random_forest_config["n_estimators"],
        perf_counter() - start,
    )

    return model
