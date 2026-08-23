from __future__ import annotations

import logging
from time import perf_counter

import pandas as pd
from numpy.typing import ArrayLike
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def train_xgboost(
    X_train: ArrayLike,
    y_train: pd.Series,
    config: dict,
) -> XGBClassifier:
    """Train the XGBoost model."""

    start = perf_counter()

    modelling = config["parameters"]["modelling"]
    xgb_config = modelling["xgboost"]

    logger.info(
        "XGBoost training started: " "rows=%s features=%s event_rate=%.6f",
        f"{X_train.shape[0]:,}",
        X_train.shape[1],
        y_train.mean(),
    )

    model = XGBClassifier(
        n_estimators=xgb_config["n_estimators"],
        max_depth=xgb_config["max_depth"],
        learning_rate=xgb_config["learning_rate"],
        subsample=xgb_config["subsample"],
        colsample_bytree=xgb_config["colsample_bytree"],
        min_child_weight=xgb_config["min_child_weight"],
        reg_alpha=xgb_config["reg_alpha"],
        reg_lambda=xgb_config["reg_lambda"],
        objective=xgb_config["objective"],
        eval_metric=xgb_config["eval_metric"],
        random_state=modelling["random_state"],
        n_jobs=xgb_config["n_jobs"],
        scale_pos_weight=xgb_config["scale_pos_weight"],
        class_weight=xgb_config["class_weight"],
    )

    model.fit(X_train, y_train)

    logger.info(
        "XGBoost training completed: " "trees=%s duration_seconds=%.2f",
        model.n_estimators,
        perf_counter() - start,
    )

    return model
