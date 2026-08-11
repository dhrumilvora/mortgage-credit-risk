from __future__ import annotations

import logging
from time import perf_counter

import pandas as pd
from sklearn.linear_model import LogisticRegression
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


def train_logistic_regression(
    X_train: ArrayLike,
    y_train: pd.Series,
    config: dict,
) -> LogisticRegression:
    """Train the baseline logistic regression model."""

    start = perf_counter()

    modelling = config["parameters"]["modelling"]
    logistic = modelling["logistic_regression"]

    logger.info(
        "Logistic regression training started: " "rows=%s features=%s event_rate=%.6f",
        f"{X_train.shape[0]:,}",
        X_train.shape[1],
        y_train.mean(),
    )

    model = LogisticRegression(
        penalty=logistic["penalty"],
        C=logistic["C"],
        solver=logistic["solver"],
        max_iter=logistic["max_iter"],
        class_weight=logistic["class_weight"],
        random_state=modelling["random_state"],
    )

    model.fit(X_train, y_train)

    logger.info(
        "Logistic regression training completed: "
        "coefficients=%s intercept=%s "
        "duration_seconds=%.2f",
        model.coef_.shape[1],
        model.intercept_.shape[0],
        perf_counter() - start,
    )

    return model
