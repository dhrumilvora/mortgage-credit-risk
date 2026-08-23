from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from credit_risk.modelling.models.xgboost import train_xgboost
from credit_risk.modelling.models.linear_regression import (
    train_logistic_regression,
)
from credit_risk.modelling.models.lightgbm import train_lightgbm
from credit_risk.modelling.models.random_forest import train_random_forest


def train_model(
    X_train,
    y_train,
    config: dict,
) -> XGBClassifier | LogisticRegression:

    if X_train.shape[0] == 0:
        raise ValueError("Training feature matrix is empty.")

    if y_train.shape[0] == 0:
        raise ValueError("Training target is empty.")

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            "Training features and target contain different " "numbers of rows."
        )

    if y_train.nunique() < 2:
        raise ValueError("Training target must contain at least two classes.")

    algorithm = config["parameters"]["modelling"]["algorithm"]

    if algorithm == "logistic_regression":
        return train_logistic_regression(
            X_train,
            y_train,
            config,
        )

    elif algorithm == "xgboost":
        return train_xgboost(
            X_train,
            y_train,
            config,
        )
    elif algorithm == "random_forest":
        return train_random_forest(X_train, y_train, config)
    elif algorithm == "lightgbm":
        return train_lightgbm(X_train, y_train, config)
    raise ValueError(f"Unsupported modelling algorithm: {algorithm}")
