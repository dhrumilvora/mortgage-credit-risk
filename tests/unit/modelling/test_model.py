"""Tests for baseline model training."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from credit_risk.modelling.model import train_logistic_regression


@pytest.fixture
def config() -> dict:
    """Minimal configuration required for logistic regression."""

    return {
        "parameters": {
            "modelling": {
                "random_state": 42,
                "logistic_regression": {
                    "penalty": "l2",
                    "C": 1.0,
                    "solver": "lbfgs",
                    "max_iter": 1000,
                    "class_weight": "balanced",
                },
            },
        },
    }


@pytest.fixture
def X_train() -> pd.DataFrame:
    """Synthetic transformed training features."""

    return pd.DataFrame(
        {
            "feature_1": [0.1, 0.3, 0.2, 0.7, 0.8, 0.9],
            "feature_2": [1.0, 2.0, 3.0, 2.0, 1.0, 0.5],
            "feature_3": [0, 1, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def y_train() -> pd.Series:
    """Binary training target."""

    return pd.Series(
        [0, 0, 0, 1, 1, 1],
        name="ever_90dpd_24m",
    )


def test_train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> None:
    """Model should train successfully."""

    model = train_logistic_regression(
        X_train,
        y_train,
        config,
    )

    assert isinstance(
        model,
        LogisticRegression,
    )


def test_model_learns_two_classes(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> None:
    """Model should learn both target classes."""

    model = train_logistic_regression(
        X_train,
        y_train,
        config,
    )

    assert model.classes_.tolist() == [0, 1]


def test_number_of_coefficients_matches_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> None:
    """Coefficient count should equal feature count."""

    model = train_logistic_regression(
        X_train,
        y_train,
        config,
    )

    assert model.coef_.shape == (
        1,
        X_train.shape[1],
    )


def test_empty_training_features_raise(
    y_train: pd.Series,
    config: dict,
) -> None:
    """Training feature matrix must not be empty."""

    X_train = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="Training feature matrix is empty",
    ):
        train_logistic_regression(
            X_train,
            y_train,
            config,
        )


def test_empty_training_target_raise(
    X_train: pd.DataFrame,
    config: dict,
) -> None:
    """Training target must not be empty."""

    y_train = pd.Series(
        dtype=int,
        name="ever_90dpd_24m",
    )

    with pytest.raises(
        ValueError,
        match="Training target is empty",
    ):
        train_logistic_regression(
            X_train,
            y_train,
            config,
        )


def test_training_length_mismatch_raises(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> None:
    """Training features and target must contain equal rows."""

    y_train = y_train.iloc[:-1]

    with pytest.raises(
        ValueError,
        match="different numbers of rows",
    ):
        train_logistic_regression(
            X_train,
            y_train,
            config,
        )


def test_single_class_target_raises(
    X_train: pd.DataFrame,
    config: dict,
) -> None:
    """Training target must contain at least two classes."""

    y_train = pd.Series(
        np.zeros(
            len(X_train),
            dtype=int,
        ),
        name="ever_90dpd_24m",
    )

    with pytest.raises(
        ValueError,
        match="at least two classes",
    ):
        train_logistic_regression(
            X_train,
            y_train,
            config,
        )


def test_random_state_is_applied(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> None:
    """Configured random state should be applied to the estimator."""

    model = train_logistic_regression(
        X_train,
        y_train,
        config,
    )

    assert model.random_state == 42


def test_configuration_is_applied(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> None:
    """Estimator should respect modelling configuration."""

    model = train_logistic_regression(
        X_train,
        y_train,
        config,
    )

    assert model.penalty == "l2"
    assert model.C == 1.0
    assert model.solver == "lbfgs"
    assert model.max_iter == 1000
    assert model.class_weight == "balanced"
