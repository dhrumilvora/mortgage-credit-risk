from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy.special import logit
import pandas as pd


def fit_calibration_isotonic(
    y_true: pd.Series,
    y_proba: np.ndarray,
) -> IsotonicRegression:

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if len(y_true) != len(y_proba):
        raise ValueError(
            "Calibration target and predictions must have the same length."
        )

    if len(y_true) == 0:
        raise ValueError("Calibration dataset is empty.")

    if np.unique(y_true).size < 2:
        raise ValueError("Calibration target must contain at least two classes.")

    if not np.isfinite(y_proba).all():
        raise ValueError("Calibration probabilities contain non-finite values.")

    if ((y_proba < 0) | (y_proba > 1)).any():
        raise ValueError("Calibration probabilities must be between 0 and 1.")

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        increasing=True,
        out_of_bounds="clip",
    )

    calibrator.fit(
        y_proba,
        y_true,
    )

    return calibrator


def apply_calibration_isotonic(
    y_proba: np.ndarray,
    calibrator: IsotonicRegression,
) -> np.ndarray:

    y_proba = np.asarray(y_proba)

    if not np.isfinite(y_proba).all():
        raise ValueError("Predicted probabilities contain non-finite values.")

    if ((y_proba < 0) | (y_proba > 1)).any():
        raise ValueError("Predicted probabilities must be between 0 and 1.")

    calibrated_proba = calibrator.predict(
        y_proba,
    )

    return np.asarray(
        calibrated_proba,
        dtype=float,
    )


def fit_calibration_platt(
    y_true: pd.Series,
    y_proba: np.ndarray,
) -> LogisticRegression:

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if len(y_true) != len(y_proba):
        raise ValueError(
            "Calibration target and predictions must have the same length."
        )

    if len(y_true) == 0:
        raise ValueError("Calibration dataset is empty.")

    if np.unique(y_true).size < 2:
        raise ValueError("Calibration target must contain at least two classes.")

    eps = np.finfo(float).eps

    clipped_proba = np.clip(
        y_proba,
        eps,
        1.0 - eps,
    )

    logit_prediction = logit(
        clipped_proba,
    )

    calibration_model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
    )

    calibration_model.fit(
        logit_prediction.reshape(-1, 1),
        y_true,
    )

    return calibration_model


def apply_calibration_platt(
    y_proba: np.ndarray,
    calibration_model: LogisticRegression,
) -> np.ndarray:

    eps = np.finfo(float).eps

    clipped_proba = np.clip(
        y_proba,
        eps,
        1.0 - eps,
    )

    logit_prediction = logit(
        clipped_proba,
    )

    calibrated_proba = calibration_model.predict_proba(logit_prediction.reshape(-1, 1))[
        :, 1
    ]

    return calibrated_proba


def fit_calibration(y_true: pd.Series, y_proba: np.ndarray, config: dict):
    """
    Fit a calibration model to the predicted probabilities and true labels.
    """
    if config["parameters"]["evaluation"]["calibration"]["method"] == "isotonic":
        return fit_calibration_isotonic(y_true, y_proba)
    elif config["parameters"]["evaluation"]["calibration"]["method"] == "platt":
        return fit_calibration_platt(y_true, y_proba)

    raise ValueError(
        f"Unknown calibration method: {config['parameters']['evaluation']['calibration']['method']}"
    )


def apply_calibration(y_proba: np.ndarray, calibration_model, config: dict):
    """
    Apply a calibration model to the predicted probabilities.
    """
    if config["parameters"]["evaluation"]["calibration"]["method"] == "isotonic":
        return apply_calibration_isotonic(y_proba, calibration_model)
    elif config["parameters"]["evaluation"]["calibration"]["method"] == "platt":
        return apply_calibration_platt(y_proba, calibration_model)

    raise ValueError(
        f"Unknown calibration method: {config['parameters']['evaluation']['calibration']['method']}"
    )
