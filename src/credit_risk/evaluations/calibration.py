from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


def _validate_calibration_inputs(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)

    if y_true.ndim != 1:
        y_true = y_true.ravel()

    if y_proba.ndim != 1:
        y_proba = y_proba.ravel()

    if y_true.shape[0] == 0:
        raise ValueError("Calibration dataset is empty.")

    if y_true.shape[0] != y_proba.shape[0]:
        raise ValueError(
            "Calibration target and predictions must have the same length."
        )

    if np.unique(y_true).size < 2:
        raise ValueError("Calibration target must contain at least two classes.")

    if not np.isfinite(y_proba).all():
        raise ValueError("Calibration probabilities contain non-finite values.")

    if ((y_proba < 0) | (y_proba > 1)).any():
        raise ValueError("Calibration probabilities must be between 0 and 1.")

    return y_true.astype(int), y_proba


# ---------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------


def fit_calibration_isotonic(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
) -> IsotonicRegression:
    y_true, y_proba = _validate_calibration_inputs(y_true, y_proba)

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        increasing=True,
        out_of_bounds="clip",
    )

    calibrator.fit(y_proba, y_true)
    return calibrator


def fit_calibration_platt(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
) -> LogisticRegression:
    y_true, y_proba = _validate_calibration_inputs(y_true, y_proba)

    eps = np.finfo(float).eps
    clipped_proba = np.clip(y_proba, eps, 1.0 - eps)
    logit_prediction = logit(clipped_proba)

    calibrator = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
    )

    calibrator.fit(logit_prediction.reshape(-1, 1), y_true)
    return calibrator


def fit_calibration(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    config: dict,
):
    method = config["parameters"]["evaluation"]["calibration"]["method"].strip().lower()

    if method == "isotonic":
        return fit_calibration_isotonic(y_true, y_proba)

    if method == "platt":
        return fit_calibration_platt(y_true, y_proba)

    raise ValueError(f"Unknown calibration method: {method}")


# ---------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------


def apply_calibration_isotonic(
    y_proba: np.ndarray,
    calibrator: IsotonicRegression,
) -> np.ndarray:
    y_proba = np.asarray(y_proba, dtype=float)

    if not np.isfinite(y_proba).all():
        raise ValueError("Predicted probabilities contain non-finite values.")

    if ((y_proba < 0) | (y_proba > 1)).any():
        raise ValueError("Predicted probabilities must be between 0 and 1.")

    return np.asarray(calibrator.predict(y_proba), dtype=float)


def apply_calibration_platt(
    y_proba: np.ndarray,
    calibrator: LogisticRegression,
) -> np.ndarray:
    y_proba = np.asarray(y_proba, dtype=float)

    if not np.isfinite(y_proba).all():
        raise ValueError("Predicted probabilities contain non-finite values.")

    if ((y_proba < 0) | (y_proba > 1)).any():
        raise ValueError("Predicted probabilities must be between 0 and 1.")

    eps = np.finfo(float).eps
    clipped_proba = np.clip(y_proba, eps, 1.0 - eps)
    logit_prediction = logit(clipped_proba)

    return np.asarray(
        calibrator.predict_proba(logit_prediction.reshape(-1, 1))[:, 1],
        dtype=float,
    )


def apply_calibration(
    y_proba: np.ndarray,
    calibration_model,
    config: dict,
) -> np.ndarray:
    method = config["parameters"]["evaluation"]["calibration"]["method"].strip().lower()

    if method == "isotonic":
        return apply_calibration_isotonic(y_proba, calibration_model)

    if method == "platt":
        return apply_calibration_platt(y_proba, calibration_model)

    raise ValueError(f"Unknown calibration method: {method}")


# ---------------------------------------------------------------------
# Calibration curve / bins
# ---------------------------------------------------------------------


def calculate_calibration_metrics(
    y_true,
    y_proba,
    bins: list[list[float]],
) -> pd.DataFrame:
    y_true, y_proba = _validate_calibration_inputs(y_true, y_proba)

    if not bins:
        raise ValueError("Calibration bins cannot be empty.")

    for lower, upper in bins:
        if lower < 0 or upper > 1 or lower >= upper:
            raise ValueError(f"Invalid calibration bin: [{lower}, {upper}]")

    bin_edges = [bins[0][0], *[upper for _, upper in bins]]

    if not np.isclose(bin_edges[0], 0.0) or not np.isclose(bin_edges[-1], 1.0):
        raise ValueError(
            "Calibration bins must cover the full probability range [0, 1]."
        )

    if any(upper <= lower for lower, upper in zip(bin_edges[:-1], bin_edges[1:])):
        raise ValueError(
            "Calibration bins must be strictly increasing and non-overlapping."
        )

    evaluation_df = pd.DataFrame(
        {
            "y_true": y_true,
            "predicted_pd": y_proba,
        }
    )

    evaluation_df["calibration_bin"] = pd.cut(
        evaluation_df["predicted_pd"],
        bins=bin_edges,
        labels=False,
        include_lowest=True,
        right=True,
    )
    evaluation_df["calibration_bin"] += 1

    calibration_metrics = (
        evaluation_df.dropna(subset=["calibration_bin"])
        .groupby("calibration_bin", observed=True)
        .agg(
            population=("y_true", "size"),
            average_predicted_pd=("predicted_pd", "mean"),
            actual_event_rate=("y_true", "mean"),
        )
        .reset_index()
    )

    calibration_metrics["population_share"] = (
        calibration_metrics["population"] / evaluation_df.shape[0]
    )

    calibration_metrics["calibration_gap"] = (
        calibration_metrics["actual_event_rate"]
        - calibration_metrics["average_predicted_pd"]
    )

    calibration_metrics["absolute_calibration_gap"] = calibration_metrics[
        "calibration_gap"
    ].abs()

    calibration_metrics["calibration_bin"] = calibration_metrics[
        "calibration_bin"
    ].astype(int)

    return calibration_metrics


# ---------------------------------------------------------------------
# Generic calibration statistics
# ---------------------------------------------------------------------


def calculate_calibration_statistics(
    y_true,
    y_proba,
    calibration_metrics: pd.DataFrame | None = None,
) -> dict:
    """Calculate calibration statistics independent of calibrator type."""
    y_true, y_proba = _validate_calibration_inputs(y_true, y_proba)

    mean_predicted_pd = float(np.mean(y_proba))
    actual_event_rate = float(np.mean(y_true))

    observed_expected_ratio = (
        actual_event_rate / mean_predicted_pd if mean_predicted_pd > 0 else None
    )

    statistics = {
        "mean_predicted_pd": mean_predicted_pd,
        "actual_event_rate": actual_event_rate,
        "observed_expected_ratio": observed_expected_ratio,
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba, labels=[0, 1])),
    }

    if calibration_metrics is not None and not calibration_metrics.empty:
        weights = (
            calibration_metrics["population"] / calibration_metrics["population"].sum()
        )
        absolute_gaps = calibration_metrics["absolute_calibration_gap"]
        statistics["ece"] = float(np.sum(weights * absolute_gaps))
        statistics["mce"] = float(absolute_gaps.max())
    else:
        statistics["ece"] = None
        statistics["mce"] = None

    return statistics


# ---------------------------------------------------------------------
# Model-specific calibration parameters
# ---------------------------------------------------------------------


def calculate_calibration_summary(
    calibration_model,
    method: str,
) -> dict:
    """Return model-specific metadata for the fitted calibration model."""
    method = method.strip().lower()

    if method == "platt":
        if not isinstance(calibration_model, LogisticRegression):
            raise TypeError(
                "Platt calibration requires a LogisticRegression calibrator."
            )

        return {
            "calibration_method": "platt",
            "calibration_intercept": float(calibration_model.intercept_[0]),
            "calibration_slope": float(calibration_model.coef_[0, 0]),
        }

    if method == "isotonic":
        if not isinstance(calibration_model, IsotonicRegression):
            raise TypeError(
                "Isotonic calibration requires an IsotonicRegression calibrator."
            )

        thresholds = getattr(calibration_model, "X_thresholds_", None)
        calibrated_values = getattr(calibration_model, "y_thresholds_", None)

        return {
            "calibration_method": "isotonic",
            "n_thresholds": int(len(thresholds)) if thresholds is not None else None,
            "min_calibrated_pd": (
                float(np.min(calibrated_values))
                if calibrated_values is not None
                else None
            ),
            "max_calibrated_pd": (
                float(np.max(calibrated_values))
                if calibrated_values is not None
                else None
            ),
        }

    raise ValueError(f"Unknown calibration method: {method}")
