from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

logger = logging.getLogger(__name__)
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin

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


def fit_calibration_beta(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
) -> BetaCalibrator:
    y_true, y_proba = _validate_calibration_inputs(y_true, y_proba)

    eps = np.finfo(float).eps
    clipped_proba = np.clip(y_proba, eps, 1.0 - eps)

    log_proba = np.log(clipped_proba)
    log_one_minus_proba = np.log1p(-clipped_proba)

    def objective(params: np.ndarray) -> float:
        a, b, c = params

        logits = a * log_proba + b * log_one_minus_proba + c

        # Numerically stable binary log-loss
        return float(np.mean(np.logaddexp(0.0, logits) - y_true * logits))

    result = minimize(
        objective,
        x0=np.array([1.0, 1.0, 0.0]),
        method="L-BFGS-B",
    )

    if not result.success:
        raise RuntimeError(f"Beta calibration failed: {result.message}")

    a, b, c = result.x

    a, b, c = result.x
    
    calibrator = BetaCalibrator()
    calibrator.a_ = float(a)
    calibrator.b_ = float(b)
    calibrator.c_ = float(c)
    
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
    if method == "beta":
        return fit_calibration_beta(y_true, y_proba)

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


def apply_calibration_beta(
    y_proba: np.ndarray,
    calibrator: BetaCalibrator,
) -> np.ndarray:
    y_proba = np.asarray(y_proba, dtype=float)

    if not np.isfinite(y_proba).all():
        raise ValueError("Predicted probabilities contain non-finite values.")

    if ((y_proba < 0) | (y_proba > 1)).any():
        raise ValueError("Predicted probabilities must be between 0 and 1.")

    eps = np.finfo(float).eps
    clipped_proba = np.clip(y_proba, eps, 1.0 - eps)

    logits = (
        calibrator.a_ * np.log(clipped_proba)
        + calibrator.b_ * np.log1p(-clipped_proba)
        + calibrator.c_
    )

    return np.asarray(expit(logits), dtype=float)


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
    if method == "beta":
        return apply_calibration_beta(y_proba, calibration_model)

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


class BetaCalibrator(BaseEstimator, ClassifierMixin):
    """Beta calibration model."""

    def __init__(self):
        self.a_: float | None = None
        self.b_: float | None = None
        self.c_: float | None = None

    def fit(
        self,
        y_proba: np.ndarray,
        y_true: np.ndarray,
    ) -> "BetaCalibrator":
        y_true = np.asarray(y_true, dtype=float)
        y_proba = np.asarray(y_proba, dtype=float)

        eps = np.finfo(float).eps
        clipped_proba = np.clip(y_proba, eps, 1.0 - eps)

        log_proba = np.log(clipped_proba)
        log_one_minus_proba = np.log1p(-clipped_proba)

        def objective(params: np.ndarray) -> float:
            a, b, c = params

            logits = a * log_proba + b * log_one_minus_proba + c

            return float(np.mean(np.logaddexp(0.0, logits) - y_true * logits))

        result = minimize(
            objective,
            x0=np.array([1.0, 1.0, 0.0]),
            method="L-BFGS-B",
        )

        if not result.success:
            raise RuntimeError(f"Beta calibration failed: {result.message}")

        self.a_, self.b_, self.c_ = result.x

        return self

    def predict_proba(
        self,
        y_proba: np.ndarray,
    ) -> np.ndarray:
        y_proba = np.asarray(y_proba, dtype=float)

        if not np.isfinite(y_proba).all():
            raise ValueError("Predicted probabilities contain non-finite values.")

        if ((y_proba < 0) | (y_proba > 1)).any():
            raise ValueError("Predicted probabilities must be between 0 and 1.")

        if self.a_ is None:
            raise RuntimeError("Beta calibrator has not been fitted.")

        eps = np.finfo(float).eps
        clipped_proba = np.clip(
            y_proba,
            eps,
            1.0 - eps,
        )

        logits = (
            self.a_ * np.log(clipped_proba)
            + self.b_ * np.log1p(-clipped_proba)
            + self.c_
        )

        calibrated = expit(logits)

        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(
        self,
        y_proba: np.ndarray,
    ) -> np.ndarray:
        return self.predict_proba(y_proba)[:, 1]


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
        
        
    if method == "beta":
        if not isinstance(calibration_model, BetaCalibrator):
            raise TypeError(
        "Beta calibration requires a BetaCalibrator calibrator."
    )

        return {
        "calibration_method": "beta",
        "a": float(calibration_model.a_),
        "b": float(calibration_model.b_),
        "c": float(calibration_model.c_),
    }


    raise ValueError(f"Unknown calibration method: {method}")
