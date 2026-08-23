from __future__ import annotations
import numpy as np
import pandas as pd
import logging
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from credit_risk.evaluations.metrics import (
    calculate_confusion_matrix,
    calculate_ds_metrics,
    calculate_ks,
)
from credit_risk.evaluations.risks import (
    calculate_calibration_metrics,
    calculate_credit_risk_metrics,
    calculate_risk_deciles,
    calculate_calibration_summary,
)

logger = logging.getLogger(__name__)


def generate_predictions(
    model, preprocessor, X, threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    if X.shape[0] == 0:
        raise ValueError("Prediction feature matrix is empty.")
    if not 0 <= threshold <= 1:
        raise ValueError(
            f"Prediction threshold must be between 0 and 1, got {threshold}."
        )
    X_transformed = preprocessor.transform(X)

    y_proba = np.asarray(model.predict_proba(X_transformed)[:, 1])
    y_pred = (y_proba >= threshold).astype(int)

    if y_pred.shape[0] != X.shape[0]:
        raise ValueError(
            "Predictions contain a different number of rows "
            "than the input feature matrix."
        )

    if y_proba.shape[0] != X.shape[0]:
        raise ValueError(
            "Predicted probabilities contain a different number "
            "of rows than the input feature matrix."
        )

    if not np.isfinite(y_proba).all():
        raise ValueError("Predicted probabilities contain non-finite values.")

    if ((y_proba < 0) | (y_proba > 1)).any():
        raise ValueError("Predicted probabilities must be between 0 and 1.")

    logger.info(
        "Predictions generated: rows=%s",
        f"{X.shape[0]:,}",
    )

    return y_pred, y_proba


def evaluate_dataset(
    y_true,
    y_pred,
    y_proba,
    n_deciles: int,
    calibration_bins: list[list[float]],
) -> dict:
    """
    Run the complete evaluation suite for a single dataset.

    Parameters
    ----------
    y_true
        Actual binary target values.
    y_pred
        Predicted binary class labels.
    y_proba
        Predicted probability of the positive class.
    n_deciles
        Number of risk deciles.
    calibration_bins
        Fixed probability bins used for calibration.

    Returns
    -------
    dict
        Complete evaluation results.
    """

    logger.info(
        "Starting dataset evaluation: rows=%s",
        f"{len(y_true):,}",
    )

    ds_metrics = calculate_ds_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
    )

    ds_metrics["ks"] = calculate_ks(
        y_true=y_true,
        y_proba=y_proba,
    )

    confusion_matrix = calculate_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
    )

    credit_risk_metrics = calculate_credit_risk_metrics(
        y_true=y_true,
        y_proba=y_proba,
    )

    risk_deciles = calculate_risk_deciles(
        y_true=y_true,
        y_proba=y_proba,
        n_deciles=n_deciles,
    )

    calibration = calculate_calibration_metrics(
        y_true=y_true,
        y_proba=y_proba,
        bins=calibration_bins,
    )

    calibration_summary = calculate_calibration_summary(y_true=y_true, y_proba=y_proba)

    roc_curve_data = calculate_roc_curve_data(
        y_true=y_true,
        y_proba=y_proba,
    )

    ks_curve_data = calculate_ks_curve_data(
        y_true=y_true,
        y_proba=y_proba,
    )

    results = {
        "ds_metrics": ds_metrics,
        "confusion_matrix": confusion_matrix,
        "credit_risk_metrics": credit_risk_metrics,
        "risk_deciles": risk_deciles,
        "calibration": calibration,
        "calibration_summary": calibration_summary,
        "roc_curve": roc_curve_data,
        "ks_curve": ks_curve_data,
    }

    logger.info(
        "Dataset evaluation completed: roc_auc=%.4f ks=%.4f",
        ds_metrics["roc_auc"],
        ds_metrics["ks"],
    )

    return results


def calculate_roc_curve_data(
    y_true,
    y_proba,
) -> dict:
    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_proba,
    )

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }


def calculate_ks_curve_data(
    y_true,
    y_proba,
) -> pd.DataFrame:

    data = pd.DataFrame(
        {
            "y_true": np.asarray(y_true),
            "y_proba": np.asarray(y_proba),
        }
    )

    data = data.sort_values(
        "y_proba",
        ascending=False,
    ).reset_index(drop=True)

    total_bad = (data["y_true"] == 1).sum()
    total_good = (data["y_true"] == 0).sum()

    if total_bad == 0 or total_good == 0:
        raise ValueError("KS curve requires both positive and negative observations.")

    data["cum_bad"] = (data["y_true"] == 1).cumsum() / total_bad

    data["cum_good"] = (data["y_true"] == 0).cumsum() / total_good

    data["ks"] = (data["cum_bad"] - data["cum_good"]).abs()

    data["population_pct"] = np.arange(1, len(data) + 1) / len(data)

    return data[
        [
            "population_pct",
            "cum_bad",
            "cum_good",
            "ks",
        ]
    ]


def calculate_top_k_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    top_fractions: list[float] | None = None,
) -> pd.DataFrame:
    """
    Calculate event capture and precision at the top-risk portion
    of the population.

    Rows are ranked from highest predicted risk to lowest.
    """
    if top_fractions is None:
        top_fractions = [0.05, 0.10, 0.20]

    evaluation_df = (
        pd.DataFrame(
            {
                "actual": y_true.to_numpy(),
                "predicted_pd": y_pred,
            }
        )
        .sort_values(
            "predicted_pd",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    total_events = evaluation_df["actual"].sum()
    total_population = len(evaluation_df)

    results = []

    for fraction in top_fractions:
        top_n = max(
            1,
            int(np.ceil(total_population * fraction)),
        )

        top_population = evaluation_df.iloc[:top_n]

        events_captured = top_population["actual"].sum()

        results.append(
            {
                "top_fraction": fraction,
                "population": top_n,
                "population_share": top_n / total_population,
                "events_captured": int(events_captured),
                "event_capture_rate": (
                    events_captured / total_events if total_events > 0 else np.nan
                ),
                "precision": top_population["actual"].mean(),
                "average_predicted_pd": (top_population["predicted_pd"].mean()),
                "lift": (
                    top_population["actual"].mean() / evaluation_df["actual"].mean()
                    if evaluation_df["actual"].mean() > 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(results)


def fit_calibration(
    y_true: pd.Series,
    y_proba: np.ndarray,
) -> tuple[float, float]:
    """
    Fit logistic calibration on a validation dataset.

    Returns:
        calibration_intercept
        calibration_slope
    """
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

    # Keep probabilities strictly inside (0, 1) before logit.
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

    return (
        float(calibration_model.intercept_[0]),
        float(calibration_model.coef_[0][0]),
    )


def apply_calibration(
    y_proba: np.ndarray,
    intercept: float,
    slope: float,
) -> np.ndarray:
    """
    Apply a fitted logistic calibration mapping.
    """
    eps = np.finfo(float).eps

    clipped_proba = np.clip(
        y_proba,
        eps,
        1.0 - eps,
    )

    logit_prediction = logit(
        clipped_proba,
    )

    calibrated_logit = intercept + slope * logit_prediction

    return expit(
        calibrated_logit,
    )
