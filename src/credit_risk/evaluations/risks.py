from __future__ import annotations
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def calculate_credit_risk_metrics(y_true, y_proba) -> dict:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if y_true.shape[0] == 0:
        raise ValueError("Evaluation target is empty.")

    if y_true.shape[0] != y_proba.shape[0]:
        raise ValueError(
            "Evaluation target and predicted probabilities contain "
            "different numbers of rows."
        )

    if np.unique(y_true).size < 2:
        raise ValueError("Evaluation target must contain at least two classes.")

    if not np.isfinite(y_proba).all():
        raise ValueError("Predicted probabilities contain non-finite values.")

    if ((y_proba < 0) | (y_proba > 1)).any():
        raise ValueError("Predicted probabilities must be between 0 and 1.")
    actual_event_rate = float(y_true.mean())
    average_predicted_df = float(y_proba.mean())
    metrics = {
        "actual_event_rate": actual_event_rate,
        "average_predicted_df": average_predicted_df,
    }
    logger.info(
        "Credit-risk metrics calculated: "
        "actual_event_rate=%.6f average_predicted_df=%.6f",
        actual_event_rate,
        average_predicted_df,
    )

    return metrics


def calculate_risk_deciles(y_true, y_proba, n_deciles: int = 10) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if y_true.shape[0] == 0:
        raise ValueError("Evaluation target is empty.")

    if y_true.shape[0] != y_proba.shape[0]:
        raise ValueError(
            "Evaluation target and predicted probabilities contain "
            "different numbers of rows."
        )

    if np.unique(y_true).size < 2:
        raise ValueError("Evaluation target must contain at least two classes.")

    if not np.isfinite(y_proba).all():
        raise ValueError("Predicted probabilities contain non-finite values.")

    if ((y_proba < 0) | (y_proba > 1)).any():
        raise ValueError("Predicted probabilities must be between 0 and 1.")

    if n_deciles < 2:
        raise ValueError("Number of risk deciles must be at least 2.")

    evaluation_df = pd.DataFrame(
        {
            "y_true": y_true,
            "predicted_df": y_proba,
        }
    )

    evaluation_df["risk_decile"] = pd.qcut(
        evaluation_df["predicted_df"], q=n_deciles, labels=False, duplicates="drop"
    )
    evaluation_df["risk_decile"] = evaluation_df["risk_decile"] + 1
    total_population = evaluation_df.shape[0]
    total_events = evaluation_df["y_true"].sum()

    decile_metrics = evaluation_df.groupby("risk_decile", as_index=False).agg(
        population=("y_true", "size"),
        events=("y_true", "sum"),
        average_predicted_df=("predicted_df", "mean"),
        actual_event_rate=("y_true", "mean"),
    )
    decile_metrics["population_share"] = decile_metrics["population"] / total_population

    decile_metrics["event_share"] = (
        decile_metrics["events"] / total_events if total_events > 0 else 0.0
    )

    decile_metrics["cumulative_event_share"] = decile_metrics["event_share"].cumsum()

    overall_event_rate = total_events / total_population
    decile_metrics["lift"] = (
        decile_metrics["actual_event_rate"] / overall_event_rate
        if overall_event_rate > 0
        else np.nan
    )

    logger.info(
        "Risk deciles calculated: rows=%s deciles=%s",
        f"{total_population:,}",
        decile_metrics["risk_decile"].nunique(),
    )

    return decile_metrics


def calculate_calibration_metrics(
    y_true,
    y_proba,
    bins: list[list[float]],
) -> pd.DataFrame:

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if y_true.shape[0] == 0:
        raise ValueError("Evaluation target is empty.")

    if y_true.shape[0] != y_proba.shape[0]:
        raise ValueError(
            "Evaluation target and predicted probabilities contain "
            "different numbers of rows."
        )

    if np.unique(y_true).size < 2:
        raise ValueError("Evaluation target must contain at least two classes.")

    if not np.isfinite(y_proba).all():
        raise ValueError("Predicted probabilities contain non-finite values.")

    if ((y_proba < 0) | (y_proba > 1)).any():
        raise ValueError("Predicted probabilities must be between 0 and 1.")

    if not bins:
        raise ValueError("Calibration bins cannot be empty.")

    for lower, upper in bins:
        if lower < 0 or upper > 1 or lower >= upper:
            raise ValueError(f"Invalid calibration bin: [{lower}, {upper}]")

    evaluation_df = pd.DataFrame(
        {
            "y_true": y_true,
            "predicted_pd": y_proba,
        }
    )

    bin_edges = [
        bins[0][0],
        *[upper for _, upper in bins],
    ]

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
        .groupby(
            "calibration_bin",
            observed=True,
        )
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

    calibration_metrics["calibration_bin"] = calibration_metrics[
        "calibration_bin"
    ].astype(int)

    logger.info(
        "Calibration metrics calculated: rows=%s bins=%s",
        f"{evaluation_df.shape[0]:,}",
        calibration_metrics.shape[0],
    )

    return calibration_metrics


def calculate_calibration_summary(
    y_true,
    y_proba,
) -> dict:
    """
    Calculate global calibration intercept and slope.

    The calibration model is:

        logit(P(Y=1)) = intercept + slope * logit(predicted_probability)

    Ideal values:
        intercept = 0
        slope = 1
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if y_true.shape[0] == 0:
        raise ValueError("Evaluation target is empty.")

    if y_true.shape[0] != y_proba.shape[0]:
        raise ValueError(
            "Evaluation target and predicted probabilities contain "
            "different numbers of rows."
        )

    if np.unique(y_true).size < 2:
        raise ValueError(
            "Calibration assessment requires both positive and negative observations."
        )

    if not np.isfinite(y_proba).all():
        raise ValueError("Predicted probabilities contain non-finite values.")

    if ((y_proba < 0) | (y_proba > 1)).any():
        raise ValueError("Predicted probabilities must be between 0 and 1.")

    # Avoid infinite logits for probabilities exactly equal to 0 or 1.
    clipped_proba = np.clip(
        y_proba,
        1e-15,
        1 - 1e-15,
    )

    logit_proba = np.log(clipped_proba / (1 - clipped_proba)).reshape(-1, 1)

    from sklearn.linear_model import LogisticRegression

    calibration_model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
    )

    calibration_model.fit(
        logit_proba,
        y_true,
    )

    intercept = float(calibration_model.intercept_[0])
    slope = float(calibration_model.coef_[0, 0])

    logger.info(
        "Calibration summary calculated: intercept=%.6f slope=%.6f",
        intercept,
        slope,
    )

    return {
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }
