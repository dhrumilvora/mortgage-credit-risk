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

