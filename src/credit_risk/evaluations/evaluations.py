from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve

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
    model,
    preprocessor,
    X,
    threshold: float = 0.5,
    engine: str = "pandas",
    config: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate binary predictions and positive-class probabilities.

    Pandas:
        Uses sklearn preprocessor + model.

    PySpark:
        Uses fitted Spark PipelineModel + Spark ML model and collects
        only prediction/probability columns back to Pandas.
    """

    if not 0 <= threshold <= 1:
        raise ValueError(
            f"Prediction threshold must be between 0 and 1, got {threshold}."
        )

    if engine == "pandas":
        return _generate_predictions_pandas(
            model=model,
            preprocessor=preprocessor,
            X=X,
            threshold=threshold,
        )

    if engine == "pyspark":
        return _generate_predictions_pyspark(
            model=model,
            preprocessor=preprocessor,
            X=X,
            threshold=threshold,
            config=config,
        )

    raise ValueError(f"Unsupported evaluation engine: {engine}")


def _generate_predictions_pandas(
    model,
    preprocessor,
    X: pd.DataFrame,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions using sklearn/Pandas."""

    if X.empty:
        raise ValueError("Prediction feature matrix is empty.")

    X_transformed = preprocessor.transform(
        X,
    )

    y_proba = np.asarray(
        model.predict_proba(
            X_transformed,
        )[:, 1]
    )

    y_pred = (y_proba >= threshold).astype(int)

    _validate_predictions(
        n_rows=len(X),
        y_pred=y_pred,
        y_proba=y_proba,
    )

    logger.info(
        "Pandas predictions generated: rows=%s",
        f"{len(X):,}",
    )

    return y_pred, y_proba


def _generate_predictions_pyspark(
    model,
    preprocessor,
    X,
    threshold: float,
    config: dict | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions using Spark ML."""

    if config is None:
        raise ValueError("Configuration is required for PySpark evaluation.")

    from pyspark.sql import functions as F

    if X.limit(1).count() == 0:
        raise ValueError("Prediction feature matrix is empty.")

    # The fitted Spark preprocessing PipelineModel owns the
    # preprocessing logic. No fitting occurs during evaluation.
    transformed = preprocessor.transform(
        X,
    )

    predictions = model.transform(
        transformed,
    )

    predictions = predictions.withColumn(
        "__y_proba",
        F.col("probability")[1],
    )

    predictions = predictions.withColumn(
        "__y_pred",
        (F.col("__y_proba") >= F.lit(threshold)).cast("int"),
    )

    result = predictions.select(
        "__y_pred",
        "__y_proba",
    ).toPandas()

    if result.empty:
        raise ValueError("Spark prediction result is empty.")

    y_pred = result["__y_pred"].to_numpy(
        dtype="int8",
    )

    y_proba = result["__y_proba"].to_numpy(
        dtype="float64",
    )

    _validate_predictions(
        n_rows=len(result),
        y_pred=y_pred,
        y_proba=y_proba,
    )

    logger.info(
        "PySpark predictions generated: rows=%s",
        f"{len(result):,}",
    )

    return y_pred, y_proba


def _validate_predictions(
    n_rows: int,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> None:

    if y_pred.shape[0] != n_rows:
        raise ValueError(
            "Predictions contain a different number of rows "
            "than the input feature matrix."
        )

    if y_proba.shape[0] != n_rows:
        raise ValueError(
            "Predicted probabilities contain a different number "
            "of rows than the input feature matrix."
        )

    if not np.isfinite(y_proba).all():
        raise ValueError("Predicted probabilities contain non-finite values.")

    if ((y_proba < 0) | (y_proba > 1)).any():
        raise ValueError("Predicted probabilities must be between 0 and 1.")


def evaluate_dataset(
    y_true,
    y_pred,
    y_proba,
    n_deciles: int,
    calibration_bins: list[list[float]],
) -> dict:
    """Run the complete evaluation suite for a single dataset."""

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

    calibration_summary = calculate_calibration_summary(
        y_true=y_true,
        y_proba=y_proba,
    )

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

    if top_fractions is None:
        top_fractions = [
            0.05,
            0.10,
            0.20,
        ]

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
                "population_share": (top_n / total_population),
                "events_captured": int(events_captured),
                "event_capture_rate": (
                    events_captured / total_events if total_events > 0 else np.nan
                ),
                "precision": (top_population["actual"].mean()),
                "average_predicted_pd": (top_population["predicted_pd"].mean()),
                "lift": (
                    top_population["actual"].mean() / evaluation_df["actual"].mean()
                    if evaluation_df["actual"].mean() > 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(results)


def evaluate_thresholds(
    y_true: pd.Series,
    y_proba: pd.Series,
    config: dict,
) -> pd.DataFrame:

    threshold_config = config["parameters"]["evaluation"]["threshold_selection"]

    if not threshold_config["enabled"]:
        return pd.DataFrame()

    thresholds = threshold_config["candidate_thresholds"]

    if not thresholds:
        raise ValueError(
            "evaluation.threshold_selection.candidate_thresholds " "cannot be empty."
        )

    y_true_array = np.asarray(y_true)
    y_proba_array = np.asarray(y_proba)

    if y_true_array.ndim != 1:
        raise ValueError("y_true must be one-dimensional.")

    if y_proba_array.ndim != 1:
        raise ValueError("y_proba must be one-dimensional.")

    if len(y_true_array) != len(y_proba_array):
        raise ValueError("y_true and y_proba must contain the same number of rows.")

    if np.isnan(y_proba_array).any():
        raise ValueError("Predicted probabilities contain missing values.")

    if ((y_proba_array < 0) | (y_proba_array > 1)).any():
        raise ValueError("Predicted probabilities must lie between 0 and 1.")

    if pd.isna(y_true_array).any():
        raise ValueError("Target contains missing values.")

    total_population = len(
        y_true_array,
    )

    total_events = int(
        y_true_array.sum(),
    )

    if total_population == 0:
        raise ValueError("Threshold evaluation dataset is empty.")

    if total_events == 0:
        raise ValueError("Threshold evaluation requires at least one positive event.")

    results = []

    for threshold in thresholds:

        threshold = float(threshold)

        if not 0 < threshold < 1:
            raise ValueError(
                "Each candidate threshold must be strictly " "between 0 and 1."
            )

        y_pred = (y_proba_array >= threshold).astype("int8")

        population_flagged = int(
            y_pred.sum(),
        )

        true_positive = int(((y_true_array == 1) & (y_pred == 1)).sum())

        false_positive = int(((y_true_array == 0) & (y_pred == 1)).sum())

        false_negative = int(((y_true_array == 1) & (y_pred == 0)).sum())

        precision = precision_score(
            y_true_array,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_true_array,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_true_array,
            y_pred,
            zero_division=0,
        )

        results.append(
            {
                "threshold": threshold,
                "population_flagged": population_flagged,
                "population_flagged_pct": (population_flagged / total_population),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": precision,
                "recall": recall,
                "event_capture_rate": recall,
                "f1": f1,
            }
        )

    return pd.DataFrame(results).sort_values("threshold").reset_index(drop=True)
