from __future__ import annotations
import logging
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,

)

import numpy as np

logger = logging.getLogger(__name__)


def calculate_ds_metrics(y_true, y_pred, y_proba) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)
    if y_true.shape[0] == 0:
        raise ValueError("Evaluation target is empty.")

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            "Evaluation target and predictions contain " "different numbers of rows."
        )

    if y_true.shape[0] != y_proba.shape[0]:
        raise ValueError(
            "Evaluation target and predicted probabilities contain "
            "different numbers of rows."
        )

    if np.unique(y_true).size < 2:
        raise ValueError("Evaluation target must contain at least two classes.")
    metrics = {
        "roc_auc": roc_auc_score(
            y_true,
            y_proba,
        ),
        "pr_auc": average_precision_score(
            y_true,
            y_proba,
        ),
        "log_loss": log_loss(
            y_true,
            y_proba,
        ),
        "brier_score": brier_score_loss(
            y_true,
            y_proba,
        ),
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }

    logger.info(
        "DS evaluation metrics calculated: " "rows=%s roc_auc=%.6f pr_auc=%.6f",
        f"{y_true.shape[0]:,}",
        metrics["roc_auc"],
        metrics["pr_auc"],
    )
    return metrics


def calculate_ks(
    y_true,
    y_proba,
) -> float:
    """
    Calculate Kolmogorov-Smirnov statistic.

    KS is only defined when both positive and negative
    observations are present.
    """

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    positive_sorted = np.sort(y_proba[y_true == 1])

    negative_sorted = np.sort(y_proba[y_true == 0])

    n_positive = positive_sorted.shape[0]
    n_negative = negative_sorted.shape[0]

    if n_positive == 0 or n_negative == 0:
        raise ValueError(
            "KS calculation requires both positive and "
            "negative observations. "
            f"positive={n_positive:,}, "
            f"negative={n_negative:,}"
        )

    positive_cdf = (
        np.searchsorted(
            positive_sorted,
            y_proba,
            side="right",
        )
        / n_positive
    )

    negative_cdf = (
        np.searchsorted(
            negative_sorted,
            y_proba,
            side="right",
        )
        / n_negative
    )

    ks = float(np.max(np.abs(positive_cdf - negative_cdf)))

    logger.info(
        "KS calculated: rows=%s ks=%.6f",
        f"{y_true.shape[0]:,}",
        ks,
    )

    return ks


def calculate_confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape[0] == 0:
        raise ValueError("Evaluation target is empty.")

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            "Evaluation target and predictions contain " "different numbers of rows."
        )

    if np.unique(y_true).size < 2:
        raise ValueError("Evaluation target must contain at least two classes.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Evaluation target must contain only binary classes.")

    if not np.isin(y_pred, [0, 1]).all():
        raise ValueError("Predictions must contain only binary classes.")

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    confusion_matrix = {
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }

    logger.info(
        "Confusion matrix calculated: " "tn=%s fp=%s fn=%s tp=%s",
        tn,
        fp,
        fn,
        tp,
    )

    return confusion_matrix
