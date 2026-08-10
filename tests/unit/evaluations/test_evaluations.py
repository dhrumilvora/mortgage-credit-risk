import numpy as np
import pytest

from credit_risk.evaluations.evaluations import evaluate_dataset, generate_predictions


class IdentityPreprocessor:
    def transform(self, X):
        return X


class FixedProbabilityModel:
    def predict_proba(self, X):
        return np.column_stack([1 - X[:, 0], X[:, 0]])


def test_generate_predictions_applies_threshold() -> None:
    y_pred, y_proba = generate_predictions(
        FixedProbabilityModel(), IdentityPreprocessor(), np.array([[0.2], [0.7]]), threshold=0.5
    )
    assert y_pred.tolist() == [0, 1]
    assert y_proba.tolist() == [0.2, 0.7]


def test_evaluate_dataset_returns_complete_evaluation_suite() -> None:
    result = evaluate_dataset(
        y_true=np.array([0, 0, 1, 1]),
        y_pred=np.array([0, 0, 1, 1]),
        y_proba=np.array([0.1, 0.3, 0.7, 0.9]),
        n_deciles=2,
        calibration_bins=[[0.0, 0.5], [0.5, 1.0]],
    )
    assert {"ds_metrics", "confusion_matrix", "credit_risk_metrics", "risk_deciles", "calibration", "roc_curve", "ks_curve"} <= result.keys()
    assert result["ds_metrics"]["roc_auc"] == 1.0
    assert result["confusion_matrix"]["true_positive"] == 2


def test_evaluate_dataset_rejects_single_class_target() -> None:
    with pytest.raises(ValueError, match="at least two classes"):
        evaluate_dataset(
            y_true=np.array([0, 0]), y_pred=np.array([0, 0]), y_proba=np.array([0.1, 0.2]),
            n_deciles=2, calibration_bins=[[0.0, 1.0]],
        )
