import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from credit_risk.evaluations.shap import (
    _get_shap_config,
    _get_shap_dir,
    build_shap_importance,
    compute_shap_values,
)


class IdentityPreprocessor:
    def transform(self, X):
        return X.to_numpy()

    def get_feature_names_out(self):
        return np.array(["feature_a", "feature_b"])


class XGBoostLikeModel:
    __module__ = "xgboost.sklearn"

    def predict_proba(self, X):
        return np.column_stack([1 - X[:, 0], X[:, 0]])


def test_shap_config_is_read_from_evaluation_section() -> None:
    config = {"parameters": {"evaluation": {"shap": {"enabled": False}}}}
    assert _get_shap_config(config) == {"enabled": False}


def test_xgboost_uses_tree_explainer(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    class FakeTreeExplainer:
        def __init__(self, model, **kwargs):
            calls["tree"] = kwargs

        def __call__(self, X):
            return SimpleNamespace(values=np.ones_like(X))

    fake_shap = SimpleNamespace(TreeExplainer=FakeTreeExplainer)
    monkeypatch.setitem(sys.modules, "shap", fake_shap)

    values, transformed = compute_shap_values(
        XGBoostLikeModel(),
        IdentityPreprocessor(),
        pd.DataFrame({"feature_a": [0.1, 0.2], "feature_b": [1.0, 2.0]}),
        background_size=1,
        max_samples=2,
    )

    assert values.shape == (2, 2)
    assert transformed.columns.tolist() == ["feature_a", "feature_b"]
    assert calls["tree"]["model_output"] == "raw"
    assert calls["tree"]["feature_perturbation"] == "tree_path_dependent"


def test_shap_output_path_uses_same_run_model_reference(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = {
        "catalog": {"base": tmp_path, "model_evaluation": {"folder_name": "artifacts", "file_name": "evaluation", "file_type": "folder"}},
        "parameters": {
            "modelling": {"version": "same-run", "algorithm": "xgboost"},
            "evaluation": {
                "mode": "same_run",
                "model": {"version": "other", "type": "logistic_regression"},
            },
        },
    }
    calls = {}
    monkeypatch.setattr(
        "credit_risk.evaluations.shap.create_path",
        lambda base, catalog, key, *parts, **kwargs: calls.setdefault("parts", parts) and tmp_path,
    )

    result = _get_shap_dir(config, "validation")

    assert calls["parts"] == ("same-run", "xgboost")
    assert result == tmp_path / "shap" / "validation"


def test_zero_shap_values_have_zero_percentage_importance() -> None:
    importance = build_shap_importance(np.zeros((2, 2)), ["a", "b"])
    assert importance["mean_abs_shap_pct"].tolist() == [0.0, 0.0]
