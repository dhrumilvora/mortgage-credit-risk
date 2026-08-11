from pathlib import Path

import pandas as pd
import pytest

from credit_risk.pipelines.evaluation import run_evaluation_pipeline


def test_existing_model_uses_saved_feature_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        "catalog": {"base": "data"},
        "parameters": {
            "target": {"name": "ever_90dpd_24m"},
            "modelling": {
                "version": "current", "algorithm": "logistic_regression",
                "features": {
                    "numerical_features": ["current_feature"],
                    "categorical_features": [],
                    "engineered_features": [],
                },
            },
            "evaluation": {
                "skip": False,
                "mode": "existing_model",
                "model": {"version": "persisted", "type": "logistic_regression"},
                "datasets": {"validation": True, "oot": False},
            },
        },
    }
    calls = {}
    saved_features = {
        "numerical_features": ["saved_feature"],
        "categorical_features": [],
        "engineered_features": [],
    }

    monkeypatch.setattr(
        "credit_risk.pipelines.evaluation.load_training_config",
        lambda model_config: {"features": saved_features},
    )
    monkeypatch.setattr(
        "credit_risk.pipelines.evaluation.load_model_artifacts",
        lambda model_config: calls.setdefault("artifact_config", model_config) and (object(), object()),
    )
    monkeypatch.setattr(
        "credit_risk.pipelines.evaluation.create_path",
        lambda *args, **kwargs: Path("validation.parquet"),
    )
    monkeypatch.setattr(
        "credit_risk.pipelines.evaluation.pd.read_parquet",
        lambda path: pd.DataFrame({"saved_feature": [1], "ever_90dpd_24m": [0]}),
    )

    def fake_evaluate_split(model, preprocessor, df, config):
        calls["scoring_features"] = config["parameters"]["modelling"]["features"]
        return {"ok": True}

    monkeypatch.setattr("credit_risk.pipelines.evaluation.evaluate_split", fake_evaluate_split)
    monkeypatch.setattr(
        "credit_risk.pipelines.evaluation.save_evaluation_results",
        lambda **kwargs: calls.setdefault("saved", kwargs),
    )

    run_evaluation_pipeline(config)

    assert calls["artifact_config"]["parameters"]["modelling"]["version"] == "persisted"
    assert calls["scoring_features"] == saved_features
    assert calls["saved"]["validation_evaluation"] == {"ok": True}
