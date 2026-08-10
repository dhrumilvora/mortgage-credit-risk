"""Tests for modelling pipeline orchestration."""

import pandas as pd
import pytest

from credit_risk.pipelines.modelling import run_modelling_pipeline


@pytest.fixture
def config() -> dict:
    """Minimal configuration required by the modelling pipeline."""

    return {
        "catalog": {
            "base": "unused",
        },
        "parameters": {
            "target": {
                "name": "ever_90dpd_24m",
            },
            "modelling": {
                "skip": False,
                "vintages_train": [2015],
                "vintages_oot": [2016],
                "version": "unit-test",
                "algorithm": "logistic_regression",
                "validation_size": 0.20,
                "random_state": 42,
                "stratify": True,
                "logistic_regression": {
                    "penalty": "l2",
                    "C": 1.0,
                    "solver": "lbfgs",
                    "max_iter": 1000,
                    "class_weight": "balanced",
                },
            },
        },
    }


@pytest.fixture
def development_df() -> pd.DataFrame:
    """Synthetic development population."""

    return pd.DataFrame(
        {
            "loan_id": [1, 2, 3, 4],
            "vintage": [2015] * 4,
            "ever_90dpd_24m": [0, 0, 0, 1],
        }
    )


def test_modelling_pipeline_orchestration(
    monkeypatch: pytest.MonkeyPatch,
    config: dict,
    development_df: pd.DataFrame,
) -> None:
    """Pipeline should orchestrate the modelling workflow."""

    train_df = development_df.iloc[:3].copy()
    validation_df = development_df.iloc[3:].copy()

    X_train = pd.DataFrame(
        {
            "credit_score": [700, 720, 680],
        }
    )

    y_train = pd.Series(
        [0, 0, 1],
        name="ever_90dpd_24m",
    )

    X_validation = pd.DataFrame(
        {
            "credit_score": [650],
        }
    )

    y_validation = pd.Series(
        [1],
        name="ever_90dpd_24m",
    )

    calls = {
        "writes": [],
        "path_keys": [],
    }

    oot_df = development_df.assign(vintage=2016)

    def mock_load(config_arg, vintages):
        calls["config"] = config_arg
        calls.setdefault("vintages", []).append(vintages)
        return development_df if vintages == [2015] else oot_df

    def mock_split(df, config_arg):
        calls["split_df"] = df
        calls["split_config"] = config_arg
        return train_df, validation_df

    def mock_split_features_target(df, config_arg):
        if df is train_df:
            return X_train, y_train

        if df is validation_df:
            return X_validation, y_validation

        pytest.fail("Unexpected dataframe passed to split_features_target.")

    class MockPreprocessor:
        def __init__(self):
            self.fit_transform_called = False
            self.transform_called = False

        def fit_transform(self, X):
            self.fit_transform_called = True
            calls["fit_transform_input"] = X
            return X

        def transform(self, X):
            self.transform_called = True
            calls["transform_input"] = X
            return X

    preprocessor = MockPreprocessor()

    def mock_build_preprocessor(config_arg):
        return preprocessor

    def mock_create_path(
        base_path,
        catalog,
        key,
        *subfolders,
        must_exist=True,
    ):
        calls["path_keys"].append(key)
        return f"/tmp/{key}.parquet"

    def mock_write_parquet(df, path):
        calls["writes"].append((df, path))

    monkeypatch.setattr(
        "credit_risk.pipelines.modelling.load_modelling_vintage",
        mock_load,
    )

    monkeypatch.setattr(
        "credit_risk.pipelines.modelling.stratified_data_split",
        mock_split,
    )

    monkeypatch.setattr(
        "credit_risk.pipelines.modelling.split_features_target",
        mock_split_features_target,
    )

    monkeypatch.setattr(
        "credit_risk.pipelines.modelling.build_preprocessor",
        mock_build_preprocessor,
    )

    monkeypatch.setattr(
        "credit_risk.pipelines.modelling.create_path",
        mock_create_path,
    )

    monkeypatch.setattr(
        "credit_risk.pipelines.modelling.write_parquet",
        mock_write_parquet,
    )

    monkeypatch.setattr(
        "credit_risk.pipelines.modelling.train_logistic_regression",
        lambda *args: "model",
    )

    monkeypatch.setattr(
        "credit_risk.pipelines.modelling.save_artifacts",
        lambda *args: calls.setdefault("artifacts_saved", True),
    )

    result = run_modelling_pipeline(config)

    assert result is None

    assert calls["config"] is config
    assert calls["vintages"] == [[2015], [2016]]

    assert calls["split_df"] is development_df
    assert calls["split_config"] is config

    assert calls["path_keys"] == [
        "train_df",
        "validation_df",
        "oot_df",
    ]

    assert len(calls["writes"]) == 3

    assert calls["writes"][0][0] is train_df
    assert calls["writes"][1][0] is validation_df
    assert calls["writes"][2][0] is oot_df

    assert preprocessor.fit_transform_called
    assert preprocessor.transform_called

    assert calls["fit_transform_input"] is X_train
    assert calls["transform_input"] is X_validation
    assert calls["artifacts_saved"]


def test_modelling_pipeline_skip(
    monkeypatch: pytest.MonkeyPatch,
    config: dict,
) -> None:
    """Pipeline should perform no work when modelling is skipped."""

    config["parameters"]["modelling"]["skip"] = True

    def fail_if_called(*args, **kwargs):
        pytest.fail("Modelling pipeline should not execute when skip=True.")

    monkeypatch.setattr(
        "credit_risk.pipelines.modelling.load_modelling_vintage",
        fail_if_called,
    )

    monkeypatch.setattr(
        "credit_risk.pipelines.modelling.stratified_data_split",
        fail_if_called,
    )

    monkeypatch.setattr(
        "credit_risk.pipelines.modelling.create_path",
        fail_if_called,
    )

    monkeypatch.setattr(
        "credit_risk.pipelines.modelling.write_parquet",
        fail_if_called,
    )

    monkeypatch.setattr(
        "credit_risk.pipelines.modelling.split_features_target",
        fail_if_called,
    )

    monkeypatch.setattr(
        "credit_risk.pipelines.modelling.build_preprocessor",
        fail_if_called,
    )

    result = run_modelling_pipeline(config)

    assert result is None
