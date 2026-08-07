"""Tests for modelling pipeline orchestration."""

from pathlib import Path

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
            "modelling": {
                "skip": False,
                "vintages_train": [2015],
            },
        },
    }


@pytest.fixture
def development_df() -> pd.DataFrame:
    """Synthetic development population."""

    return pd.DataFrame(
        {
            "loan_id": [1, 2, 3, 4],
            "vintage": [2015, 2015, 2015, 2015],
            "ever_90dpd_24m": [0, 0, 0, 1],
        }
    )


def test_modelling_pipeline_orchestration(
    monkeypatch: pytest.MonkeyPatch,
    config: dict,
    development_df: pd.DataFrame,
) -> None:
    """Pipeline should load, split, and persist development data."""

    train_df = development_df.iloc[:3].copy()
    validation_df = development_df.iloc[3:].copy()

    calls = {
        "writes": [],
        "path_keys": [],
    }

    def mock_load(config_arg, vintages):
        calls["load_config"] = config_arg
        calls["vintages"] = vintages

        return development_df

    def mock_split(df, config_arg):
        calls["split_df"] = df
        calls["split_config"] = config_arg

        return train_df, validation_df

    paths = {
        "train_df": Path("train.parquet"),
        "validation_df": Path("validation.parquet"),
    }

    def mock_create_path(
        base_path,
        catalog,
        key,
        must_exist=True,
    ):
        calls["path_keys"].append(key)

        return paths[key]

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
        "credit_risk.pipelines.modelling.create_path",
        mock_create_path,
    )

    monkeypatch.setattr(
        "credit_risk.pipelines.modelling.write_parquet",
        mock_write_parquet,
    )

    result = run_modelling_pipeline(config)

    # Pipeline is an orchestrator and should not return datasets.
    assert result is None

    # Correct vintages should be loaded.
    assert calls["vintages"] == [2015]
    assert calls["load_config"] is config

    # Loaded development population should be passed directly
    # to the splitting stage.
    assert calls["split_df"] is development_df
    assert calls["split_config"] is config

    # Correct output paths should be resolved.
    assert calls["path_keys"] == [
        "train_df",
        "validation_df",
    ]

    # Both populations should be persisted.
    assert len(calls["writes"]) == 2

    written_train_df, written_train_path = calls["writes"][0]

    assert written_train_df is train_df
    assert written_train_path == Path("train.parquet")

    written_validation_df, written_validation_path = calls["writes"][1]

    assert written_validation_df is validation_df
    assert written_validation_path == Path("validation.parquet")


def test_modelling_pipeline_skip(
    monkeypatch: pytest.MonkeyPatch,
    config: dict,
) -> None:
    """Pipeline should perform no work when modelling is skipped."""

    config["parameters"]["modelling"]["skip"] = True

    def fail_if_called(*args, **kwargs):
        pytest.fail("Modelling stage should not run when modelling is skipped.")

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

    result = run_modelling_pipeline(config)

    assert result is None
