"""Tests for modelling dataset loading utilities."""

from pathlib import Path

import pandas as pd
import pytest

from credit_risk.modelling.data import load_modelling_vintage


@pytest.fixture
def config() -> dict:
    """Minimal configuration required by the modelling loader."""

    return {
        "catalog": {
            "base": "unused",
        },
        "parameters": {
            "data": {
                "data_provider": "freddie_mac",
            },
        },
    }


def test_load_single_vintage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: dict,
) -> None:
    """A single model-input vintage should load successfully."""

    input_df = pd.DataFrame(
        {
            "loan_id": [1, 2, 3],
            "credit_score": [700, 720, 680],
            "ever_90dpd_24m": [0, 0, 1],
        }
    )

    path = tmp_path / "model_input_2015.parquet"
    input_df.to_parquet(path)

    monkeypatch.setattr(
        "credit_risk.modelling.data.create_path",
        lambda *args, **kwargs: path,
    )

    result = load_modelling_vintage(
        config,
        [2015],
    )

    assert len(result) == 3
    assert result["loan_id"].tolist() == [1, 2, 3]
    assert result["vintage"].tolist() == [2015, 2015, 2015]


def test_load_multiple_vintages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: dict,
) -> None:
    """Multiple vintages should be vertically concatenated."""

    df_2015 = pd.DataFrame(
        {
            "loan_id": [1, 2],
            "ever_90dpd_24m": [0, 1],
        }
    )

    df_2016 = pd.DataFrame(
        {
            "loan_id": [3, 4],
            "ever_90dpd_24m": [0, 0],
        }
    )

    path_2015 = tmp_path / "model_input_2015.parquet"
    path_2016 = tmp_path / "model_input_2016.parquet"

    df_2015.to_parquet(path_2015)
    df_2016.to_parquet(path_2016)

    paths = {
        2015: path_2015,
        2016: path_2016,
    }

    def mock_create_path(
        base_path,
        catalog,
        key,
        data_provider,
        year,
        must_exist=True,
    ):
        return paths[year]

    monkeypatch.setattr(
        "credit_risk.modelling.data.create_path",
        mock_create_path,
    )

    result = load_modelling_vintage(
        config,
        [2015, 2016],
    )

    assert len(result) == 4

    assert result["loan_id"].tolist() == [
        1,
        2,
        3,
        4,
    ]

    assert result["vintage"].tolist() == [
        2015,
        2015,
        2016,
        2016,
    ]


def test_empty_vintage_list_raises(
    config: dict,
) -> None:
    """At least one vintage must be supplied."""

    with pytest.raises(
        ValueError,
        match="At least one modelling vintage",
    ):
        load_modelling_vintage(
            config,
            [],
        )


def test_duplicate_loan_ids_across_vintages_raise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: dict,
) -> None:
    """A loan must not occur in multiple modelling vintages."""

    df_2015 = pd.DataFrame(
        {
            "loan_id": [1, 2],
            "ever_90dpd_24m": [0, 1],
        }
    )

    df_2016 = pd.DataFrame(
        {
            "loan_id": [2, 3],
            "ever_90dpd_24m": [0, 0],
        }
    )

    path_2015 = tmp_path / "model_input_2015.parquet"
    path_2016 = tmp_path / "model_input_2016.parquet"

    df_2015.to_parquet(path_2015)
    df_2016.to_parquet(path_2016)

    paths = {
        2015: path_2015,
        2016: path_2016,
    }

    def mock_create_path(
        base_path,
        catalog,
        key,
        data_provider,
        year,
        must_exist=True,
    ):
        return paths[year]

    monkeypatch.setattr(
        "credit_risk.modelling.data.create_path",
        mock_create_path,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate loan_id",
    ):
        load_modelling_vintage(
            config,
            [2015, 2016],
        )


def test_original_vintage_files_are_not_modified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: dict,
) -> None:
    """Vintage metadata should be added only to the returned dataset."""

    input_df = pd.DataFrame(
        {
            "loan_id": [1, 2],
            "ever_90dpd_24m": [0, 1],
        }
    )

    path = tmp_path / "model_input_2015.parquet"
    input_df.to_parquet(path)

    monkeypatch.setattr(
        "credit_risk.modelling.data.create_path",
        lambda *args, **kwargs: path,
    )

    result = load_modelling_vintage(
        config,
        [2015],
    )

    persisted = pd.read_parquet(path)

    assert "vintage" in result.columns
    assert "vintage" not in persisted.columns
