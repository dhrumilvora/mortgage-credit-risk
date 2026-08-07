"""Tests for development dataset splitting utilities."""

import pandas as pd
import pytest

from credit_risk.modelling.split import stratified_data_split


@pytest.fixture
def config() -> dict:
    """Minimal configuration required by the development split."""

    return {
        "parameters": {
            "target": {
                "name": "ever_90dpd_24m",
            },
            "modelling": {
                "validation_size": 0.20,
                "random_state": 42,
                "stratify": True,
            },
        },
    }


@pytest.fixture
def modelling_df() -> pd.DataFrame:
    """Create a deterministic loan-level modelling population."""

    return pd.DataFrame(
        {
            "loan_id": range(1, 101),
            "credit_score": range(600, 700),
            "ever_90dpd_24m": [0] * 80 + [1] * 20,
        }
    )


def test_split_sizes(
    modelling_df: pd.DataFrame,
    config: dict,
) -> None:
    """Train and validation populations should have expected sizes."""

    train_df, validation_df = stratified_data_split(
        modelling_df,
        config,
    )

    assert len(train_df) == 80
    assert len(validation_df) == 20
    assert len(train_df) + len(validation_df) == len(modelling_df)


def test_stratification_preserves_event_rate(
    modelling_df: pd.DataFrame,
    config: dict,
) -> None:
    """Stratified split should preserve the target event rate."""

    target = config["parameters"]["target"]["name"]

    train_df, validation_df = stratified_data_split(
        modelling_df,
        config,
    )

    original_rate = modelling_df[target].mean()
    train_rate = train_df[target].mean()
    validation_rate = validation_df[target].mean()

    assert train_rate == pytest.approx(original_rate)
    assert validation_rate == pytest.approx(original_rate)


def test_train_validation_do_not_overlap(
    modelling_df: pd.DataFrame,
    config: dict,
) -> None:
    """A loan must belong to only one development population."""

    train_df, validation_df = stratified_data_split(
        modelling_df,
        config,
    )

    train_loans = set(train_df["loan_id"])
    validation_loans = set(validation_df["loan_id"])

    assert train_loans.isdisjoint(validation_loans)


def test_split_preserves_all_loans(
    modelling_df: pd.DataFrame,
    config: dict,
) -> None:
    """Every input loan should appear exactly once after splitting."""

    train_df, validation_df = stratified_data_split(
        modelling_df,
        config,
    )

    split_loans = set(train_df["loan_id"]) | set(validation_df["loan_id"])

    assert split_loans == set(modelling_df["loan_id"])


def test_split_is_reproducible(
    modelling_df: pd.DataFrame,
    config: dict,
) -> None:
    """The same random state should produce the same split."""

    train_1, validation_1 = stratified_data_split(
        modelling_df,
        config,
    )

    train_2, validation_2 = stratified_data_split(
        modelling_df,
        config,
    )

    assert train_1["loan_id"].tolist() == train_2["loan_id"].tolist()
    assert validation_1["loan_id"].tolist() == validation_2["loan_id"].tolist()


def test_non_stratified_split_runs(
    modelling_df: pd.DataFrame,
    config: dict,
) -> None:
    """The split should support disabling stratification."""

    config["parameters"]["modelling"]["stratify"] = False

    train_df, validation_df = stratified_data_split(
        modelling_df,
        config,
    )

    assert len(train_df) == 80
    assert len(validation_df) == 20


def test_missing_target_column_raises(
    modelling_df: pd.DataFrame,
    config: dict,
) -> None:
    """Missing target columns should fail immediately."""

    modelling_df = modelling_df.drop(columns="ever_90dpd_24m")

    with pytest.raises(
        ValueError,
        match="Target column not found",
    ):
        stratified_data_split(
            modelling_df,
            config,
        )


def test_missing_target_values_raise(
    modelling_df: pd.DataFrame,
    config: dict,
) -> None:
    """Missing target values should not be allowed."""

    modelling_df.loc[0, "ever_90dpd_24m"] = None

    with pytest.raises(
        ValueError,
        match="Target column contains missing values",
    ):
        stratified_data_split(
            modelling_df,
            config,
        )


@pytest.mark.parametrize(
    "validation_size",
    [
        0,
        1,
        -0.10,
        1.10,
    ],
)
def test_invalid_validation_size_raises(
    modelling_df: pd.DataFrame,
    config: dict,
    validation_size: float,
) -> None:
    """Validation size must be strictly between zero and one."""

    config["parameters"]["modelling"]["validation_size"] = validation_size

    with pytest.raises(
        ValueError,
        match="validation_size must be between 0 and 1",
    ):
        stratified_data_split(
            modelling_df,
            config,
        )


def test_single_class_target_raises(
    modelling_df: pd.DataFrame,
    config: dict,
) -> None:
    """Development data must contain both target classes."""

    modelling_df["ever_90dpd_24m"] = 0

    with pytest.raises(
        ValueError,
        match="Target must contain at least two classes",
    ):
        stratified_data_split(
            modelling_df,
            config,
        )
