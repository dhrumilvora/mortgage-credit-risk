"""Tests for performance feature selection."""

import pandas as pd
import pytest

from credit_risk.features.eligibility_performance import (
    BASELINE_FEATURES,
    IDENTIFIER_FIELDS,
    STATE_FIELDS,
    TERMINATION_FIELDS,
    TIME_FIELDS,
)
from credit_risk.features.performance import select_baseline_features

EXPECTED_COLUMNS = (
    IDENTIFIER_FIELDS
    + TIME_FIELDS
    + BASELINE_FEATURES
    + STATE_FIELDS
    + TERMINATION_FIELDS
)


@pytest.fixture
def performance_df() -> pd.DataFrame:
    """Create a minimal performance dataset containing all required fields."""

    data = {column: [1, 2] for column in EXPECTED_COLUMNS}

    data["unused_feature"] = [100, 200]

    return pd.DataFrame(data)


def test_select_baseline_features_selects_expected_columns(
    performance_df: pd.DataFrame,
) -> None:
    """Selected dataset should contain exactly the expected fields."""

    result = select_baseline_features(performance_df)

    assert result.columns.tolist() == EXPECTED_COLUMNS


def test_select_baseline_features_drops_unselected_columns(
    performance_df: pd.DataFrame,
) -> None:
    """Fields outside the baseline specification should be removed."""

    result = select_baseline_features(performance_df)

    assert "unused_feature" not in result.columns


def test_select_baseline_features_preserves_rows(
    performance_df: pd.DataFrame,
) -> None:
    """Feature selection should not change the number of observations."""

    result = select_baseline_features(performance_df)

    assert len(result) == len(performance_df)


def test_select_baseline_features_returns_copy(
    performance_df: pd.DataFrame,
) -> None:
    """Changes to the result should not modify the source DataFrame."""

    result = select_baseline_features(performance_df)

    column = EXPECTED_COLUMNS[0]

    result.loc[0, column] = 999

    assert performance_df.loc[0, column] != 999


def test_select_baseline_features_raises_for_missing_column(
    performance_df: pd.DataFrame,
) -> None:
    """Missing required fields should prevent feature selection."""

    missing_column = EXPECTED_COLUMNS[-1]

    performance_df = performance_df.drop(columns=missing_column)

    with pytest.raises(KeyError):
        select_baseline_features(performance_df)
