"""Performance feature selection for canonical mortgage data."""

from __future__ import annotations

import pandas as pd

NUMERIC_SENTINELS = {
    "estimated_ltv": 999,
}


CATEGORICAL_SENTINELS = {
    "current_loan_delinquency_status": "XX",
    # "net_sales_proceeds": "U",
}
from credit_risk.features.eligibility_performance import (
    BASELINE_FEATURES,
    CHALLENGER_FEATURES,
    IDENTIFIER_FIELDS,
    STATE_FIELDS,
    TERMINATION_FIELDS,
    TIME_FIELDS,
    validate_features,
)


def normalize_sentinel_values(df: pd.DataFrame) -> pd.DataFrame:
    """Convert documented sentinel values to missing values."""

    result = df.copy()

    for column, sentinel in NUMERIC_SENTINELS.items():
        if column in result.columns:
            result[column] = result[column].replace(sentinel, pd.NA)

    for column, sentinel in CATEGORICAL_SENTINELS.items():
        if column in result.columns:
            result[column] = result[column].replace(sentinel, pd.NA)

    return result


def select_baseline_features(df: pd.DataFrame) -> pd.DataFrame:

    columns = (
        IDENTIFIER_FIELDS
        + TIME_FIELDS
        + BASELINE_FEATURES
        + STATE_FIELDS
        + TERMINATION_FIELDS
        + CHALLENGER_FEATURES
    )

    return df.loc[:, columns].copy()


def build_performance(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Apply finalized performance preprocessing."""

    validate_features(
        df.columns,
    )
    result = select_baseline_features(
        df,
    )
    result = normalize_sentinel_values(result)
    return result
