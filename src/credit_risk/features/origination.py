import pandas as pd

from credit_risk.features.eligibility import (
    BASELINE_FEATURES,
    IDENTIFIER_FEATURE,
)

NUMERIC_SENTINELS = {
    "original_dti": 999,
    "original_ltv": 999,
    "original_cltv": 999,
}


CATEGORICAL_SENTINELS = {
    "first_time_homebuyer_flag": "9",
}


def select_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select identifiers and baseline origination features."""

    columns = IDENTIFIER_FEATURE + BASELINE_FEATURES

    return df[columns].copy()


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


def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add missingness indicators for informative missing fields."""

    result = df.copy()

    if "original_dti" in result.columns:
        result["original_dti_missing"] = result["original_dti"].isna().astype("int8")

    return result
