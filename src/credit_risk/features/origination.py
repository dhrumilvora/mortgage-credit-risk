import pandas as pd
from credit_risk.features.eligibility_origination import (
    validate_baseline_features,
)

NUMERIC_SENTINELS = {
    "credit_score": 9999,
    "original_dti": 999,
    "original_ltv": 999,
    "original_cltv": 999,
    "number_of_borrowers": 99,
    "number_of_units": 99,
    "mi_percentage": 999,
    "original_loan_term": 999,
}

CATEGORICAL_SENTINELS = {
    "first_time_homebuyer_flag": "9",
    "property_type": "99",
    "occupancy_status": "9",
    "channel": "9",
    "loan_purpose": "9",
}


def select_baseline_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Select identifiers and baseline origination features."""
    parameters = config["parameters"]
    columns = filter(
        None,
        [parameters["data"]["id_col"]]
        + parameters["data"]["preprocess"]["features"]["numerical_features"]
        + parameters["data"]["preprocess"]["features"]["categorical_features"],
    )

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


def build_origination(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Apply finalized origination preprocessing."""

    validate_baseline_features(
        df.columns,
        config,
    )

    result = select_baseline_features(
        df,
        config,
    )

    result = normalize_sentinel_values(
        result,
    )

    result = add_missing_indicators(
        result,
    )

    return result
