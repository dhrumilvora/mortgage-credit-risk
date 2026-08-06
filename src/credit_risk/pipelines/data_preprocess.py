import pandas as pd

from credit_risk.features.eligibility import validate_baseline_features

from credit_risk.features.origination import (
    select_baseline_features,
    normalize_sentinel_values,
    add_missing_indicators,
)


def build_origination(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the finalized origination preprocessing steps."""

    validate_baseline_features(df.columns)

    result = select_baseline_features(df)
    result = normalize_sentinel_values(result)
    result = add_missing_indicators(result)

    return result
