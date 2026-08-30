"""Performance feature selection for canonical mortgage data."""

from __future__ import annotations

import pandas as pd

from credit_risk.features.eligibility_performance import (
    BASELINE_FEATURES,
    CHALLENGER_FEATURES,
    IDENTIFIER_FIELDS,
    STATE_FIELDS,
    TERMINATION_FIELDS,
    TIME_FIELDS,
)


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
