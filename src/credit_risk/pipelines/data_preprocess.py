import pandas as pd
from pathlib import Path
from credit_risk.features.eligibility_origination import (
    validate_baseline_features,
)
from credit_risk.features.eligibility_performance import (
    validate_features,
)
import credit_risk.features.origination as origination
import credit_risk.features.performance as performance

from credit_risk.target.delinquency import (
    build_24m_serious_delinquency_target,
)


def build_origination(df: pd.DataFrame) -> pd.DataFrame:
    """Apply finalized origination preprocessing."""

    validate_baseline_features(df.columns)

    result = origination.select_baseline_features(df)
    result = origination.normalize_sentinel_values(result)
    result = origination.add_missing_indicators(result)

    return result


def build_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Select fields required for baseline performance processing."""

    validate_features(df.columns)

    return performance.select_baseline_features(df)


def build_master_dataset(
    origination_df: pd.DataFrame,
    performance_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the master loan-month dataset."""

    orig = build_origination(origination_df)
    perf = build_performance(performance_df)

    return perf.merge(
        orig,
        on="loan_id",
        how="inner",
        validate="many_to_one",
    )


def build_modeling_dataset(origination: Path, performance: Path) -> pd.DataFrame:
    """Build the final loan-level modelling dataset."""
    origination_df = pd.read_parquet(origination)
    performance_df = pd.read_parquet(performance)

    master = build_master_dataset(
        origination_df,
        performance_df,
    )

    target = build_24m_serious_delinquency_target(master)

    # Origination attributes are constant across loan-months,
    # so retain one record per eligible loan.
    origination_features = master.drop_duplicates(subset="loan_id")

    modeling = origination_features.merge(
        target,
        on="loan_id",
        how="inner",
        validate="one_to_one",
    )

    return modeling.reset_index(drop=True)
