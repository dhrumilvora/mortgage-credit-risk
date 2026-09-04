from pyspark.sql import DataFrame, functions as F

from credit_risk.features.eligibility_performance import (
    BASELINE_FEATURES,
    IDENTIFIER_FIELDS,
    TIME_FIELDS,
    STATE_FIELDS,
    TERMINATION_FIELDS,
    CHALLENGER_FEATURES,
)
from credit_risk.features.performance import (
    CATEGORICAL_SENTINELS,
    NUMERIC_SENTINELS,
)


def normalize_sentinel_values_spark(
    df: DataFrame,
) -> DataFrame:
    """
    Convert documented sentinel values to Spark nulls.

    Equivalent to the existing Pandas implementation.
    """

    result = df

    for column, sentinel in NUMERIC_SENTINELS.items():
    
        if column not in result.columns:
            continue
    
        column_type = result.schema[column].dataType
    
        result = result.withColumn(
            column,
            F.when(
                F.col(column) == F.lit(sentinel).cast(column_type),
                F.lit(None).cast(column_type),
            ).otherwise(
                F.col(column),
            ),
        )

    for column, sentinel in CATEGORICAL_SENTINELS.items():

        if column not in result.columns:
            continue

        result = result.withColumn(
            column,
            F.when(
                F.col(column) == F.lit(sentinel),
                F.lit(None),
            ).otherwise(
                F.col(column),
            ),
        )

    return result


def select_baseline_features_spark(
    df: DataFrame,
) -> DataFrame:
    """
    Select the exact same performance columns as the Pandas
    select_baseline_features() implementation.
    """

    columns = (
        IDENTIFIER_FIELDS
        + TIME_FIELDS
        + BASELINE_FEATURES
        + STATE_FIELDS
        + CHALLENGER_FEATURES
        + TERMINATION_FIELDS
    )

    missing_columns = sorted(
        set(columns) - set(df.columns),
    )

    if missing_columns:
        raise ValueError(
            "Missing required performance fields: " + ", ".join(missing_columns)
        )

    return df.select(
        *columns,
    )


def build_performance_spark(
    df: DataFrame,
) -> DataFrame:
    """
    Apply finalized performance preprocessing using Spark.

    The performance preprocessing contract consists of:

        1. select configured baseline columns
        2. normalize documented sentinel values
    """

    result = select_baseline_features_spark(
        df,
    )

    result = normalize_sentinel_values_spark(
        result,
    )

    return result
