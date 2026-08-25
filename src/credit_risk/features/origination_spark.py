"""PySpark origination feature construction for mortgage credit-risk modelling."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from credit_risk.features.origination import (
    CATEGORICAL_SENTINELS,
    NUMERIC_SENTINELS,
)
from credit_risk.features.eligibility_origination import (
    validate_baseline_features,
)


def _validate_required_columns(
    df: DataFrame,
    config: dict,
) -> None:
    """
    Validate the columns required by the existing origination contract.

    The validation logic mirrors the existing Pandas pipeline.
    """

    validate_baseline_features(
        df.columns,
        config,
    )


def select_baseline_features_spark(
    df: DataFrame,
    config: dict,
) -> DataFrame:
    """
    Select the exact origination columns configured for preprocessing.

    Equivalent to:

        origination.select_baseline_features(...)
    """

    parameters = config["parameters"]

    id_column = parameters["data"]["id_col"]

    numerical_features = parameters["data"]["preprocess"]["features"][
        "numerical_features"
    ]

    categorical_features = parameters["data"]["preprocess"]["features"][
        "categorical_features"
    ]

    columns = [
        column
        for column in ([id_column] + numerical_features + categorical_features)
        if column is not None
    ]

    missing_columns = sorted(
        set(columns) - set(df.columns),
    )

    if missing_columns:
        raise ValueError(
            "Missing required origination fields: " + ", ".join(missing_columns)
        )

    return df.select(
        *columns,
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

        result = result.withColumn(
            column,
            F.when(
                F.col(column) == F.lit(sentinel),
                F.lit(None),
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


def add_missing_indicators_spark(
    df: DataFrame,
) -> DataFrame:
    """
    Add missingness indicators for informative missing fields.

    Equivalent to:

        result["original_dti_missing"] =
            result["original_dti"].isna().astype("int8")
    """

    result = df

    if "original_dti" in result.columns:

        result = result.withColumn(
            "original_dti_missing",
            F.when(
                F.col("original_dti").isNull(),
                F.lit(1),
            )
            .otherwise(
                F.lit(0),
            )
            .cast("byte"),
        )

    return result


def build_origination_spark(
    df: DataFrame,
    config: dict,
) -> DataFrame:
    """
    Apply the complete baseline origination preprocessing contract.

    Order is intentionally identical to the existing Pandas pipeline:

        1. validate required fields
        2. select baseline fields
        3. normalize sentinel values
        4. add missingness indicators
    """

    _validate_required_columns(
        df,
        config,
    )

    result = select_baseline_features_spark(
        df,
        config,
    )

    result = normalize_sentinel_values_spark(
        result,
    )

    result = add_missing_indicators_spark(
        result,
    )

    return result
