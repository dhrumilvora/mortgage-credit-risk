"""Optimized PySpark point-in-time behavioral feature construction."""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------


def _validate_required_columns(
    df: DataFrame,
    required_columns: set[str],
    context: str,
) -> None:
    """Validate that required columns exist."""

    missing_columns = sorted(required_columns - set(df.columns))

    if missing_columns:
        raise ValueError(
            f"Missing columns required for {context}: " + ", ".join(missing_columns)
        )


# ----------------------------------------------------------------------
# Serious delinquency
# ----------------------------------------------------------------------


def add_prior_serious_delinquency_flag_spark(
    df: DataFrame,
    config: dict,
) -> DataFrame:
    """
    Add row-level serious-delinquency flag.

    Serious delinquency:

        numeric DPD >= configured threshold
        OR current status == "RA"
    """

    _validate_required_columns(
        df,
        {
            "current_loan_delinquency_status",
        },
        "serious delinquency flag",
    )

    threshold = config["parameters"]["target"]["serious_delinquency_threshold"]

    delinquency_numeric = F.expr("try_cast(current_loan_delinquency_status as double)")

    serious_delinquency = (delinquency_numeric >= F.lit(threshold)) | (
        F.col("current_loan_delinquency_status") == F.lit("RA")
    )

    return df.withColumn(
        "is_serious_delinquency",
        F.coalesce(
            serious_delinquency,
            F.lit(False),
        ),
    )


# ----------------------------------------------------------------------
# Standalone behavioral risk-set function
# ----------------------------------------------------------------------


def build_behavioral_risk_set_spark(
    master: DataFrame,
    observation_age: int,
    config: dict,
) -> DataFrame:
    """
    Build the eligible point-in-time population for one observation age.

    This function is retained for standalone use.

    The main production path uses the vectorized
    build_behavioral_features_spark() implementation instead of
    repeatedly calling this function.
    """

    if observation_age < 0:
        raise ValueError(
            "observation_age must be non-negative, " f"got {observation_age}."
        )

    _validate_required_columns(
        master,
        {
            "loan_id",
            "calculated_loan_age",
            "current_loan_delinquency_status",
            "zero_balance_code",
        },
        "behavioral risk set",
    )

    threshold = config["parameters"]["target"]["serious_delinquency_threshold"]

    delinquency_numeric = F.expr("try_cast(current_loan_delinquency_status as double)")

    serious_flag = (delinquency_numeric >= F.lit(threshold)) | (
        F.col("current_loan_delinquency_status") == F.lit("RA")
    )

    # One cumulative window for the standalone calculation.
    prior_window = (
        Window.partitionBy("loan_id")
        .orderBy(
            F.col("calculated_loan_age").asc(),
        )
        .rangeBetween(
            Window.unboundedPreceding,
            Window.currentRow,
        )
    )

    working = master.withColumn(
        "_serious_flag",
        F.coalesce(
            serious_flag,
            F.lit(False),
        ),
    ).withColumn(
        "_has_prior_serious_delinquency",
        F.coalesce(
            F.max(F.col("_serious_flag").cast("int")).over(
                prior_window,
            ),
            F.lit(0),
        ).cast("boolean"),
    )

    eligible = (
        working.filter(F.col("calculated_loan_age") == F.lit(observation_age))
        .filter(~F.col("_has_prior_serious_delinquency"))
        .filter(F.col("zero_balance_code").isNull())
        .withColumn(
            "observation_age",
            F.lit(observation_age).cast("int"),
        )
        .drop(
            "_serious_flag",
            "_has_prior_serious_delinquency",
        )
    )

    return eligible


# ----------------------------------------------------------------------
# Optimized complete behavioral feature population
# ----------------------------------------------------------------------


def build_behavioral_features_spark(
    master: DataFrame,
    config: dict,
) -> DataFrame:
    """
    Build the complete point-in-time behavioral feature population.

    Output grain:
        loan_id x observation_age

    Adds current-state, lifetime-history, and recent-window features for:
        - delinquency
        - modification
        - payment deferral
        - borrower assistance
        - disaster delinquency
        - interest-rate steps
        - UPB composition
        - DDLPI recency
        - UPB/rate trajectory

    All history is calculated through and including the observation row.
    """

    behavioral_config = config["parameters"]["behavioral"]

    observation_ages = behavioral_config.get("observation_ages", [])
    lookback_windows_months = behavioral_config.get("lookback_windows_months", [])

    if not observation_ages:
        raise ValueError("parameters.behavioral.observation_ages cannot be empty.")

    observation_ages = [int(age) for age in observation_ages]

    invalid_ages = [age for age in observation_ages if age < 0]
    if invalid_ages:
        raise ValueError(
            "parameters.behavioral.observation_ages contains "
            "negative values: " + ", ".join(str(age) for age in invalid_ages)
        )

    observation_ages = list(dict.fromkeys(observation_ages))
    max_observation_age = max(observation_ages)

    lookback_windows_months = [int(window) for window in lookback_windows_months]

    invalid_windows = [window for window in lookback_windows_months if window <= 0]
    if invalid_windows:
        raise ValueError(
            "parameters.behavioral.lookback_windows_months must "
            "contain only positive integers: "
            + ", ".join(str(window) for window in invalid_windows)
        )

    lookback_windows_months = list(dict.fromkeys(lookback_windows_months))

    # ------------------------------------------------------------------
    # Validate input contract once.
    # ------------------------------------------------------------------

    _validate_required_columns(
        master,
        {
            "loan_id",
            "period",
            "calculated_loan_age",
            "current_loan_delinquency_status",
            "zero_balance_code",
            "original_upb",
            "current_actual_upb",
            "original_interest_rate",
            "current_interest_rate",
            "current_non_interest_bearing_upb",
            "current_interest_bearing_upb",
            "ddlpi",
            "modification_flag",
            "payment_deferral_flag",
            "borrower_assistance_plan",
            "delinquency_due_to_disaster",
            "interest_rate_step_indicator",
        },
        "behavioral feature construction",
    )

    # Only history through the latest required observation age is needed.
    working = master.filter(F.col("calculated_loan_age") <= F.lit(max_observation_age))

    # ------------------------------------------------------------------
    # Current-row base features.
    # ------------------------------------------------------------------

    current_dpd_numeric = F.expr("try_cast(current_loan_delinquency_status as double)")

    serious_threshold = config["parameters"]["target"]["serious_delinquency_threshold"]

    serious_flag = (current_dpd_numeric >= F.lit(serious_threshold)) | (
        F.col("current_loan_delinquency_status") == F.lit("RA")
    )

    # Freddie binary indicators: Y = active.
    modification_flag = (
        F.when(
            F.upper(F.trim(F.col("modification_flag"))) == "Y",
            1,
        )
        .otherwise(0)
        .cast("byte")
    )

    payment_deferral_flag = (
        F.when(
            F.upper(F.trim(F.col("payment_deferral_flag"))) == "Y",
            1,
        )
        .otherwise(0)
        .cast("byte")
    )

    borrower_assistance_flag = (
        F.when(
            F.upper(F.trim(F.col("borrower_assistance_plan"))) == "Y",
            1,
        )
        .otherwise(0)
        .cast("byte")
    )

    disaster_delinquency_flag = (
        F.when(
            F.upper(F.trim(F.col("delinquency_due_to_disaster"))) == "Y",
            1,
        )
        .otherwise(0)
        .cast("byte")
    )

    rate_step_flag = (
        F.when(
            F.upper(F.trim(F.col("interest_rate_step_indicator"))) == "Y",
            1,
        )
        .otherwise(0)
        .cast("byte")
    )

    # ------------------------------------------------------------------
    # UPB composition.
    # ------------------------------------------------------------------

    non_interest_bearing_upb_pct = F.when(
        F.col("current_actual_upb").isNull() | (F.col("current_actual_upb") == 0),
        F.lit(None).cast("double"),
    ).otherwise(F.col("current_non_interest_bearing_upb") / F.col("current_actual_upb"))

    interest_bearing_upb_pct = F.when(
        F.col("current_actual_upb").isNull() | (F.col("current_actual_upb") == 0),
        F.lit(None).cast("double"),
    ).otherwise(F.col("current_interest_bearing_upb") / F.col("current_actual_upb"))

    # Monthly Period[M] values are represented by Spark as ordinals.
    months_since_ddlpi = F.when(
        F.col("ddlpi").isNull(),
        F.lit(None).cast("double"),
    ).otherwise((F.col("period") - F.col("ddlpi")).cast("double"))

    upb_change = F.col("current_actual_upb") - F.col("original_upb")

    working = working.select(
        "*",
        current_dpd_numeric.alias("current_dpd_numeric"),
        F.when(current_dpd_numeric >= 30, 1)
        .otherwise(0)
        .cast("byte")
        .alias("current_dpd_30_plus"),
        F.when(current_dpd_numeric >= 60, 1)
        .otherwise(0)
        .cast("byte")
        .alias("current_dpd_60_plus"),
        F.when(current_dpd_numeric > 0, 1)
        .otherwise(0)
        .cast("byte")
        .alias("current_delinquency_flag"),
        F.coalesce(
            serious_flag,
            F.lit(False),
        ).alias("is_serious_delinquency"),
        modification_flag.alias("current_modification_flag"),
        payment_deferral_flag.alias("current_payment_deferral_flag"),
        borrower_assistance_flag.alias("current_borrower_assistance_flag"),
        disaster_delinquency_flag.alias("current_disaster_delinquency_flag"),
        rate_step_flag.alias("current_rate_step_flag"),
        upb_change.alias("upb_change_from_origination"),
        F.when(
            F.col("original_upb").isNull() | (F.col("original_upb") == 0),
            F.lit(None).cast("double"),
        )
        .otherwise(upb_change / F.col("original_upb"))
        .alias("upb_pct_change_from_origination"),
        (F.col("current_interest_rate") - F.col("original_interest_rate")).alias(
            "rate_change_from_origination"
        ),
        non_interest_bearing_upb_pct.alias("non_interest_bearing_upb_pct"),
        interest_bearing_upb_pct.alias("interest_bearing_upb_pct"),
        months_since_ddlpi.alias("months_since_ddlpi"),
    )

    # ------------------------------------------------------------------
    # Recent behavioral windows.
    # ------------------------------------------------------------------

    recent_feature_expressions = []

    for window_months in lookback_windows_months:

        recent_window = (
            Window.partitionBy("loan_id")
            .orderBy(F.col("calculated_loan_age").asc())
            .rangeBetween(
                -(window_months - 1),
                Window.currentRow,
            )
        )

        recent_feature_expressions.extend(
            [
                F.sum(F.col("current_dpd_30_plus"))
                .over(recent_window)
                .cast("short")
                .alias(f"dpd_30_count_{window_months}m"),
                F.sum(F.col("current_dpd_60_plus"))
                .over(recent_window)
                .cast("short")
                .alias(f"dpd_60_count_{window_months}m"),
                F.max(F.col("current_dpd_numeric"))
                .over(recent_window)
                .alias(f"max_dpd_{window_months}m"),
                F.sum(F.col("current_delinquency_flag"))
                .over(recent_window)
                .cast("short")
                .alias(f"delinquency_months_{window_months}m"),
                F.sum(F.col("current_modification_flag"))
                .over(recent_window)
                .cast("short")
                .alias(f"modification_count_{window_months}m"),
                F.sum(F.col("current_payment_deferral_flag"))
                .over(recent_window)
                .cast("short")
                .alias(f"payment_deferral_count_{window_months}m"),
                F.sum(F.col("current_borrower_assistance_flag"))
                .over(recent_window)
                .cast("short")
                .alias(f"borrower_assistance_count_{window_months}m"),
                F.sum(F.col("current_disaster_delinquency_flag"))
                .over(recent_window)
                .cast("short")
                .alias(f"disaster_delinquency_count_{window_months}m"),
                F.sum(F.col("current_rate_step_flag"))
                .over(recent_window)
                .cast("short")
                .alias(f"rate_step_count_{window_months}m"),
            ]
        )

    if recent_feature_expressions:
        working = working.select(
            "*",
            *recent_feature_expressions,
        )

    # ------------------------------------------------------------------
    # Shared lifetime history window.
    # ------------------------------------------------------------------

    history_window = (
        Window.partitionBy("loan_id")
        .orderBy(F.col("calculated_loan_age").asc())
        .rowsBetween(
            Window.unboundedPreceding,
            Window.currentRow,
        )
    )

    working = working.select(
        "*",
        F.max(F.col("current_dpd_numeric"))
        .over(history_window)
        .alias("max_dpd_to_date"),
        F.max(F.col("current_dpd_30_plus"))
        .over(history_window)
        .cast("byte")
        .alias("ever_30dpd_to_date"),
        F.max(F.col("current_dpd_60_plus"))
        .over(history_window)
        .cast("byte")
        .alias("ever_60dpd_to_date"),
        F.sum(F.col("current_delinquency_flag"))
        .over(history_window)
        .cast("short")
        .alias("delinquency_months_to_date"),
        F.max(F.col("current_modification_flag"))
        .over(history_window)
        .cast("byte")
        .alias("ever_modified"),
        F.max(F.col("current_payment_deferral_flag"))
        .over(history_window)
        .cast("byte")
        .alias("ever_payment_deferred"),
        F.max(F.col("current_borrower_assistance_flag"))
        .over(history_window)
        .cast("byte")
        .alias("ever_borrower_assistance"),
        F.max(F.col("current_disaster_delinquency_flag"))
        .over(history_window)
        .cast("byte")
        .alias("ever_disaster_delinquency"),
        F.max(
            F.when(
                F.col("current_delinquency_flag") == 1,
                F.col("calculated_loan_age"),
            )
        )
        .over(history_window)
        .alias("_last_delinquency_age"),
        F.max(F.col("is_serious_delinquency").cast("int"))
        .over(history_window)
        .cast("boolean")
        .alias("_has_prior_serious_delinquency"),
    )

    working = working.withColumn(
        "months_since_last_delinquency",
        F.when(
            F.col("_last_delinquency_age").isNull(),
            F.lit(None).cast("double"),
        ).otherwise(F.col("calculated_loan_age") - F.col("_last_delinquency_age")),
    )

    # ------------------------------------------------------------------
    # Final point-in-time population.
    # ------------------------------------------------------------------

    behavioral_features = (
        working.filter(F.col("calculated_loan_age").isin(observation_ages))
        .filter(
            ~F.coalesce(
                F.col("_has_prior_serious_delinquency"),
                F.lit(False),
            )
        )
        .filter(F.col("zero_balance_code").isNull())
        .withColumn(
            "observation_age",
            F.col("calculated_loan_age").cast("int"),
        )
        .drop(
            "is_serious_delinquency",
            "_has_prior_serious_delinquency",
            "_last_delinquency_age",
            "current_modification_flag",
            "current_payment_deferral_flag",
            "current_borrower_assistance_flag",
            "current_disaster_delinquency_flag",
            "current_rate_step_flag",
        )
    )

    logger.info(
        "Spark behavioral feature plan configured: "
        "observation_ages=%s "
        "lookback_windows_months=%s "
        "max_observation_age=%s "
        "recent_feature_count=%s",
        observation_ages,
        lookback_windows_months,
        max_observation_age,
        len(recent_feature_expressions),
    )

    return behavioral_features


# ----------------------------------------------------------------------
# Lifecycle clock
# ----------------------------------------------------------------------


def add_calculated_loan_age_spark(
    df: DataFrame,
) -> DataFrame:
    """
    Calculate loan age from the canonical monthly Period[M] ordinal.

    Pandas reads the Parquet fields as Period[M], e.g.:

        2015-05

    Spark reads the underlying monthly ordinal, e.g.:

        544

    The Pandas calculation:

        months(period - first_payment_date) + 1

    is therefore equivalent to:

        period - first_payment_date + 1
    """

    _validate_required_columns(
        df,
        {
            "period",
            "first_payment_date",
        },
        "calculated loan age",
    )

    return df.withColumn(
        "calculated_loan_age",
        (F.col("period") - F.col("first_payment_date") + F.lit(1)).cast("int"),
    )


# ----------------------------------------------------------------------
# Historical feature helper
# ----------------------------------------------------------------------


def add_behavioral_history_features_spark(
    df: DataFrame,
) -> DataFrame:
    """
    Add leakage-safe historical behavioral features.

    This standalone function remains available for callers that need
    only the historical feature layer.

    The complete production behavioral pipeline uses the fused
    implementation in build_behavioral_features_spark() to avoid
    calculating the same windows twice.
    """

    _validate_required_columns(
        df,
        {
            "loan_id",
            "calculated_loan_age",
            "current_loan_delinquency_status",
        },
        "behavioral history features",
    )

    current_dpd_numeric = F.expr("try_cast(current_loan_delinquency_status as double)")

    result = df.select(
        "*",
        current_dpd_numeric.alias("current_dpd_numeric"),
        F.when(
            current_dpd_numeric >= 30,
            F.lit(1),
        )
        .otherwise(
            F.lit(0),
        )
        .cast("byte")
        .alias("current_dpd_30_plus"),
        F.when(
            current_dpd_numeric >= 60,
            F.lit(1),
        )
        .otherwise(
            F.lit(0),
        )
        .cast("byte")
        .alias("current_dpd_60_plus"),
        F.when(
            current_dpd_numeric > 0,
            F.lit(1),
        )
        .otherwise(
            F.lit(0),
        )
        .cast("byte")
        .alias("current_delinquency_flag"),
    )

    history_window = (
        Window.partitionBy("loan_id")
        .orderBy(F.col("calculated_loan_age").asc())
        .rowsBetween(
            Window.unboundedPreceding,
            Window.currentRow,
        )
    )

    result = result.select(
        "*",
        F.max(F.col("current_dpd_numeric"))
        .over(history_window)
        .alias("max_dpd_to_date"),
        F.max(F.col("current_dpd_30_plus"))
        .over(history_window)
        .cast("byte")
        .alias("ever_30dpd_to_date"),
        F.max(F.col("current_dpd_60_plus"))
        .over(history_window)
        .cast("byte")
        .alias("ever_60dpd_to_date"),
        F.sum(F.col("current_delinquency_flag"))
        .over(history_window)
        .cast("short")
        .alias("delinquency_months_to_date"),
        F.max(
            F.when(
                F.col("current_delinquency_flag") == 1,
                F.col("calculated_loan_age"),
            )
        )
        .over(history_window)
        .alias("_last_delinquency_age"),
    )

    return result.withColumn(
        "months_since_last_delinquency",
        F.when(
            F.col("_last_delinquency_age").isNull(),
            F.lit(None).cast("double"),
        ).otherwise(F.col("calculated_loan_age") - F.col("_last_delinquency_age")),
    ).drop("_last_delinquency_age")


# ----------------------------------------------------------------------
# Loan trajectory helper
# ----------------------------------------------------------------------


def add_loan_trajectory_features_spark(
    df: DataFrame,
) -> DataFrame:
    """
    Add post-origination loan trajectory features.
    """

    _validate_required_columns(
        df,
        {
            "original_upb",
            "current_actual_upb",
            "original_interest_rate",
            "current_interest_rate",
        },
        "loan trajectory features",
    )

    upb_change = F.col("current_actual_upb") - F.col("original_upb")

    return df.select(
        "*",
        upb_change.alias("upb_change_from_origination"),
        F.when(
            F.col("original_upb").isNull(),
            F.lit(None).cast("double"),
        )
        .when(
            F.col("original_upb") == 0,
            F.lit(None).cast("double"),
        )
        .otherwise(upb_change / F.col("original_upb"))
        .alias("upb_pct_change_from_origination"),
        (F.col("current_interest_rate") - F.col("original_interest_rate")).alias(
            "rate_change_from_origination"
        ),
    )
