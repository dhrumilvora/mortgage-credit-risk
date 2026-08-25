"""PySpark behavioral target construction for mortgage credit-risk modelling."""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


def _validate_required_columns(
    df: DataFrame,
    required_columns: set[str],
    context: str,
) -> None:
    """Validate that a Spark DataFrame contains required columns."""

    missing_columns = sorted(required_columns - set(df.columns))

    if missing_columns:
        raise ValueError(
            f"Missing columns required for {context}: " + ", ".join(missing_columns)
        )


# ----------------------------------------------------------------------
# Serious delinquency
# ----------------------------------------------------------------------


def add_serious_delinquency_flag_spark(
    df: DataFrame,
    config: dict,
) -> DataFrame:
    """
    Add a row-level serious-delinquency indicator.

    A serious delinquency is defined as either:

    - numeric delinquency status >= configured threshold; or
    - REO acquisition status ("RA").

    This matches the existing Pandas implementation.

    Non-numeric delinquency values such as "RA" are converted to null
    for numeric comparison and are handled explicitly through the
    "RA" condition.
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
# Observation population
# ----------------------------------------------------------------------


def _build_observation_population_spark(
    df: DataFrame,
    observation_age: int,
) -> DataFrame:
    """
    Build the population observable at a specific loan age.

    Output grain:

        one row per loan
    """

    return (
        df.filter(F.col("calculated_loan_age") == F.lit(observation_age))
        .select(
            "loan_id",
            "calculated_loan_age",
        )
        .dropDuplicates(
            ["loan_id"],
        )
    )


# ----------------------------------------------------------------------
# Future window
# ----------------------------------------------------------------------


def _build_future_window_spark(
    df: DataFrame,
    observation_age: int,
    prediction_horizon: int,
) -> DataFrame:
    """
    Select performance observations in the forward prediction window.

    Window:

        observation_age + 1
        through
        observation_age + prediction_horizon
    """

    future_start = observation_age + 1

    future_end = observation_age + prediction_horizon

    return df.filter(
        F.col("calculated_loan_age").between(
            future_start,
            future_end,
        )
    )


# ----------------------------------------------------------------------
# Future outcome summary
# ----------------------------------------------------------------------


def _summarize_future_outcome_spark(
    future_df: DataFrame,
) -> DataFrame:
    """
    Summarize future performance to one row per loan.

    Summary contains:

    - last observed loan age in the future window;
    - whether serious delinquency ever occurred;
    - final non-null zero-balance code observed in the future window.
    """

    _validate_required_columns(
        future_df,
        {
            "loan_id",
            "calculated_loan_age",
            "is_serious_delinquency",
            "zero_balance_code",
        },
        "future outcome summary",
    )

    # ------------------------------------------------------------------
    # Find the latest non-null zero-balance code by loan.
    #
    # Ordering descending means FIRST(ignoreNulls=True) is the same
    # business operation as taking the last non-null value after
    # chronological ascending sort in the Pandas implementation.
    # ------------------------------------------------------------------

    future_window = (
        Window.partitionBy(
            "loan_id",
        )
        .orderBy(
            F.col("calculated_loan_age").desc(),
        )
        .rowsBetween(
            Window.unboundedPreceding,
            Window.unboundedFollowing,
        )
    )

    summarized = (
        future_df.select(
            "loan_id",
            "calculated_loan_age",
            "is_serious_delinquency",
            "zero_balance_code",
        )
        .withColumn(
            "_last_non_null_zero_balance_code",
            F.first(
                F.col("zero_balance_code"),
                ignorenulls=True,
            ).over(
                future_window,
            ),
        )
        .groupBy(
            "loan_id",
        )
        .agg(
            F.max(
                "calculated_loan_age",
            ).alias(
                "last_future_loan_age",
            ),
            F.max(
                F.col("is_serious_delinquency").cast("int"),
            )
            .cast("boolean")
            .alias(
                "ever_serious_delinquency",
            ),
            F.first(
                F.col("_last_non_null_zero_balance_code"),
                ignorenulls=True,
            ).alias(
                "final_zero_balance_code",
            ),
        )
    )

    return summarized


# ----------------------------------------------------------------------
# Observability
# ----------------------------------------------------------------------


def _add_observability_flags_spark(
    future_summary: DataFrame,
    observation_age: int,
    prediction_horizon: int,
    voluntary_payoff_zbc,
) -> DataFrame:
    """
    Add future-horizon completion and observability flags.

    An outcome is observable when:

    - the complete prediction horizon is observed;
    - serious delinquency is observed; or
    - the loan voluntarily pays off before the end of the horizon.
    """

    future_end = observation_age + prediction_horizon

    result = future_summary.withColumn(
        "completed_horizon",
        F.coalesce(
            F.col("last_future_loan_age")
            >= F.lit(
                future_end,
            ),
            F.lit(False),
        ),
    )

    result = result.withColumn(
        "voluntary_early_payoff",
        F.coalesce(
            (
                F.col("last_future_loan_age")
                < F.lit(
                    future_end,
                )
            )
            & (
                F.col("final_zero_balance_code")
                == F.lit(
                    voluntary_payoff_zbc,
                )
            ),
            F.lit(False),
        ),
    )

    result = result.withColumn(
        "is_outcome_observable",
        (
            F.coalesce(
                F.col("ever_serious_delinquency"),
                F.lit(False),
            )
            | F.coalesce(
                F.col("completed_horizon"),
                F.lit(False),
            )
            | F.coalesce(
                F.col("voluntary_early_payoff"),
                F.lit(False),
            )
        ),
    )

    return result


# ----------------------------------------------------------------------
# Observation grain validation
# ----------------------------------------------------------------------


def _validate_observation_grain_spark(
    observation_df: DataFrame,
    observation_age: int,
) -> None:
    """
    Validate that the observation population contains at most one row
    per loan for a given observation age.
    """

    duplicate = (
        observation_df.groupBy(
            "loan_id",
        )
        .count()
        .filter(
            F.col("count") > 1,
        )
        .limit(1)
    )

    if duplicate.count() > 0:
        raise ValueError(
            "Multiple performance rows found for the same loan at "
            f"observation_age={observation_age}."
        )


# ----------------------------------------------------------------------
# Single observation-age target
# ----------------------------------------------------------------------


def _build_observation_target_spark(
    df: DataFrame,
    observation_age: int,
    prediction_horizon: int,
    voluntary_payoff_zbc,
) -> DataFrame:
    """
    Build the behavioral target for one observation age.

    Output grain:

        loan_id x observation_age
    """

    observation_df = _build_observation_population_spark(
        df=df,
        observation_age=observation_age,
    )

    _validate_observation_grain_spark(
        observation_df=observation_df,
        observation_age=observation_age,
    )

    future_df = _build_future_window_spark(
        df=df,
        observation_age=observation_age,
        prediction_horizon=prediction_horizon,
    )

    future_summary = _summarize_future_outcome_spark(
        future_df=future_df,
    )

    future_summary = _add_observability_flags_spark(
        future_summary=future_summary,
        observation_age=observation_age,
        prediction_horizon=prediction_horizon,
        voluntary_payoff_zbc=voluntary_payoff_zbc,
    )

    result = (
        observation_df.join(
            future_summary,
            on="loan_id",
            how="left",
        )
        .withColumn(
            "is_target_eligible",
            F.coalesce(
                F.col("is_outcome_observable"),
                F.lit(False),
            ),
        )
        .withColumn(
            "future_90dpd_12m",
            F.coalesce(
                F.col("ever_serious_delinquency"),
                F.lit(False),
            ).cast("byte"),
        )
        .withColumn(
            "observation_age",
            F.lit(observation_age).cast("int"),
        )
        .filter(F.col("is_target_eligible"))
        .select(
            "loan_id",
            "observation_age",
            "future_90dpd_12m",
        )
    )

    return result


# ----------------------------------------------------------------------
# Complete target
# ----------------------------------------------------------------------


def build_behavioral_target_spark(
    df: DataFrame,
    config: dict,
) -> DataFrame:
    """
    Build point-in-time behavioral targets.

    Each eligible row represents a loan at a configured observation age.

    Target definition:

        future_90dpd_12m = 1

    when serious delinquency occurs between:

        observation_age + 1

    and:

        observation_age + prediction_horizon

    Output grain:

        loan_id x observation_age

    The implementation deliberately avoids per-observation-age
    count() actions. Spark remains lazy until a downstream action,
    such as the final write, is triggered.
    """

    behavioral_config = config["parameters"]["behavioral"]

    observation_ages = behavioral_config.get(
        "observation_ages",
        [],
    )

    if not observation_ages:
        raise ValueError("parameters.behavioral.observation_ages " "cannot be empty.")

    observation_ages = list(dict.fromkeys(int(age) for age in observation_ages))

    invalid_ages = [age for age in observation_ages if age < 0]

    if invalid_ages:
        raise ValueError(
            "parameters.behavioral.observation_ages contains "
            "negative values: " + ", ".join(str(age) for age in invalid_ages)
        )

    prediction_horizon = behavioral_config["prediction_horizon_months"]

    voluntary_payoff_zbc = config["parameters"]["target"]["voluntary_payoffs_zbc"]

    _validate_required_columns(
        df,
        {
            "loan_id",
            "calculated_loan_age",
            "current_loan_delinquency_status",
            "zero_balance_code",
        },
        "behavioral target construction",
    )

    # ------------------------------------------------------------------
    # Add serious-delinquency flag exactly once.
    # ------------------------------------------------------------------

    working = add_serious_delinquency_flag_spark(
        df=df,
        config=config,
    )

    # ------------------------------------------------------------------
    # Build each observation-age target lazily.
    #
    # We intentionally do NOT call:
    #
    #     observation_target.limit(1).count()
    #
    # because that would execute the full Spark lineage repeatedly.
    # ------------------------------------------------------------------

    results: list[DataFrame] = []

    for observation_age in observation_ages:
        observation_target = _build_observation_target_spark(
            df=working,
            observation_age=observation_age,
            prediction_horizon=prediction_horizon,
            voluntary_payoff_zbc=voluntary_payoff_zbc,
        )

        results.append(
            observation_target,
        )

    if not results:
        return (
            working.select(
                "loan_id",
            )
            .limit(0)
            .withColumn(
                "observation_age",
                F.lit(None).cast("int"),
            )
            .withColumn(
                "future_90dpd_12m",
                F.lit(None).cast("byte"),
            )
        )

    # ------------------------------------------------------------------
    # Union all configured observation ages.
    # ------------------------------------------------------------------

    target = results[0]

    for observation_target in results[1:]:
        target = target.unionByName(
            observation_target,
        )

    # ------------------------------------------------------------------
    # Final grain validation.
    # ------------------------------------------------------------------

    duplicate = (
        target.groupBy(
            "loan_id",
            "observation_age",
        )
        .count()
        .filter(
            F.col("count") > 1,
        )
        .limit(1)
    )

    if duplicate.count() > 0:
        raise ValueError(
            "Behavioral target contains duplicate " "loan_id x observation_age rows."
        )

    logger.info(
        "Spark behavioral target plan configured: "
        "observation_ages=%s prediction_horizon=%s",
        observation_ages,
        prediction_horizon,
    )

    return target
