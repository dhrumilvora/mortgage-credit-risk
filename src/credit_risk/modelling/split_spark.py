"""PySpark development dataset splitting utilities."""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


def _validate_common_inputs(
    df: DataFrame,
    config: dict,
) -> tuple[str, str, float, int]:
    """Validate inputs shared by Spark splitting strategies."""

    target_config = config["parameters"]["target"]
    modelling_config = config["parameters"]["modelling"]
    approach = config["parameters"]["modelling_approach"]

    target = target_config["name"]
    validation_size = modelling_config["validation_size"]
    random_state = modelling_config["random_state"]
    split_type = modelling_config["train_test_split"]

    if approach not in {
        "origination",
        "behavioral",
    }:
        raise ValueError(f"Unsupported modelling approach: {approach}")

    required_columns = {
        target,
    }

    if approach == "behavioral":
        required_columns.update(
            {
                "loan_id",
                "observation_age",
            }
        )

    missing_columns = sorted(required_columns - set(df.columns))

    if missing_columns:
        raise ValueError(
            "Missing modelling split columns: " + ", ".join(missing_columns)
        )

    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")

    return (
        approach,
        target,
        validation_size,
        random_state,
    )


def stratified_data_split_spark(
    df: DataFrame,
    config: dict,
) -> tuple[DataFrame, DataFrame]:
    """
    Split development data into training and validation populations.

    Origination:
        Row-level stratified split.

    Behavioral:
        Loan-level random split. All observations belonging to the same
        loan remain in the same partition.

    The behavioral implementation is equivalent in intent to the
    Pandas GroupShuffleSplit implementation, while keeping the data
    distributed in Spark.
    """

    (
        approach,
        target,
        validation_size,
        random_state,
    ) = _validate_common_inputs(
        df,
        config,
    )

    # ------------------------------------------------------------------
    # Validate target.
    # ------------------------------------------------------------------

    target_null = (
        df.filter(
            F.col(target).isNull(),
        )
        .limit(1)
        .count()
    )

    if target_null > 0:
        raise ValueError(f"Target column contains missing values: {target}")

    target_classes = df.select(target).distinct().limit(2).count()

    if target_classes < 2:
        raise ValueError(f"Target must contain at least two classes: {target}")

    # ------------------------------------------------------------------
    # Origination
    #
    # Preserve row-level stratification.
    #
    # Spark's sampleBy provides approximate stratification. The split
    # remains distributed and avoids converting the dataset to Pandas.
    # ------------------------------------------------------------------

    if approach == "origination":

        if config["parameters"]["modelling"]["stratify"]:

            fractions = {
                row[target]: validation_size
                for row in (df.select(target).distinct().collect())
            }

            validation_df = df.sampleBy(
                col=target,
                fractions=fractions,
                seed=random_state,
            )

            validation_keys = validation_df.select(
                *df.columns,
            )

            train_df = df.join(
                validation_keys,
                on=df.columns,
                how="left_anti",
            )

        else:

            validation_df = df.sample(
                withReplacement=False,
                fraction=validation_size,
                seed=random_state,
            )

            validation_keys = validation_df.select(
                *df.columns,
            )

            train_df = df.join(
                validation_keys,
                on=df.columns,
                how="left_anti",
            )

    # ------------------------------------------------------------------
    # Behavioral
    #
    # Assign the split at loan level.
    #
    # This is critical. We must never independently sample individual
    # observations because that could put age-6 and age-12 observations
    # for the same loan into different populations.
    # ------------------------------------------------------------------

    elif approach == "behavioral":

        null_loans = (
            df.filter(
                F.col("loan_id").isNull(),
            )
            .limit(1)
            .count()
        )

        if null_loans > 0:
            raise ValueError("loan_id contains missing values.")

        loan_assignments = (
            df.select("loan_id")
            .distinct()
            .withColumn(
                "_random",
                F.rand(random_state),
            )
            .withColumn(
                "_split",
                F.when(
                    F.col("_random") < F.lit(validation_size),
                    F.lit("validation"),
                ).otherwise(
                    F.lit("train"),
                ),
            )
            .select(
                "loan_id",
                "_split",
            )
        )

        validation_loans = loan_assignments.filter(
            F.col("_split") == F.lit("validation"),
        ).select("loan_id")

        train_loans = loan_assignments.filter(
            F.col("_split") == F.lit("train"),
        ).select("loan_id")

        validation_df = df.join(
            validation_loans,
            on="loan_id",
            how="inner",
        )

        train_df = df.join(
            train_loans,
            on="loan_id",
            how="inner",
        )

        # --------------------------------------------------------------
        # Explicit leakage validation.
        # --------------------------------------------------------------

        overlap = (
            train_loans.join(
                validation_loans,
                on="loan_id",
                how="inner",
            )
            .limit(1)
            .count()
        )

        if overlap > 0:
            raise ValueError(
                "Loan leakage detected between training and " "validation sets."
            )

    else:
        raise ValueError(f"Unsupported modelling approach: {approach}")

    # ------------------------------------------------------------------
    # Validate non-empty populations.
    # ------------------------------------------------------------------

    if train_df.limit(1).count() == 0:
        raise ValueError("Training population is empty.")

    if validation_df.limit(1).count() == 0:
        raise ValueError("Validation population is empty.")

    logger.info(
        "Spark random data split completed: " "approach=%s validation_size=%.3f",
        approach,
        validation_size,
    )

    return train_df, validation_df


def yearly_data_split_spark(
    df: DataFrame,
    config: dict,
) -> tuple[DataFrame, DataFrame]:
    """
    Split modelling data using chronological vintage boundaries.

    Training observations come from configured training vintages.

    Validation observations come from configured validation vintages.

    No data is collected into Python.
    """

    modelling_config = config["parameters"]["modelling"]
    approach = config["parameters"]["modelling_approach"]

    train_vintages = sorted(set(modelling_config["vintages_train"]))

    validation_vintages = sorted(set(modelling_config["vintages_test"]))

    if not train_vintages:
        raise ValueError("At least one training vintage must be provided.")

    if not validation_vintages:
        raise ValueError("At least one validation vintage must be provided.")

    if "vintage" not in df.columns:
        raise ValueError("Vintage column not found in modelling dataset.")

    overlapping_vintages = set(train_vintages).intersection(validation_vintages)

    if overlapping_vintages:
        raise ValueError(
            "Training and validation vintages overlap: "
            + ", ".join(
                map(
                    str,
                    sorted(overlapping_vintages),
                )
            )
        )

    latest_training_vintage = max(train_vintages)

    earliest_validation_vintage = min(validation_vintages)

    if earliest_validation_vintage <= latest_training_vintage:
        raise ValueError(
            "Validation vintages must be strictly later than "
            "all training vintages. "
            f"latest_training={latest_training_vintage}, "
            f"earliest_validation={earliest_validation_vintage}"
        )

    if approach == "behavioral":

        required_columns = {
            "loan_id",
            "observation_age",
        }

        missing_columns = sorted(required_columns - set(df.columns))

        if missing_columns:
            raise ValueError(
                "Missing behavioral split columns: " + ", ".join(missing_columns)
            )

        null_loans = (
            df.filter(
                F.col("loan_id").isNull(),
            )
            .limit(1)
            .count()
        )

        if null_loans > 0:
            raise ValueError("loan_id contains missing values.")

    elif approach != "origination":

        raise ValueError(f"Unsupported modelling approach: {approach}")

    # ------------------------------------------------------------------
    # Create chronological populations.
    # ------------------------------------------------------------------

    train_df = df.filter(F.col("vintage").isin(train_vintages))

    validation_df = df.filter(F.col("vintage").isin(validation_vintages))

    # ------------------------------------------------------------------
    # Validate non-empty populations.
    # ------------------------------------------------------------------

    if train_df.limit(1).count() == 0:
        raise ValueError(
            "Training population is empty for configured vintages: "
            + ", ".join(
                map(
                    str,
                    train_vintages,
                )
            )
        )

    if validation_df.limit(1).count() == 0:
        raise ValueError(
            "Validation population is empty for configured vintages: "
            + ", ".join(
                map(
                    str,
                    validation_vintages,
                )
            )
        )

    # ------------------------------------------------------------------
    # Behavioral leakage validation.
    #
    # This should normally be impossible if vintages represent
    # origination cohorts, but we explicitly verify it.
    # ------------------------------------------------------------------

    if approach == "behavioral":

        overlap = (
            train_df.select("loan_id")
            .distinct()
            .join(
                validation_df.select("loan_id").distinct(),
                on="loan_id",
                how="inner",
            )
            .limit(1)
            .count()
        )

        if overlap > 0:
            raise ValueError(
                "Loan leakage detected between chronological "
                "training and validation populations."
            )

    logger.info(
        "Spark chronological data split completed: "
        "train_vintages=%s validation_vintages=%s",
        ", ".join(
            map(
                str,
                train_vintages,
            )
        ),
        ", ".join(
            map(
                str,
                validation_vintages,
            )
        ),
    )

    return train_df, validation_df


def split_dataset_spark(
    df: DataFrame,
    config: dict,
) -> tuple[DataFrame, DataFrame]:
    """
    Dispatch to the configured Spark train/validation split strategy.
    """

    split_type = config["parameters"]["modelling"]["train_test_split"]

    if split_type == "random":
        return stratified_data_split_spark(
            df,
            config,
        )

    if split_type == "yearly":
        return yearly_data_split_spark(
            df,
            config,
        )

    raise ValueError(f"Unsupported Train Test Split: {split_type}")
