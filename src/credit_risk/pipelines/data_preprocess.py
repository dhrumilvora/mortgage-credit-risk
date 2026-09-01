"""Preprocessing pipeline for mortgage credit-risk model construction."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

import pandas as pd
from pyspark.sql import DataFrame, functions as F

from credit_risk.data.writers import write_parquet, write_spark_parquet
from credit_risk.data.readers import read_spark_parquet
from credit_risk.features import behavioral, origination, performance
from credit_risk.features.behavioral_spark import (
    add_calculated_loan_age_spark,
    build_behavioral_features_spark,
)
from credit_risk.features.eligibility_origination import (
    validate_baseline_features,
)
from credit_risk.features.eligibility_performance import (
    BASELINE_FEATURES,
    CHALLENGER_FEATURES,
    IDENTIFIER_FIELDS,
    STATE_FIELDS,
    TERMINATION_FIELDS,
    TIME_FIELDS,
    validate_features,
)
from credit_risk.features.origination_spark import (
    build_origination_spark,
)
from credit_risk.target.behavioral import (
    build_behavioral_target,
)
from credit_risk.target.behavioral_spark import (
    build_behavioral_target_spark,
)
from credit_risk.target.delinquency import (
    build_24m_serious_delinquency_target,
)
from credit_risk.utils.config import create_path
from credit_risk.features.feature_engineering import (
    apply_transformations,
    build_interaction_features,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Existing Pandas preprocessing contracts
# ----------------------------------------------------------------------


def build_origination(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Apply finalized origination preprocessing."""

    validate_baseline_features(
        df.columns,
        config,
    )

    result = origination.select_baseline_features(
        df,
        config,
    )

    result = origination.normalize_sentinel_values(
        result,
    )

    result = origination.add_missing_indicators(
        result,
    )

    return result


def build_performance(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Apply finalized performance preprocessing."""

    validate_features(
        df.columns,
    )

    return performance.select_baseline_features(
        df,
    )


def build_master_dataset(
    origination_df: pd.DataFrame,
    performance_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the Pandas master loan-month dataset.

    Reference implementation only.
    """

    return performance_df.merge(
        origination_df,
        on="loan_id",
        how="inner",
        validate="many_to_one",
    )


# ----------------------------------------------------------------------
# Spark performance preprocessing
# ----------------------------------------------------------------------


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

    The current performance preprocessing contract consists only of
    selecting the configured baseline columns.
    """

    return select_baseline_features_spark(
        df,
    )


# ----------------------------------------------------------------------
# Spark master join
# ----------------------------------------------------------------------


def build_master_dataset_spark(
    origination_features: DataFrame,
    performance_features: DataFrame,
) -> DataFrame:
    """
    Build the Spark loan-month master dataset.

    Equivalent to the Pandas:

        performance_features.merge(
            origination_features,
            on="loan_id",
            how="inner",
            validate="many_to_one",
        )

    Spark does not expose pandas' validate argument, so the
    one-row-per-loan origination contract remains a data-quality
    invariant.
    """

    return performance_features.join(
        origination_features,
        on="loan_id",
        how="inner",
    )


# ----------------------------------------------------------------------
# Pandas origination pipeline
# ----------------------------------------------------------------------


def build_modelling_dataset_origination_pandas(
    config: dict,
) -> None:
    """
    Existing Pandas origination modelling pipeline.

    Retained as the reference implementation.
    """

    data_config = config["parameters"]["data"]
    approach = config["parameters"]["modelling_approach"]

    if data_config["preprocess"]["skip"]:
        logger.info(
            "Preprocessing skipped by configuration",
        )
        return

    start = perf_counter()

    for vintage in data_config["all_vintages"]:

        provider = data_config["data_provider"]

        logger.info(
            "Pandas origination preprocessing started: " "provider=%s vintage=%s",
            provider,
            vintage,
        )

        origination_path = create_path(
            config["catalog"]["base"],
            config["catalog"],
            "origination_path",
            provider,
            vintage,
        )

        performance_path = create_path(
            config["catalog"]["base"],
            config["catalog"],
            "performance_path",
            provider,
            vintage,
        )

        origination_df = pd.read_parquet(
            origination_path,
        )

        performance_df = pd.read_parquet(
            performance_path,
        )

        origination_features = build_origination(
            origination_df,
            config,
        )

        performance_features = build_performance(
            performance_df,
        )

        master = build_master_dataset(
            origination_features,
            performance_features,
        )

        target = build_24m_serious_delinquency_target(
            master,
            config,
        )

        modelling = origination_features.merge(
            target,
            on="loan_id",
            how="inner",
            validate="one_to_one",
        ).reset_index(
            drop=True,
        )

        modelling = apply_transformations(
            modelling,
            config,
        )

        modelling = build_interaction_features(
            modelling,
            config,
        )

        model_input_path = create_path(
            config["catalog"]["base"],
            config["catalog"],
            "model_input_path",
            approach,
            provider,
            vintage,
            must_exist=False,
        )

        write_parquet(
            modelling,
            model_input_path,
        )

        logger.info(
            "Pandas origination preprocessing completed: "
            "provider=%s vintage=%s rows=%s columns=%s "
            "events=%s path=%s duration_seconds=%.2f",
            provider,
            vintage,
            f"{len(modelling):,}",
            modelling.shape[1],
            f"{int(modelling['ever_90dpd_24m'].sum()):,}",
            model_input_path,
            perf_counter() - start,
        )


# ----------------------------------------------------------------------
# Pandas behavioral pipeline
# ----------------------------------------------------------------------


def build_modelling_dataset_behavioral_pandas(
    config: dict,
) -> None:
    """
    Existing Pandas behavioral modelling pipeline.

    Retained as the reference implementation.
    """

    data_config = config["parameters"]["data"]
    approach = config["parameters"]["modelling_approach"]

    if data_config["preprocess"]["skip"]:
        logger.info(
            "Preprocessing skipped by configuration",
        )
        return

    start = perf_counter()

    for vintage in data_config["all_vintages"]:

        provider = data_config["data_provider"]

        logger.info(
            "Pandas behavioral preprocessing started: " "provider=%s vintage=%s",
            provider,
            vintage,
        )

        origination_path = create_path(
            config["catalog"]["base"],
            config["catalog"],
            "origination_path",
            provider,
            vintage,
        )

        performance_path = create_path(
            config["catalog"]["base"],
            config["catalog"],
            "performance_path",
            provider,
            vintage,
        )

        origination_df = pd.read_parquet(
            origination_path,
        )

        performance_df = pd.read_parquet(
            performance_path,
        )

        origination_features = build_origination(
            origination_df,
            config,
        )

        origination_dates = origination_df[
            [
                "loan_id",
                "first_payment_date",
            ]
        ].drop_duplicates(
            subset=["loan_id"],
        )

        origination_features = origination_features.merge(
            origination_dates,
            on="loan_id",
            how="left",
            validate="one_to_one",
        )

        performance_features = build_performance(
            performance_df,
        )

        master = build_master_dataset(
            origination_features,
            performance_features,
        )

        master = behavioral.add_calculated_loan_age(
            master,
        )

        behavioral_features = behavioral.build_behavioral_features(
            master,
            config,
        )

        if behavioral_features.empty:
            logger.warning(
                "No behavioral feature population generated: " "provider=%s vintage=%s",
                provider,
                vintage,
            )
            continue

        target = build_behavioral_target(
            master,
            config,
        )

        if target.empty:
            logger.warning(
                "No behavioral target generated: " "provider=%s vintage=%s",
                provider,
                vintage,
            )
            continue

        target_duplicate_mask = target.duplicated(
            subset=[
                "loan_id",
                "observation_age",
            ],
            keep=False,
        )

        if target_duplicate_mask.any():

            duplicate_count = int(
                target_duplicate_mask.sum(),
            )

            raise ValueError(
                "Behavioral target contains duplicate "
                "loan_id x observation_age rows: "
                f"{duplicate_count:,}",
            )

        modelling = behavioral_features.merge(
            target,
            on=[
                "loan_id",
                "observation_age",
            ],
            how="inner",
            validate="one_to_one",
        ).reset_index(
            drop=True,
        )

        if modelling.empty:
            logger.warning(
                "Behavioral modelling dataset is empty after "
                "feature-target merge: provider=%s vintage=%s",
                provider,
                vintage,
            )
            continue

        modelling_duplicate_mask = modelling.duplicated(
            subset=[
                "loan_id",
                "observation_age",
            ],
            keep=False,
        )

        if modelling_duplicate_mask.any():

            duplicate_count = int(
                modelling_duplicate_mask.sum(),
            )

            raise ValueError(
                "Behavioral modelling dataset contains duplicate "
                "loan_id x observation_age rows: "
                f"{duplicate_count:,}",
            )

        if not (modelling["calculated_loan_age"] == modelling["observation_age"]).all():

            raise ValueError(
                "Behavioral modelling dataset contains rows where "
                "calculated_loan_age != observation_age.",
            )

        model_input_path = create_path(
            config["catalog"]["base"],
            config["catalog"],
            "model_input_path",
            approach,
            provider,
            vintage,
            must_exist=False,
        )

        write_parquet(
            modelling,
            model_input_path,
        )

        logger.info(
            "Pandas behavioral preprocessing completed: "
            "provider=%s vintage=%s rows=%s columns=%s "
            "unique_loans=%s observation_ages=%s events=%s "
            "path=%s duration_seconds=%.2f",
            provider,
            vintage,
            f"{len(modelling):,}",
            modelling.shape[1],
            f"{modelling['loan_id'].nunique():,}",
            sorted(modelling["observation_age"].unique().tolist()),
            f"{int(modelling['future_90dpd_12m'].sum()):,}",
            model_input_path,
            perf_counter() - start,
        )


# ----------------------------------------------------------------------
# PySpark behavioral pipeline
# ----------------------------------------------------------------------


def build_modelling_dataset_behavioral_pyspark(config: dict, spark) -> None:
    """
    Build the behavioral modelling dataset using PySpark.

    Heavy performance, master, behavioral-feature, and target
    operations remain Spark DataFrames throughout preprocessing.
    """

    data_config = config["parameters"]["data"]
    approach = config["parameters"]["modelling_approach"]

    if data_config["preprocess"]["skip"]:
        logger.info(
            "Preprocessing skipped by configuration",
        )
        return

    start = perf_counter()

    try:

        for vintage in data_config["all_vintages"]:

            provider = data_config["data_provider"]

            vintage_start = perf_counter()

            logger.info(
                "PySpark behavioral preprocessing started: " "provider=%s vintage=%s",
                provider,
                vintage,
            )

            # ----------------------------------------------------------
            # Resolve canonical paths
            # ----------------------------------------------------------

            origination_path = create_path(
                config["catalog"]["base"],
                config["catalog"],
                "origination_path",
                provider,
                vintage,
            )

            performance_path = create_path(
                config["catalog"]["base"],
                config["catalog"],
                "performance_path",
                provider,
                vintage,
            )

            # ----------------------------------------------------------
            # Read canonical datasets
            # ----------------------------------------------------------

            logger.info(
                "Spark reading canonical origination data: %s",
                origination_path,
            )

            origination_df = read_spark_parquet(
                spark,
                origination_path,
            )

            logger.info(
                "Spark reading canonical performance data: %s",
                performance_path,
            )

            performance_df = read_spark_parquet(
                spark,
                performance_path,
            )

            logger.info(
                "Canonical Spark datasets loaded: "
                "vintage=%s origination_columns=%s "
                "performance_columns=%s",
                vintage,
                len(origination_df.columns),
                len(performance_df.columns),
            )

            # ----------------------------------------------------------
            # Origination preprocessing
            #
            # This reproduces the existing Pandas:
            #
            #   select_baseline_features
            #   normalize_sentinel_values
            #   add_missing_indicators
            # ----------------------------------------------------------

            origination_features = build_origination_spark(
                origination_df,
                config,
            )

            # ----------------------------------------------------------
            # Preserve first_payment_date for the V2 lifecycle clock.
            #
            # This field is intentionally not part of the baseline
            # origination feature contract.
            # ----------------------------------------------------------

            origination_dates = origination_df.select(
                "loan_id",
                "first_payment_date",
            ).dropDuplicates(
                ["loan_id"],
            )

            origination_features = origination_features.join(
                origination_dates,
                on="loan_id",
                how="left",
            )

            # ----------------------------------------------------------
            # Performance preprocessing
            #
            # This is lazy column selection. The full performance
            # dataset is NOT converted to Pandas.
            # ----------------------------------------------------------

            performance_features = build_performance_spark(
                performance_df,
            )

            logger.info(
                "Spark feature preprocessing configured: "
                "vintage=%s origination_columns=%s "
                "performance_columns=%s",
                vintage,
                len(origination_features.columns),
                len(performance_features.columns),
            )

            # ----------------------------------------------------------
            # Release canonical DataFrame references.
            #
            # Spark transformations remain lazy, so this does not
            # materialize or duplicate the data.
            # ----------------------------------------------------------

            del origination_df
            del origination_dates
            del performance_df
            # ----------------------------------------------------------
            # Build master loan-month dataset.
            # ----------------------------------------------------------

            master = build_master_dataset_spark(
                origination_features=origination_features,
                performance_features=performance_features,
            )

            # Once the join plan has been created, these references
            # are no longer needed separately.
            del origination_features
            del performance_features

            # ----------------------------------------------------------
            # Build V2 lifecycle clock.
            # ----------------------------------------------------------

            master = add_calculated_loan_age_spark(
                master,
            )
            master.filter(F.col("calculated_loan_age").isin([6, 12])).agg(
                F.count("*").alias("rows"),
                F.count("ddlpi").alias("ddlpi_non_null"),
                F.count("delinquent_accrued_interest").alias(
                    "delinquent_accrued_interest_non_null"
                ),
            ).show()

            logger.info(
                "Spark master transformation configured: " "vintage=%s columns=%s",
                vintage,
                len(master.columns),
            )

            # ----------------------------------------------------------
            # Build point-in-time behavioral features.
            # ----------------------------------------------------------

            behavioral_features = build_behavioral_features_spark(
                master,
                config,
            )

            logger.info(
                "Spark behavioral feature transformation configured: "
                "vintage=%s columns=%s",
                vintage,
                len(behavioral_features.columns),
            )

            # ----------------------------------------------------------
            # Build forward-looking behavioral target.
            # ----------------------------------------------------------

            target = build_behavioral_target_spark(
                master,
                config,
            )

            logger.info(
                "Spark behavioral target transformation configured: "
                "vintage=%s columns=%s",
                vintage,
                len(target.columns),
            )

            # The master is no longer referenced after both feature
            # and target plans have been constructed.
            del master

            # ----------------------------------------------------------
            # Join features to target.
            #
            # Expected grain:
            #     loan_id x observation_age
            # ----------------------------------------------------------

            modelling = behavioral_features.join(
                target,
                on=[
                    "loan_id",
                    "observation_age",
                ],
                how="inner",
            )

            # ----------------------------------------------------------
            # Defensive duplicate check.
            #
            # This is intentionally performed as a Spark aggregation,
            # not by collecting the complete dataset.
            # ----------------------------------------------------------

            duplicate_keys = (
                modelling.groupBy(
                    "loan_id",
                    "observation_age",
                )
                .count()
                .filter("count > 1")
                .limit(1)
            )

            if duplicate_keys.count() > 0:
                raise ValueError(
                    "Behavioral modelling dataset contains duplicate "
                    "loan_id x observation_age rows."
                )

            # ----------------------------------------------------------
            # Validate calculated loan age.
            #
            # Both columns should contain the same value for every
            # output observation.
            # ----------------------------------------------------------

            invalid_age_rows = modelling.filter(
                "calculated_loan_age != observation_age"
            ).limit(1)

            if invalid_age_rows.count() > 0:
                raise ValueError(
                    "Behavioral modelling dataset contains rows where "
                    "calculated_loan_age != observation_age."
                )

            # ----------------------------------------------------------
            # Persist final modelling dataset.
            #
            # This is still a Spark write. No giant toPandas() call.
            # ----------------------------------------------------------

            model_input_path = create_path(
                config["catalog"]["base"],
                config["catalog"],
                "model_input_path",
                approach,
                provider,
                vintage,
                must_exist=False,
            )

            write_spark_parquet(
                modelling,
                model_input_path,
            )

            logger.info(
                "PySpark behavioral preprocessing completed: "
                "provider=%s vintage=%s columns=%s "
                "path=%s duration_seconds=%.2f",
                provider,
                vintage,
                len(modelling.columns),
                model_input_path,
                perf_counter() - vintage_start,
            )

            # ----------------------------------------------------------
            # Release the completed vintage's logical plan references.
            # ----------------------------------------------------------

            del behavioral_features
            del target
            del modelling

    finally:

        logger.info(
            "PySpark behavioral preprocessing duration_seconds=%.2f",
            perf_counter() - start,
        )


# ----------------------------------------------------------------------
# PySpark origination pipeline
# ----------------------------------------------------------------------


def build_modelling_dataset_origination_pyspark(config: dict, spark) -> None:
    """
    Build the origination modelling dataset using PySpark.

    This path is separate from the behavioral pipeline because the
    target and final modelling grain are different.
    """

    data_config = config["parameters"]["data"]
    approach = config["parameters"]["modelling_approach"]

    if data_config["preprocess"]["skip"]:
        logger.info(
            "Preprocessing skipped by configuration",
        )
        return

    start = perf_counter()

    try:

        for vintage in data_config["all_vintages"]:

            provider = data_config["data_provider"]

            vintage_start = perf_counter()

            logger.info(
                "PySpark origination preprocessing started: " "provider=%s vintage=%s",
                provider,
                vintage,
            )

            origination_path = create_path(
                config["catalog"]["base"],
                config["catalog"],
                "origination_path",
                provider,
                vintage,
            )

            performance_path = create_path(
                config["catalog"]["base"],
                config["catalog"],
                "performance_path",
                provider,
                vintage,
            )

            origination_df = read_spark_parquet(
                spark,
                origination_path,
            )

            performance_df = read_spark_parquet(
                spark,
                performance_path,
            )

            origination_features = build_origination_spark(
                origination_df,
                config,
            )

            performance_features = build_performance_spark(
                performance_df,
            )

            master = build_master_dataset_spark(
                origination_features,
                performance_features,
            )

            raise NotImplementedError(
                "PySpark origination target construction has not yet "
                "been migrated. Use modelling_approach='behavioral' "
                "for the current Spark migration."
            )

            logger.info(
                "PySpark origination preprocessing completed: "
                "provider=%s vintage=%s duration_seconds=%.2f",
                provider,
                vintage,
                perf_counter() - vintage_start,
            )

    finally:

        logger.info(
            "PySpark origination preprocessing duration_seconds=%.2f",
            perf_counter() - start,
        )


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------


def build_modelling_dataset(config: dict, spark=None) -> None:
    """
    Build the modelling dataset using the configured preprocessing
    engine.

    PySpark is the default engine.
    Pandas remains available as the reference implementation.
    """

    approach = config["parameters"]["modelling_approach"]

    preprocess_config = config["parameters"]["data"]["preprocess"]

    if preprocess_config["skip"]:
        logger.info(
            "Preprocessing skipped by configuration",
        )
        return

    engine = config["parameters"].get(
        "engine",
    )

    if not isinstance(
        engine,
        str,
    ):
        raise ValueError(
            "preprocess.engine must be a string.",
        )

    engine = engine.strip().lower()

    logger.info(
        "Preprocessing configuration: " "engine=%s approach=%s",
        engine,
        approach,
    )

    if engine == "pyspark":

        if approach == "behavioral":

            build_modelling_dataset_behavioral_pyspark(config, spark)

        elif approach == "origination":

            build_modelling_dataset_origination_pyspark(config, spark)

        else:

            raise ValueError(
                f"Unsupported modelling approach: {approach}. "
                "Expected 'origination' or 'behavioral'."
            )

    elif engine == "pandas":

        if approach == "behavioral":

            build_modelling_dataset_behavioral_pandas(
                config,
            )

        elif approach == "origination":

            build_modelling_dataset_origination_pandas(
                config,
            )

        else:

            raise ValueError(
                f"Unsupported modelling approach: {approach}. "
                "Expected 'origination' or 'behavioral'."
            )

    else:

        raise ValueError(
            f"Unsupported preprocessing engine: {engine}. "
            "Expected 'origination' or 'pandas'."
        )
