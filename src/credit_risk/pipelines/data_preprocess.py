"""Preprocessing pipeline for mortgage credit-risk model construction."""

from __future__ import annotations

import logging
from time import perf_counter

import pandas as pd

from credit_risk.data.writers import write_parquet
from credit_risk.features import origination, performance, behavioral
from credit_risk.features.eligibility_origination import (
    validate_baseline_features,
)
from credit_risk.features.eligibility_performance import (
    validate_features,
)
from credit_risk.target.delinquency import (
    build_24m_serious_delinquency_target,
)
from credit_risk.utils.config import create_path
from credit_risk.features.feature_engineering import (
    apply_transformations,
    build_interaction_features,
)
from credit_risk.target.behavioral import (
    build_behavioral_target,
)

logger = logging.getLogger(__name__)


def build_origination(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Apply finalized origination preprocessing."""

    validate_baseline_features(df.columns, config)

    result = origination.select_baseline_features(df, config)
    result = origination.normalize_sentinel_values(result)
    result = origination.add_missing_indicators(result)

    return result


def build_performance(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Apply finalized performance preprocessing."""

    validate_features(df.columns)

    return performance.select_baseline_features(df)


def build_master_dataset(
    origination_df: pd.DataFrame,
    performance_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the master loan-month dataset from preprocessed inputs.

    The origination dataset must contain one row per loan.
    The performance dataset may contain multiple monthly observations
    per loan.
    """

    return performance_df.merge(
        origination_df,
        on="loan_id",
        how="inner",
        validate="many_to_one",
    )


def build_modelling_dataset_origination(
    config: dict,
) -> None:
    """
    Build and persist the final loan-level modelling dataset.

    Performance history is used only for target construction.
    Final modelling features are sourced exclusively from the
    origination-time feature dataset.
    """

    data_config = config["parameters"]["data"]
    approach = config["parameters"]["modelling_approach"]

    if config["parameters"]["data"]["preprocess"]["skip"]:
        logger.info("Preprocessing skipped by configuration")
        return

    start = perf_counter()

    for vintage in config["parameters"]["data"]["all_vintages"]:

        provider = data_config["data_provider"]

        logger.info(
            "Dataset construction started: provider=%s vintage=%s",
            provider,
            vintage,
        )

        # ------------------------------------------------------------------
        # Resolve canonical input paths
        # ------------------------------------------------------------------

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

        # ------------------------------------------------------------------
        # Read canonical datasets
        # ------------------------------------------------------------------

        logger.info(
            "Reading canonical origination data: %s",
            origination_path,
        )

        origination_df = pd.read_parquet(origination_path)

        logger.info(
            "Reading canonical performance data: %s",
            performance_path,
        )

        performance_df = pd.read_parquet(performance_path)

        logger.info(
            "Canonical datasets loaded: origination_rows=%s " "performance_rows=%s",
            f"{len(origination_df):,}",
            f"{len(performance_df):,}",
        )

        # ------------------------------------------------------------------
        # Preprocess each canonical dataset exactly once
        # ------------------------------------------------------------------

        origination_features = build_origination(
            origination_df,
            config,
        )

        performance_features = build_performance(
            performance_df,
        )

        logger.info(
            "Feature preprocessing completed: "
            "origination_columns=%s performance_columns=%s",
            origination_features.shape[1],
            performance_features.shape[1],
        )

        # ------------------------------------------------------------------
        # Construct loan-month master dataset
        #
        # This dataset exists for target construction. Performance fields
        # must not be carried into the final modelling feature population.
        # ------------------------------------------------------------------

        master = build_master_dataset(
            origination_features,
            performance_features,
        )

        logger.info(
            "Master loan-month dataset constructed: rows=%s columns=%s",
            f"{len(master):,}",
            master.shape[1],
        )

        # ------------------------------------------------------------------
        # Construct loan-level target
        # ------------------------------------------------------------------

        target = build_24m_serious_delinquency_target(
            master,
            config,
        )

        logger.info(
            "Target construction completed: eligible_loans=%s events=%s",
            f"{len(target):,}",
            f"{int(target['ever_90dpd_24m'].sum()):,}",
        )

        # ------------------------------------------------------------------
        # Construct final modelling dataset
        #
        # IMPORTANT:
        # Model features come from origination_features, NOT master.
        # This prevents monthly performance information from leaking into
        # the origination-time model.
        # ------------------------------------------------------------------

        modelling = origination_features.merge(
            target,
            on="loan_id",
            how="inner",
            validate="one_to_one",
        ).reset_index(drop=True)

        modelling = apply_transformations(
            modelling,
            config,
        )

        modelling = build_interaction_features(
            modelling,
            config,
        )

        # ------------------------------------------------------------------
        # Resolve output path
        # ------------------------------------------------------------------

        model_input_path = create_path(
            config["catalog"]["base"],
            config["catalog"],
            "model_input_path",
            approach,
            provider,
            vintage,
            must_exist=False,
        )

        # ------------------------------------------------------------------
        # Persist modelling dataset
        # ------------------------------------------------------------------

        write_parquet(
            modelling,
            model_input_path,
        )

        logger.info(
            "Dataset construction completed: "
            "provider=%s vintage=%s rows=%s columns=%s events=%s "
            "path=%s duration_seconds=%.2f",
            provider,
            vintage,
            f"{len(modelling):,}",
            modelling.shape[1],
            f"{int(modelling['ever_90dpd_24m'].sum()):,}",
            model_input_path,
            perf_counter() - start,
        )


def build_modelling_dataset_behavioral(
    config: dict,
) -> None:
    """
    Build and persist the point-in-time behavioural modelling dataset.

    The behavioural modelling dataset is constructed at:

        loan_id x observation_age

    The dataset combines:
    - point-in-time behavioural features;
    - a forward-looking behavioural target.

    Future information is used only by the target construction and is
    not included in the point-in-time feature population.
    """

    data_config = config["parameters"]["data"]
    approach = config["parameters"]["modelling_approach"]

    if data_config["preprocess"]["skip"]:
        logger.info(
            "Preprocess skipped by configuration",
        )
        return

    start = perf_counter()

    for vintage in data_config["all_vintages"]:

        provider = data_config["data_provider"]

        logger.info(
            "Behavioral dataset construction started: " "provider=%s vintage=%s",
            provider,
            vintage,
        )

        # ------------------------------------------------------------------
        # Resolve canonical paths
        # ------------------------------------------------------------------

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

        # ------------------------------------------------------------------
        # Read canonical datasets
        # ------------------------------------------------------------------

        logger.info(
            "Reading canonical origination data: %s",
            origination_path,
        )

        origination_df = pd.read_parquet(
            origination_path,
        )

        logger.info(
            "Reading canonical performance data: %s",
            performance_path,
        )

        performance_df = pd.read_parquet(
            performance_path,
        )

        logger.info(
            "Canonical datasets loaded: " "origination_rows=%s performance_rows=%s",
            f"{len(origination_df):,}",
            f"{len(performance_df):,}",
        )

        # ------------------------------------------------------------------
        # Apply existing preprocessing contracts
        #
        # Reuse the existing V1 preprocessing for the common origination
        # and performance fields.
        # ------------------------------------------------------------------

        origination_features = build_origination(
            origination_df,
            config,
        )

        # first_payment_date is required only for the V2 lifecycle clock.
        # Keep this field V2-specific rather than changing the V1 feature
        # contract.
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

        logger.info(
            "Feature preprocessing completed: "
            "origination_columns=%s performance_columns=%s",
            origination_features.shape[1],
            performance_features.shape[1],
        )

        # ------------------------------------------------------------------
        # Build master loan-month dataset
        # ------------------------------------------------------------------

        master = build_master_dataset(
            origination_features,
            performance_features,
        )

        # ------------------------------------------------------------------
        # Build a consistent V2 lifecycle clock.
        #
        # calculated_loan_age is derived from:
        #
        #     period
        #     first_payment_date
        #
        # and is used by both behavioral features and behavioral target
        # construction.
        # ------------------------------------------------------------------

        master = behavioral.add_calculated_loan_age(
            master,
        )

        logger.info(
            "Master loan-month dataset constructed: " "rows=%s columns=%s",
            f"{len(master):,}",
            master.shape[1],
        )

        # ------------------------------------------------------------------
        # Build point-in-time behavioural features
        #
        # Features are restricted to information available at the
        # observation age.
        # ------------------------------------------------------------------

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

        logger.info(
            "Behavioral feature population constructed: "
            "rows=%s columns=%s unique_loans=%s observation_ages=%s",
            f"{len(behavioral_features):,}",
            behavioral_features.shape[1],
            f"{behavioral_features['loan_id'].nunique():,}",
            sorted(behavioral_features["observation_age"].unique().tolist()),
        )

        # ------------------------------------------------------------------
        # Build forward-looking behavioural target
        #
        # IMPORTANT:
        # The target builder uses future performance, but that information
        # is only used to construct the target and is never part of the
        # point-in-time feature population.
        # ------------------------------------------------------------------

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

        logger.info(
            "Behavioral target constructed: " "rows=%s events=%s",
            f"{len(target):,}",
            f"{int(target['future_90dpd_12m'].sum()):,}",
        )

        # ------------------------------------------------------------------
        # Validate target grain before joining
        # ------------------------------------------------------------------

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

        # ------------------------------------------------------------------
        # Join point-in-time features to forward-looking target
        #
        # One feature row should map to at most one target row.
        # Only target-eligible observations are retained.
        # ------------------------------------------------------------------

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

        # ------------------------------------------------------------------
        # Validate final modelling grain
        # ------------------------------------------------------------------

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

        # ------------------------------------------------------------------
        # Validate that the feature snapshot and observation age agree
        # with the calculated lifecycle clock.
        # ------------------------------------------------------------------

        if not (modelling["calculated_loan_age"] == modelling["observation_age"]).all():
            raise ValueError(
                "Behavioral modelling dataset contains rows where "
                "calculated_loan_age != observation_age.",
            )

        # ------------------------------------------------------------------
        # Log final target statistics
        # ------------------------------------------------------------------

        logger.info(
            "Behavioral modelling dataset constructed: "
            "rows=%s events=%s event_rate=%.6f",
            f"{len(modelling):,}",
            f"{int(modelling['future_90dpd_12m'].sum()):,}",
            modelling["future_90dpd_12m"].mean(),
        )

        # ------------------------------------------------------------------
        # Resolve behavioral output path
        # ------------------------------------------------------------------

        model_input_path = create_path(
            config["catalog"]["base"],
            config["catalog"],
            "model_input_path",
            approach,
            provider,
            vintage,
            must_exist=False,
        )

        # ------------------------------------------------------------------
        # Persist final V2 landmark modelling dataset
        # ------------------------------------------------------------------

        write_parquet(
            modelling,
            model_input_path,
        )

        logger.info(
            "Behavioral dataset construction completed: "
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


def build_modelling_dataset(
    config: dict,
) -> None:
    """
    Build the modelling dataset for the configured modelling approach.
    """

    approach = config["parameters"]["modelling_approach"]

    if approach == "origination":
        build_modelling_dataset_origination(config)

    elif approach == "behavioral":
        build_modelling_dataset_behavioral(config)

    else:
        raise ValueError(
            f"Unsupported modelling approach: {approach}. "
            "Expected 'origination' or 'behavioral'."
        )