"""Preprocessing pipeline for mortgage credit-risk model construction."""

from __future__ import annotations

import logging
from time import perf_counter

import pandas as pd

from credit_risk.data.writers import write_parquet
from credit_risk.features import origination, performance
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

logger = logging.getLogger(__name__)


def build_origination(df: pd.DataFrame) -> pd.DataFrame:
    """Apply finalized origination preprocessing."""

    validate_baseline_features(df.columns)

    result = origination.select_baseline_features(df)
    result = origination.normalize_sentinel_values(result)
    result = origination.add_missing_indicators(result)

    return result


def build_performance(df: pd.DataFrame) -> pd.DataFrame:
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


def build_modeling_dataset(config: dict) -> None:
    """
    Build and persist the final loan-level modelling dataset.

    Performance history is used only for target construction.
    Final modelling features are sourced exclusively from the
    origination-time feature dataset.
    """

    data_config = config["parameters"]["data"]

    if data_config["preprocess"]["skip"]:
        logger.info("Preprocessing skipped by configuration")
        return

    start = perf_counter()

    provider = data_config["data_provider"]
    vintage = data_config["vintage"]

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

    logger.info("Reading canonical origination data: %s", origination_path)
    origination_df = pd.read_parquet(origination_path)

    logger.info("Reading canonical performance data: %s", performance_path)
    performance_df = pd.read_parquet(performance_path)

    logger.info(
        "Canonical datasets loaded: origination_rows=%s performance_rows=%s",
        f"{len(origination_df):,}",
        f"{len(performance_df):,}",
    )

    # ------------------------------------------------------------------
    # Preprocess each canonical dataset exactly once
    # ------------------------------------------------------------------

    origination_features = build_origination(
        origination_df,
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

    modeling = origination_features.merge(
        target,
        on="loan_id",
        how="inner",
        validate="one_to_one",
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Resolve output path
    # ------------------------------------------------------------------

    model_input_path = create_path(
        config["catalog"]["base"],
        config["catalog"],
        "model_input_path",
        provider,
        vintage,
        must_exist=False,
    )

    # ------------------------------------------------------------------
    # Persist modelling dataset
    # ------------------------------------------------------------------

    write_parquet(
        modeling,
        model_input_path,
    )

    logger.info(
        "Dataset construction completed: "
        "provider=%s vintage=%s rows=%s columns=%s events=%s "
        "path=%s duration_seconds=%.2f",
        provider,
        vintage,
        f"{len(modeling):,}",
        modeling.shape[1],
        f"{int(modeling['ever_90dpd_24m'].sum()):,}",
        model_input_path,
        perf_counter() - start,
    )
