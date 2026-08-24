"""Development dataset splitting utilities."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit,
    train_test_split,
)

logger = logging.getLogger(__name__)


def stratified_data_split(
    df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split development data into training and validation populations."""

    target_config = config["parameters"]["target"]
    modelling_config = config["parameters"]["modelling"]
    approach = config["parameters"]["modelling_approach"]

    target = target_config["name"]
    validation_size = modelling_config["validation_size"]
    random_state = modelling_config["random_state"]
    stratify = modelling_config["stratify"]

    if target not in df.columns:
        raise ValueError(f"Target column not found in modelling dataset: {target}")

    if df[target].isna().any():
        raise ValueError(f"Target column contains missing values: {target}")

    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")

    if df[target].nunique() < 2:
        raise ValueError(f"Target must contain at least two classes: {target}")

    # ------------------------------------------------------------------
    # Origination approach
    #
    # Preserve the existing row-level stratified split exactly.
    # ------------------------------------------------------------------

    if approach == "origination":

        stratify_values = df[target] if stratify else None

        train_df, validation_df = train_test_split(
            df,
            test_size=validation_size,
            random_state=random_state,
            stratify=stratify_values,
        )

    # ------------------------------------------------------------------
    # Behavioral approach
    #
    # Keep all observations belonging to the same loan in the same
    # partition.
    # ------------------------------------------------------------------

    elif approach == "behavioral":

        required_columns = [
            "loan_id",
            "observation_age",
        ]

        missing_columns = sorted(set(required_columns) - set(df.columns))

        if missing_columns:
            raise ValueError(
                "Missing behavioral split columns: " + ", ".join(missing_columns)
            )

        if df["loan_id"].isna().any():
            raise ValueError("loan_id contains missing values.")

        group_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=validation_size,
            random_state=random_state,
        )

        train_indices, validation_indices = next(
            group_splitter.split(
                df,
                groups=df["loan_id"],
            )
        )

        train_df = df.iloc[train_indices].copy()

        validation_df = df.iloc[validation_indices].copy()

    else:
        raise ValueError(f"Unsupported modelling approach: {approach}")

    # ------------------------------------------------------------------
    # Final validation
    # ------------------------------------------------------------------

    train_df = train_df.reset_index(drop=True)

    validation_df = validation_df.reset_index(drop=True)

    # For behavioral modelling, explicitly verify that no loan appears
    # in both populations.
    if approach == "behavioral":

        train_loans = set(train_df["loan_id"])

        validation_loans = set(validation_df["loan_id"])

        overlap = train_loans.intersection(validation_loans)

        if overlap:
            raise ValueError(
                "Loan leakage detected between training and validation "
                f"sets: {len(overlap):,} overlapping loans."
            )

    return train_df, validation_df


def yearly_data_split(
    df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split modelling data using chronological vintage boundaries.

    Training observations come from the configured training vintages.
    Validation observations come from the configured validation vintages.

    For behavioral modelling, all observations for a given loan must
    remain within the same population.
    """

    modelling_config = config["parameters"]["modelling"]
    approach = config["parameters"]["modelling_approach"]

    train_vintages = modelling_config["vintages_train"]
    validation_vintages = modelling_config["vintages_test"]

    if not train_vintages:
        raise ValueError("At least one training vintage must be provided.")

    if not validation_vintages:
        raise ValueError("At least one validation vintage must be provided.")

    if "vintage" not in df.columns:
        raise ValueError("Vintage column not found in modelling dataset.")

    train_vintages = sorted(set(train_vintages))

    validation_vintages = sorted(set(validation_vintages))

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

    # --------------------------------------------------------------
    # Behavioral split
    #
    # Validate the loan identifier and ensure that observations for
    # a loan cannot be split across train and validation.
    # --------------------------------------------------------------

    if approach == "behavioral":

        required_columns = [
            "loan_id",
            "observation_age",
        ]

        missing_columns = sorted(set(required_columns) - set(df.columns))

        if missing_columns:
            raise ValueError(
                "Missing behavioral split columns: " + ", ".join(missing_columns)
            )

        if df["loan_id"].isna().any():
            raise ValueError("loan_id contains missing values.")

    elif approach != "origination":
        raise ValueError(f"Unsupported modelling approach: {approach}")

    # --------------------------------------------------------------
    # Create chronological populations
    # --------------------------------------------------------------

    train_mask = df["vintage"].isin(train_vintages)

    validation_mask = df["vintage"].isin(validation_vintages)

    train_df = df.loc[train_mask].copy()

    validation_df = df.loc[validation_mask].copy()

    # --------------------------------------------------------------
    # Validate that both populations contain data
    # --------------------------------------------------------------

    if train_df.empty:
        raise ValueError(
            "Training population is empty for configured vintages: "
            + ", ".join(
                map(
                    str,
                    train_vintages,
                )
            )
        )

    if validation_df.empty:
        raise ValueError(
            "Validation population is empty for configured vintages: "
            + ", ".join(
                map(
                    str,
                    validation_vintages,
                )
            )
        )

    # --------------------------------------------------------------
    # Behavioral leakage validation
    # --------------------------------------------------------------

    if approach == "behavioral":

        train_loans = set(train_df["loan_id"])

        validation_loans = set(validation_df["loan_id"])

        overlap = train_loans.intersection(validation_loans)

        if overlap:
            raise ValueError(
                "Loan leakage detected between chronological "
                "training and validation populations: "
                f"{len(overlap):,} overlapping loans."
            )

    # --------------------------------------------------------------
    # Reset indices
    # --------------------------------------------------------------

    train_df = train_df.reset_index(drop=True)

    validation_df = validation_df.reset_index(drop=True)

    logger.info(
        "Chronological data split completed: "
        "train_vintages=%s validation_vintages=%s "
        "train_rows=%s validation_rows=%s",
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
        f"{len(train_df):,}",
        f"{len(validation_df):,}",
    )

    return train_df, validation_df


def split_dataset(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config["parameters"]["modelling"]["train_test_split"] == "random":
        return stratified_data_split(df, config)
    elif config["parameters"]["modelling"]["train_test_split"] == "yearly":
        return yearly_data_split(df, config)
    raise ValueError(
        f"Unsupported Train Test Split: {config["parameters"]["modelling"]["train_test_split"]}"
    )
