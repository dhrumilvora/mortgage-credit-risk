"""Development dataset splitting utilities."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def stratified_data_split(
    df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split development data into training and validation populations."""

    target_config = config["parameters"]["target"]
    modelling_config = config["parameters"]["modelling"]

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

    logger.info(
        "Development split started: "
        "rows=%s validation_size=%.2f "
        "target=%s event_rate=%.6f stratify=%s",
        f"{len(df):,}",
        validation_size,
        target,
        df[target].mean(),
        stratify,
    )

    stratify_values = df[target] if stratify else None

    train_df, validation_df = train_test_split(
        df,
        test_size=validation_size,
        random_state=random_state,
        stratify=stratify_values,
    )

    train_df = train_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(drop=True)

    logger.info(
        "Development split completed: "
        "train_rows=%s validation_rows=%s "
        "train_event_rate=%.6f validation_event_rate=%.6f",
        f"{len(train_df):,}",
        f"{len(validation_df):,}",
        train_df[target].mean(),
        validation_df[target].mean(),
    )

    return train_df, validation_df
