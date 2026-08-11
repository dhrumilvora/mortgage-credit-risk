from __future__ import annotations
import logging
from time import perf_counter
import numpy as np
from credit_risk.utils.config import create_path
from credit_risk.data.writers import write_parquet
from credit_risk.modelling.data import load_modelling_vintage
from credit_risk.modelling.split import stratified_data_split
from credit_risk.modelling.preprocessing import (
    split_features_target,
    build_preprocessor,
)
from credit_risk.modelling.model import train_model
from credit_risk.modelling.artifacts import save_artifacts

logger = logging.getLogger(__name__)


def run_modelling_pipeline(config: dict) -> None:

    modelling_config = config["parameters"]["modelling"]
    if modelling_config["skip"]:
        logger.info("Modelling pipeline skipped by configuration")
        return
    start = perf_counter()

    development_vintages = modelling_config["vintages_train"]
    oot_vintages = modelling_config["vintages_oot"]
    logger.info(
        "Vintages being considered in model training pipeline: %s",
        ", ".join(np.array(development_vintages).astype(str)),
    )
    logger.info(
        "Vintages being considered for OOT: %s",
        ", ".join(np.array(oot_vintages).astype(str)),
    )

    development_df = load_modelling_vintage(config, development_vintages)
    oot_df = load_modelling_vintage(config, oot_vintages)
    train_df, test_df = stratified_data_split(development_df, config)
    train_path = create_path(
        config["catalog"]["base"], config["catalog"], "train_df", must_exist=False
    )
    test_path = create_path(
        config["catalog"]["base"], config["catalog"], "validation_df", must_exist=False
    )
    oot_path = create_path(
        config["catalog"]["base"], config["catalog"], "oot_df", must_exist=False
    )

    write_parquet(train_df, train_path)
    write_parquet(test_df, test_path)
    write_parquet(oot_df, oot_path)

    X_train, y_train = split_features_target(
        train_df,
        config,
    )

    X_validation, y_validation = split_features_target(
        test_df,
        config,
    )
    preprocessor = build_preprocessor(config)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_validation_transformed = preprocessor.transform(X_validation)

    logger.info(
        "Model matrices prepared: "
        "X_train=%s y_train=%s "
        "X_validation=%s y_validation=%s",
        X_train.shape,
        y_train.shape,
        X_validation.shape,
        y_validation.shape,
    )
    model = train_model(X_train_transformed, y_train, config)
    save_artifacts(
        model,
        preprocessor,
        {
            "training_rows": X_train_transformed.shape[0],
            "validation_rows": X_validation_transformed.shape[0],
            "training_features": X_train_transformed.shape[1],
            "event_rate": y_train.mean(),
        },
        config,
    )
    logger.info(
        "Modelling pipeline completed: "
        "train_rows=%s validation_rows=%s duration_seconds=%.2f",
        f"{len(train_df):,}",
        f"{len(test_df):,}",
        perf_counter() - start,
    )
