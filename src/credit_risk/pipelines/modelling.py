from __future__ import annotations

import logging
from time import perf_counter

import numpy as np
from pyspark import StorageLevel

from credit_risk.utils.config import create_path
from credit_risk.data.writers import write_spark_parquet, write_parquet
from pyspark.sql import DataFrame, functions as F
from credit_risk.modelling.data import load_modelling_vintage_pandas
from credit_risk.modelling.data_spark import load_modelling_vintage_spark

from credit_risk.modelling.split import split_dataset_pandas
from credit_risk.modelling.split_spark import split_dataset_spark

from credit_risk.modelling.preprocessing import (
    split_features_target,
    build_preprocessor,
)

from credit_risk.modelling.preprocessing_spark import (
    split_features_target_spark,
    fit_preprocessor_spark,
    transform_with_preprocessor_spark,
)

from credit_risk.modelling.model import train_model_pandas
from credit_risk.modelling.artifacts import save_artifacts_pandas

from credit_risk.modelling.artifacts_spark import save_artifacts_spark

from credit_risk.modelling.model_spark import train_model_spark

logger = logging.getLogger(__name__)


def run_modelling_pipeline_pandas(config: dict) -> None:
    approach = config["parameters"]["modelling_approach"]
    modelling_config = config["parameters"]["modelling"]

    start = perf_counter()

    development_vintages = None
    if modelling_config["train_test_split"] == "random":
        development_vintages = modelling_config["vintages_train"]
    elif modelling_config["train_test_split"] == "yearly":
        development_vintages = (
            modelling_config["vintages_train"] + modelling_config["vintages_test"]
        )
    oot_vintages = modelling_config["vintages_oot"]

    development_df = load_modelling_vintage_pandas(config, development_vintages)
    oot_df = load_modelling_vintage_pandas(config, oot_vintages)
    train_df, test_df = split_dataset_pandas(development_df, config)

    train_path = create_path(
        config["catalog"]["base"],
        config["catalog"],
        "train_df",
        approach,
        must_exist=False,
    )
    test_path = create_path(
        config["catalog"]["base"],
        config["catalog"],
        "validation_df",
        approach,
        must_exist=False,
    )
    oot_path = create_path(
        config["catalog"]["base"],
        config["catalog"],
        "oot_df",
        approach,
        must_exist=False,
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
    model = train_model_pandas(X_train_transformed, y_train, config)
    save_artifacts_pandas(
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


# ================================================================
# SPARK
# ================================================================


# ================================================================
# SPARK
# ================================================================


def run_modelling_pipeline_pyspark(config: dict, spark) -> None:

    approach = config["parameters"]["modelling_approach"]
    modelling_config = config["parameters"]["modelling"]
    target = config["parameters"]["target"]["name"]

    start = perf_counter()

    # ------------------------------------------------------------
    # Determine vintages
    # ------------------------------------------------------------

    if modelling_config["train_test_split"] == "random":

        development_vintages = modelling_config["vintages_train"]

    elif modelling_config["train_test_split"] == "yearly":

        development_vintages = (
            modelling_config["vintages_train"] + modelling_config["vintages_test"]
        )

    else:
        raise ValueError(
            "Unsupported train_test_split: " f"{modelling_config['train_test_split']}"
        )

    oot_vintages = modelling_config["vintages_oot"]

    logger.info(
        "Loading development vintages: %s",
        ", ".join(map(str, development_vintages)),
    )

    logger.info(
        "Loading OOT vintages: %s",
        ", ".join(map(str, oot_vintages)),
    )

    # ------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------

    step_start = perf_counter()

    development_df = load_modelling_vintage_spark(
        spark,
        config,
        development_vintages,
    )

    oot_df = load_modelling_vintage_spark(
        spark,
        config,
        oot_vintages,
    )

    logger.info(
        "Modelling data loaded in %.2f seconds",
        perf_counter() - step_start,
    )

    # ------------------------------------------------------------
    # SPLIT
    # ------------------------------------------------------------

    step_start = perf_counter()

    train_df, test_df = split_dataset_spark(
        development_df,
        config,
    )

    # development_df is only a lineage reference.
    # It is not persisted, so there is nothing useful to unpersist.
    del development_df

    logger.info(
        "Train/validation split constructed in %.2f seconds",
        perf_counter() - step_start,
    )

    # ------------------------------------------------------------
    # MATERIALIZE COUNTS
    # ------------------------------------------------------------
    #
    # IMPORTANT:
    # Do NOT persist all three datasets.
    #
    # With 25M+ train rows, 15M+ validation rows and 6M+ OOT rows,
    # persisting all three can put enormous pressure on the local
    # Spark JVM.
    #
    # We deliberately accept recomputation here in exchange for
    # memory safety.
    # ------------------------------------------------------------

    step_start = perf_counter()

    train_count = train_df.count()
    test_count = test_df.count()
    oot_count = oot_df.count()

    logger.info(
        "Train=%s validation=%s OOT=%s materialized in %.2f seconds",
        f"{train_count:,}",
        f"{test_count:,}",
        f"{oot_count:,}",
        perf_counter() - step_start,
    )

    if train_count == 0:
        raise ValueError("Training dataset is empty.")

    if test_count == 0:
        raise ValueError("Validation dataset is empty.")

    if oot_count == 0:
        raise ValueError("OOT dataset is empty.")

    # ------------------------------------------------------------
    # ARTIFACT PATHS
    # ------------------------------------------------------------

    train_path = create_path(
        config["catalog"]["base"],
        config["catalog"],
        "train_df",
        approach,
        must_exist=False,
    )

    test_path = create_path(
        config["catalog"]["base"],
        config["catalog"],
        "validation_df",
        approach,
        must_exist=False,
    )

    oot_path = create_path(
        config["catalog"]["base"],
        config["catalog"],
        "oot_df",
        approach,
        must_exist=False,
    )

    # ------------------------------------------------------------
    # OPTIONAL ARTIFACT WRITES
    # ------------------------------------------------------------
    #
    # These writes are deliberately performed without caching.
    #
    # If these files already exist and the split has not changed,
    # this block should eventually be made configurable/skippable.
    # ------------------------------------------------------------

    write_spark_parquet(
        train_df,
        train_path,
    )

    write_spark_parquet(
        test_df,
        test_path,
    )

    write_spark_parquet(
        oot_df,
        oot_path,
    )

    # ------------------------------------------------------------
    # SPLIT FEATURES / TARGET
    # ------------------------------------------------------------

    step_start = perf_counter()

    X_train, y_train = split_features_target_spark(
        train_df,
        config,
    )

    X_validation, y_validation = split_features_target_spark(
        test_df,
        config,
    )

    logger.info(
        "Feature/target split completed in %.2f seconds",
        perf_counter() - step_start,
    )

    # ------------------------------------------------------------
    # RELEASE LARGE DATAFRAMES THAT ARE NO LONGER REQUIRED
    # ------------------------------------------------------------
    #
    # We no longer need test_df or oot_df during model training.
    #
    # They were written to disk above.
    # Removing Python references allows Spark to discard their
    # lineage when no longer needed.
    # ------------------------------------------------------------

    del test_df
    del oot_df
    del X_validation
    del y_validation

    # ------------------------------------------------------------
    # PREPROCESSOR FIT
    # ------------------------------------------------------------

    step_start = perf_counter()

    preprocessor_model = fit_preprocessor_spark(
        X_train,
        config,
    )

    logger.info(
        "Spark preprocessor fitted in %.2f seconds",
        perf_counter() - step_start,
    )

    # ------------------------------------------------------------
    # TRANSFORM TRAIN
    # ------------------------------------------------------------
    #
    # IMPORTANT:
    # Do NOT persist the transformed training matrix by default.
    #
    # The transformed dataset is potentially enormous, particularly
    # after one-hot encoding.
    #
    # The model should consume it directly.
    # ------------------------------------------------------------

    step_start = perf_counter()

    X_train_transformed = transform_with_preprocessor_spark(
        X_train,
        preprocessor_model,
        config,
    )

    logger.info(
        "Training matrix transformation constructed in %.2f seconds",
        perf_counter() - step_start,
    )

    # ------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------

    logger.info("Starting model training")

    model_start = perf_counter()

    model = train_model_spark(
        X_train_transformed,
        y_train,
        config,
    )

    logger.info(
        "Model training completed in %.2f seconds",
        perf_counter() - model_start,
    )

    # ------------------------------------------------------------
    # EVENT RATE
    # ------------------------------------------------------------

    event_rate = y_train.select(F.avg(F.col(target)).alias("event_rate")).first()[
        "event_rate"
    ]

    # ------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------

    save_start = perf_counter()

    save_artifacts_spark(
        model,
        preprocessor_model,
        {
            "training_rows": train_count,
            "training_features": len(X_train_transformed.columns),
            "event_rate": event_rate,
        },
        config,
    )

    logger.info(
        "Artifacts saved in %.2f seconds",
        perf_counter() - save_start,
    )

    # ------------------------------------------------------------
    # CLEANUP
    # ------------------------------------------------------------

    # No explicit unpersist() is required because we deliberately
    # did not persist the large modelling DataFrames.
    #
    # Remove references so Python/Spark can release lineage objects.

    del X_train_transformed
    del X_train
    del y_train

    logger.info(
        "Modelling pipeline completed: "
        "train_rows=%s validation_rows=%s "
        "duration_seconds=%.2f",
        f"{train_count:,}",
        f"{test_count:,}",
        perf_counter() - start,
    )


def run_modelling_pipeline(config: dict, spark=None) -> None:
    modelling_config = config["parameters"]["modelling"]

    if modelling_config["skip"]:
        logger.info("Modelling pipeline skipped by configuration")
        return
    development_vintages = None
    if modelling_config["train_test_split"] == "random":
        development_vintages = modelling_config["vintages_train"]
    elif modelling_config["train_test_split"] == "yearly":
        development_vintages = (
            modelling_config["vintages_train"] + modelling_config["vintages_test"]
        )
    oot_vintages = modelling_config["vintages_oot"]
    logger.info(
        "Vintages being considered in model training pipeline: %s",
        ", ".join(np.array(development_vintages).astype(str)),
    )
    logger.info(
        "Vintages being considered for OOT: %s",
        ", ".join(np.array(oot_vintages).astype(str)),
    )

    engine = config["parameters"]["engine"]

    if engine == "pandas":
        run_modelling_pipeline_pandas(config)

    elif engine == "pyspark":
        # raise NotImplementedError(
        #     "Pyspark Support is being implemented", "Please use Pandas for now"
        # )
        run_modelling_pipeline_pyspark(config, spark)

    else:
        raise ValueError(f"Unsupported modelling engine: {engine}")
