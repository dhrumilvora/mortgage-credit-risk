from __future__ import annotations
import logging
from time import perf_counter
import numpy as np
from credit_risk.utils.config import create_path
from credit_risk.data.writers import write_parquet

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
from credit_risk.data.writers import write_spark_parquet
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


def run_modelling_pipeline_pyspark(config: dict, spark) -> None:
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

    development_df = load_modelling_vintage_spark(spark, config, development_vintages)
    oot_df = load_modelling_vintage_spark(spark, config, oot_vintages)
    train_df, test_df = split_dataset_spark(development_df, config)

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

    write_spark_parquet(train_df, train_path)
    write_spark_parquet(test_df, test_path)
    write_spark_parquet(oot_df, oot_path)
    target = config["parameters"]["target"]["name"]
    X_train, y_train = split_features_target_spark(
        train_df,
        config,
    )

    preprocessor_model = fit_preprocessor_spark(
        X_train,
        config,
    )
    X_train_transformed = transform_with_preprocessor_spark(
        X_train, preprocessor_model, config
    )

    X_train_transformed = X_train_transformed.cache()

    logger.info(
        "Spark model matrices prepared: " "train_columns=%s",
        X_train_transformed.columns,
    )
    model = train_model_spark(X_train_transformed, y_train, config)
    save_artifacts_spark(
        model,
        preprocessor_model,
        {
            "training_rows": X_train_transformed.count(),
            "training_features": len(X_train_transformed.columns),
            "event_rate": y_train.selectExpr(f"avg({target}) AS event_rate").first()[
                "event_rate"
            ],
        },
        config,
    )
    logger.info(
        "Modelling pipeline completed: "
        "train_rows=%s validation_rows=%s duration_seconds=%.2f",
        f"{train_df.count():,}",
        f"{test_df.count():,}",
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
