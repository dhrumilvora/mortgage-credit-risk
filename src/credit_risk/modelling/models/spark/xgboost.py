from __future__ import annotations

import logging
from time import perf_counter

from pyspark.sql import DataFrame
from xgboost.spark import SparkXGBClassifier

logger = logging.getLogger(__name__)


def train_xgboost_spark(
    training_df: DataFrame,
    config: dict,
) -> SparkXGBClassifier:
    """Train the configured Spark XGBoost model."""

    start = perf_counter()

    modelling_config = config["parameters"]["modelling"]
    xgb_config = modelling_config["xgboost"]
    training_df = training_df.repartition(xgb_config["n_jobs"])
    # --------------------------------------------------------------
    # Validate training population
    # --------------------------------------------------------------

    training_rows = training_df.count()

    if training_rows == 0:
        raise ValueError("XGBoost training population is empty.")

    # --------------------------------------------------------------
    # Training diagnostics
    # --------------------------------------------------------------

    event_rate = training_df.selectExpr("avg(label) AS event_rate").first()[
        "event_rate"
    ]

    feature_count = training_df.select("features").first()["features"].size

    logger.info(
        "XGBoost training started: " "rows=%s features=%s event_rate=%.6f",
        f"{training_rows:,}",
        feature_count,
        event_rate,
    )

    # --------------------------------------------------------------
    # Spark XGBoost
    # --------------------------------------------------------------

    model = SparkXGBClassifier(
        features_col="features",
        label_col="label",
        n_estimators=xgb_config["n_estimators"],
        max_depth=xgb_config["max_depth"],
        learning_rate=xgb_config["learning_rate"],
        subsample=xgb_config["subsample"],
        colsample_bytree=xgb_config["colsample_bytree"],
        min_child_weight=xgb_config["min_child_weight"],
        reg_alpha=xgb_config["reg_alpha"],
        reg_lambda=xgb_config["reg_lambda"],
        eval_metric=xgb_config["eval_metric"],
        num_workers=xgb_config["n_jobs"],
        scale_pos_weight=xgb_config["scale_pos_weight"],
        random_state=modelling_config["random_state"],
    )

    fitted_model = model.fit(
        training_df,
    )

    logger.info(
        "XGBoost training completed: " "trees=%s duration_seconds=%.2f",
        xgb_config["n_estimators"],
        perf_counter() - start,
    )

    return fitted_model
