from __future__ import annotations

import logging
from time import perf_counter

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from xgboost.spark import SparkXGBClassifier

logger = logging.getLogger(__name__)


def train_xgboost_spark(
    training_df: DataFrame,
    config: dict,
) -> SparkXGBClassifier:
    """Train the configured Spark XGBoost model on large-scale panel data."""

    start = perf_counter()

    modelling_config = config["parameters"]["modelling"]
    xgb_config = modelling_config["xgboost"]

    # --------------------------------------------------------------
    # Repartitioning: Use explicit target partition count or keep existing cluster partitions
    # DO NOT repartition to n_jobs/num_workers (e.g., 4), which causes severe memory spills.
    # --------------------------------------------------------------
    num_partitions = xgb_config.get("num_partitions", 200)
    training_df = training_df.repartition(num_partitions)

    # --------------------------------------------------------------
    # Efficient Single-Pass Training Diagnostics
    # Evaluates row count and event rate in a single Spark action
    # --------------------------------------------------------------
    stats = training_df.select(
        F.count("*").alias("total_rows"),
        F.avg(F.col("label").cast("double")).alias("event_rate"),
    ).collect()[0]

    training_rows = stats["total_rows"]
    event_rate = stats["event_rate"] or 0.0

    if training_rows == 0:
        raise ValueError("XGBoost training population is empty.")

    # Retrieve feature vector length without triggering full scan
    sample_row = training_df.select("features").first()
    feature_count = sample_row["features"].size if sample_row else 0

    logger.info(
        "XGBoost training started: rows=%s features=%s event_rate=%.6f",
        f"{training_rows:,}",
        feature_count,
        event_rate,
    )

    # --------------------------------------------------------------
    # Spark XGBoost Estimator Initializer
    # --------------------------------------------------------------
    model = SparkXGBClassifier(
        features_col="features",
        label_col="label",
        # Speed & Scaling Engine
        tree_method=xgb_config.get("tree_method", "hist"),  # Essential for 25M rows
        num_workers=xgb_config["n_jobs"],  # Number of Spark parallel worker tasks
        # Model Parameters
        n_estimators=xgb_config["n_estimators"],
        max_depth=xgb_config["max_depth"],
        learning_rate=xgb_config["learning_rate"],
        subsample=xgb_config["subsample"],
        colsample_bytree=xgb_config["colsample_bytree"],
        min_child_weight=xgb_config["min_child_weight"],
        reg_alpha=xgb_config["reg_alpha"],
        reg_lambda=xgb_config["reg_lambda"],
        eval_metric=xgb_config["eval_metric"],
        scale_pos_weight=xgb_config["scale_pos_weight"],
        random_state=modelling_config["random_state"],
    )

    fitted_model = model.fit(training_df)

    logger.info(
        "XGBoost training completed: trees=%s duration_seconds=%.2f",
        xgb_config["n_estimators"],
        perf_counter() - start,
    )

    return fitted_model
