from __future__ import annotations

import logging
from time import perf_counter

from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import DataFrame

logger = logging.getLogger(__name__)


def train_random_forest_spark(
    training_df: DataFrame,
    config: dict,
) -> RandomForestClassifier:
    """Train the configured Spark Random Forest model."""

    start = perf_counter()

    modelling_config = config["parameters"]["modelling"]
    rf_config = modelling_config["random_forest"]

    training_rows = training_df.count()

    if training_rows == 0:
        raise ValueError("Random Forest training population is empty.")

    event_rate = training_df.selectExpr("avg(label) AS event_rate").first()[
        "event_rate"
    ]

    logger.info(
        "Random Forest training started: " "rows=%s features=%s event_rate=%.6f",
        f"{training_rows:,}",
        training_df.select("features").first()["features"].size,
        event_rate,
    )

    model = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        probabilityCol="probability",
        rawPredictionCol="rawPrediction",
        numTrees=rf_config["n_estimators"],
        maxDepth=rf_config["max_depth"],
        minInstancesPerNode=rf_config["min_samples_leaf"],
        featureSubsetStrategy=rf_config["max_features"],
        bootstrap=rf_config["bootstrap"],
        seed=rf_config["random_state"],
    )

    fitted_model = model.fit(
        training_df,
    )

    logger.info(
        "Random Forest training completed: " "trees=%s duration_seconds=%.2f",
        rf_config["n_estimators"],
        perf_counter() - start,
    )

    return fitted_model
