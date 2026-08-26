from __future__ import annotations

import logging
from time import perf_counter

from pyspark.ml.classification import LogisticRegression
from pyspark.sql import DataFrame

logger = logging.getLogger(__name__)


def train_logistic_regression_spark(
    training_df: DataFrame,
    config: dict,
) -> LogisticRegression:
    """Train the configured Spark Logistic Regression model."""

    start = perf_counter()

    modelling_config = config["parameters"]["modelling"]
    lr_config = modelling_config["logistic_regression"]

    # --------------------------------------------------------------
    # Validate training population
    # --------------------------------------------------------------

    training_rows = training_df.count()

    if training_rows == 0:
        raise ValueError("Logistic Regression training population is empty.")

    # --------------------------------------------------------------
    # Training diagnostics
    # --------------------------------------------------------------

    event_rate = training_df.selectExpr("avg(label) AS event_rate").first()[
        "event_rate"
    ]

    logger.info(
        "Logistic Regression training started: " "rows=%s features=%s event_rate=%.6f",
        f"{training_rows:,}",
        training_df.select("features").first()["features"].size,
        event_rate,
    )

    # --------------------------------------------------------------
    # Spark Logistic Regression
    # --------------------------------------------------------------

    model = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        probabilityCol="probability",
        rawPredictionCol="rawPrediction",
        maxIter=lr_config["max_iter"],
        regParam=lr_config["reg_param"],
        elasticNetParam=lr_config["elastic_net_param"],
        fitIntercept=lr_config["fit_intercept"],
        standardization=lr_config.get(
            "standardization",
            True,
        ),
    )

    fitted_model = model.fit(
        training_df,
    )

    # --------------------------------------------------------------
    # Training completion
    # --------------------------------------------------------------

    logger.info(
        "Logistic Regression training completed: "
        "iterations=%s duration_seconds=%.2f",
        fitted_model.summary.totalIterations,
        perf_counter() - start,
    )

    return fitted_model
