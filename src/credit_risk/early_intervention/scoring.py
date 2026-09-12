from __future__ import annotations

import logging

from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession

from credit_risk.evaluations.evaluations import generate_predictions
from credit_risk.modelling.artifacts_spark import (
    load_spark_model_artifacts,
)
from credit_risk.modelling.preprocessing_spark import (
    split_features_target_spark,
)

logger = logging.getLogger(__name__)


def score_early_intervention_population(
    df: DataFrame,
    config: dict,
    spark: SparkSession,
) -> DataFrame:
    """
    Score the persisted P1 behavioral modelling population.

    The frozen P1 Spark model is used exactly as it is used by the
    existing evaluation pipeline.

    For GAM:
        the model already contains its own transformation pipeline.

    For other model types:
        the fitted preprocessing pipeline is applied before scoring.
    """

    if df.limit(1).count() == 0:
        raise ValueError("Early-intervention scoring input is empty.")

    algorithm = config["parameters"]["modelling"]["algorithm"]

    pd_horizon = config["parameters"]["behavioral"]["prediction_horizon_months"]

    pd_column = f"p1_pd_{pd_horizon}m"

    # --------------------------------------------------------------
    # Load frozen P1 artifacts.
    # --------------------------------------------------------------

    model, preprocessor = load_spark_model_artifacts(
        config,
    )

    # --------------------------------------------------------------
    # Split model features from target.
    #
    # We already have the P1 modelling population, so the target is
    # not needed for inference.
    # --------------------------------------------------------------

    X, _ = split_features_target_spark(
        df,
        config,
    )

    # --------------------------------------------------------------
    # GAM contains its own complete model pipeline.
    # --------------------------------------------------------------

    if algorithm == "gam":
        model_input = X
    else:
        model_input = preprocessor.transform(X)

    predictions = model.transform(
        model_input,
    )

    if "probability" not in predictions.columns:
        raise ValueError("Frozen P1 model did not produce a probability column.")

    prediction_scores = predictions.select(
        "loan_id",
        "calculated_loan_age",
        vector_to_array("probability")[1].alias(pd_column),
    )

    # --------------------------------------------------------------
    # Join the P1 score back to the original P1 population.
    # --------------------------------------------------------------

    result = df.join(
        prediction_scores,
        on=[
            "loan_id",
            "calculated_loan_age",
        ],
        how="inner",
    )

    if result.limit(1).count() == 0:
        raise ValueError("Early-intervention scoring produced zero scored rows.")

    return result
