from __future__ import annotations

import logging
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from credit_risk.utils.config import create_path

from credit_risk.early_intervention.flagger import (
    build_early_intervention_flags,
)
from credit_risk.early_intervention.ranker import (
    rank_early_intervention_population,
)
from credit_risk.early_intervention.scoring import (
    score_early_intervention_population,
)

logger = logging.getLogger(__name__)


def run_early_intervention_pipeline(
    config: dict,
    spark: SparkSession,
) -> None:
    """
    Run the complete early-intervention scoring system.

    For every configured P1 vintage:

        persisted P1 modelling population
            ↓
        frozen P1 model
            ↓
        3-month PD
            ↓
        intervention flag
            ↓
        intervention priority ranking
            ↓
        persisted output
    """

    early_intervention_config = config["parameters"].get(
        "early_intervention",
        {},
    )

    if early_intervention_config.get(
        "skip",
        False,
    ):
        logger.info("Early-intervention pipeline skipped by configuration.")
        return

    data_config = config["parameters"]["data"]

    approach = config["parameters"]["modelling_approach"]

    provider = data_config["data_provider"]

    vintages = data_config["all_vintages"]

    threshold = early_intervention_config.get(
        "threshold",
        config["parameters"]["evaluation"]["classification"]["threshold"],
    )

    pd_horizon = config["parameters"]["behavioral"]["prediction_horizon_months"]

    pd_column = f"p1_pd_{pd_horizon}m"

    for vintage in vintages:

        logger.info(
            "Starting early-intervention scoring: " "provider=%s vintage=%s",
            provider,
            vintage,
        )

        # ----------------------------------------------------------
        # Load the existing P1 modelling population.
        # ----------------------------------------------------------

        model_input_path = create_path(
            config["catalog"]["base"],
            config["catalog"],
            "model_input_path",
            approach,
            provider,
            vintage,
            must_exist=True,
        )

        logger.info(
            "Loading P1 modelling population: %s",
            model_input_path,
        )

        model_input = spark.read.parquet(
            str(model_input_path),
        )

        if model_input.limit(1).count() == 0:
            logger.warning(
                "P1 modelling population is empty: " "vintage=%s",
                vintage,
            )
            continue

        # ----------------------------------------------------------
        # Score using frozen P1 artifact.
        # ----------------------------------------------------------

        scored = score_early_intervention_population(
            model_input,
            config,
            spark,
        )

        # ----------------------------------------------------------
        # Add vintage metadata.
        # ----------------------------------------------------------

        scored = scored.withColumn(
            "vintage",
            scored["loan_id"] * 0 + vintage,
        )

        # ----------------------------------------------------------
        # Build intervention flag.
        # ----------------------------------------------------------

        flagged = build_early_intervention_flags(
            scored,
            threshold=threshold,
            pd_column=pd_column,
        )

        # ----------------------------------------------------------
        # Build intervention priority ranking.
        # ----------------------------------------------------------

        ranked = rank_early_intervention_population(
            flagged,
            pd_column=pd_column,
        )

        # ----------------------------------------------------------
        # Persist.
        # ----------------------------------------------------------

        output_path = create_path(
            config["catalog"]["base"],
            config["catalog"],
            "early_intervention_output",
            approach,
            provider,
            vintage,
            must_exist=False,
        )

        ranked.write.mode("overwrite").parquet(str(output_path))

        logger.info(
            "Early-intervention output saved: " "provider=%s vintage=%s path=%s",
            provider,
            vintage,
            output_path,
        )

    logger.info("Early-intervention pipeline completed.")
