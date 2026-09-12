from __future__ import annotations

import logging

from pyspark.sql import DataFrame, SparkSession

from portfolio_risk.model.artifacts import load_spark_model_artifacts
from portfolio_risk.model.scoring import score_loans
from portfolio_risk.ranking.ranker import build_risk_ranking
from portfolio_risk.utils.config import create_path

logger = logging.getLogger(__name__)


def run_risk_scoring_pipeline(
    config: dict,
    spark: SparkSession,
) -> None:
    """
    Score and rank all configured portfolio snapshots.

    All configured vintages are combined before model scoring so that
    risk ranking is performed on a unified portfolio population.

    The resulting scored and ranked portfolio is persisted and is not
    returned.
    """

    risk_config = config["parameters"]["risk"]

    if risk_config["skip"]:
        logger.info("Risk scoring pipeline skipped by configuration.")
        return
    logger.info("Risk Scoring Pipeline Started")
    snapshot_config = config["parameters"]["snapshot"]
    vintages = snapshot_config["vintages"]

    provider = config["parameters"]["snapshot"]["data_provider"]

    logger.info(
        "Starting risk scoring pipeline: provider=%s vintages=%s",
        provider,
        vintages,
    )

    # --------------------------------------------------------------
    # Load all configured snapshot vintages
    # --------------------------------------------------------------

    snapshot_dfs: list[DataFrame] = []

    for vintage in vintages:
        snapshot_path = create_path(
            config["catalog"],
            "snapshot_path",
            provider,
            vintage,
            must_exist=True,
        )

        logger.info(
            "Loading portfolio snapshot: vintage=%s path=%s",
            vintage,
            snapshot_path,
        )

        snapshot_df = spark.read.parquet(str(snapshot_path))

        if snapshot_df.limit(1).count() == 0:
            raise ValueError(f"Snapshot DataFrame is empty for vintage {vintage}.")

        snapshot_dfs.append(snapshot_df)

    if not snapshot_dfs:
        raise ValueError("No portfolio snapshots were configured.")

    # --------------------------------------------------------------
    # Unified portfolio population
    # --------------------------------------------------------------

    snapshots = snapshot_dfs[0]

    for snapshot_df in snapshot_dfs[1:]:
        snapshots = snapshots.unionByName(snapshot_df)

    logger.info(
        "Unified portfolio snapshots loaded: vintages=%s rows=%s",
        len(vintages),
        f"{snapshots.count():,}",
    )

    # --------------------------------------------------------------
    # Load P1 model artifacts
    # --------------------------------------------------------------

    model, preprocessor = load_spark_model_artifacts(
        config,
    )

    # --------------------------------------------------------------
    # Score unified portfolio
    # --------------------------------------------------------------

    scored = score_loans(
        config=config,
        df=snapshots,
        model=model,
        preprocessor=preprocessor,
    )

    pd_horizon = risk_config["pd_horizon_months"]
    score_column = f"p1_pd_{pd_horizon}m"

    if score_column not in scored.columns:
        raise ValueError(
            f"Expected scoring output column '{score_column}' " "was not produced."
        )

    # --------------------------------------------------------------
    # Unified portfolio ranking
    # --------------------------------------------------------------

    ranked = build_risk_ranking(
        df=scored,
        score_column=score_column,
    )

    # --------------------------------------------------------------
    # Persist final risk output
    # --------------------------------------------------------------

    output_path = create_path(
        config["catalog"],
        "risk_output_path",
        provider,
        must_exist=False,
    )

    ranked.write.mode("overwrite").parquet(
        str(output_path),
    )

    logger.info(
        "Risk scoring pipeline completed: " "rows=%s output_path=%s",
        f"{ranked.count():,}",
        output_path,
    )
