"""Main entry point for the mortgage credit-risk pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

from credit_risk.utils.config import read_config
from credit_risk.utils.logging import configure_logging

from credit_risk.pipelines.data_preprocess import build_modelling_dataset
from credit_risk.pipelines.ingest import ingest
from credit_risk.pipelines.reporting import run_reporting_pipeline
from credit_risk.pipelines.modelling import run_modelling_pipeline
from credit_risk.pipelines.evaluation import run_evaluation_pipeline
from credit_risk.utils.spark import create_spark_session

logger = logging.getLogger(__name__)


def run_pipeline(project_path: str | Path) -> None:
    """
    Run the end-to-end mortgage credit-risk data pipeline.

    Pipeline
    --------
    1. Ingest Freddie Mac origination and performance data.
    2. Build the configured modelling dataset using the selected
       preprocessing engine.
    3. Run data-quality reporting.
    4. Train the configured model.
    5. Evaluate model performance.

    Parameters
    ----------
    project_path:
        Project root containing the configuration directory.

    Returns
    -------
    None
        Persists modelling datasets, reports, model artifacts,
        and evaluation results to the configured catalog paths.
    """

    project_root = Path(project_path)
    config = read_config(project_root)
    spark = None
    if config["parameters"]["engine"] == "pyspark":
        spark = create_spark_session(
            config,
        )
    logging_config = config["parameters"].get("logging", {})
    configure_logging(
        level=logging_config.get("level", "INFO"),
        enabled=logging_config.get("enabled", True),
        color=logging_config.get("color", True),
    )

    start = perf_counter()
    logger.info("━━ Pipeline started ━━ project=%s", project_root.resolve())
    ingest(config)
    build_modelling_dataset(config, spark)
    run_reporting_pipeline(config)
    run_modelling_pipeline(config, spark)
    run_evaluation_pipeline(config)
    logger.info(
        "━━ Pipeline completed ━━ duration_seconds=%.2f", perf_counter() - start
    )
