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

logger = logging.getLogger(__name__)


def run_pipeline(project_path: str | Path) -> None:
    """
    Run the end-to-end mortgage credit-risk data pipeline.

    Pipeline
    --------
    1. Ingest Freddie Mac origination and performance data.
    2. Preprocess origination data.
    3. Preprocess performance data.
    4. Merge into the master loan-month dataset.
    5. Construct the 24-month serious-delinquency target.
    6. Build the final loan-level modelling dataset.
    7. Train Model based on Config.
    8. Evaluate The model Performance
    Parameters
    ----------
    project_path:
        Project root containing the ``config`` directory.

    Returns
    -------
    None
        Persists the modelling dataset and data-quality workbook to the
        configured catalog paths.
    """
    project_root = Path(project_path)
    config = read_config(project_root)
    logging_config = config["parameters"].get("logging", {})
    configure_logging(
        level=logging_config.get("level", "INFO"),
        enabled=logging_config.get("enabled", True),
        color=logging_config.get("color", True),
    )

    start = perf_counter()
    logger.info("━━ Pipeline started ━━ project=%s", project_root.resolve())
    ingest(config)
    build_modelling_dataset(config)
    run_reporting_pipeline(config)
    run_modelling_pipeline(config)
    run_evaluation_pipeline(config)
    logger.info(
        "━━ Pipeline completed ━━ duration_seconds=%.2f", perf_counter() - start
    )
