"""Main entry point for the mortgage credit-risk pipeline."""

from __future__ import annotations

from pathlib import Path

from credit_risk.pipelines.data_preprocess import build_modeling_dataset
from credit_risk.pipelines.ingest import ingest
from credit_risk.pipelines.reporting import run_reporting_pipeline
from credit_risk.utils.config import read_config


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
    config = read_config(Path(project_path))
    ingest(config)
    build_modeling_dataset(config)
    run_reporting_pipeline(config)
