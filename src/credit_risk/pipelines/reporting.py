"""Reporting pipeline for mortgage credit-risk data quality."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

from credit_risk.reporting.data_quality import build_data_quality_report
from credit_risk.reporting.excel import write_excel_report
from credit_risk.utils.config import create_path

logger = logging.getLogger(__name__)


def run_reporting_pipeline(
    config: dict,
) -> Path:
    """
    Build and persist the data-quality reporting workbook.

    Reporting is generated from persisted pipeline outputs and therefore
    can be run independently of ingestion and preprocessing.
    """

    parameters = config["parameters"]
    catalog = config["catalog"]
    start = perf_counter()
    logger.info("Data-quality reporting started: vintage=%s", parameters["data"]["vintage"])

    # Build all QC/reporting tables.
    report = build_data_quality_report(config)

    # Resolve reporting output path.
    report_path = create_path(
        base_path=catalog["base"],
        catalog=catalog,
        key="pipeline_qc",
        year=parameters["data"]["vintage"],
        must_exist=False,
    )

    # Write workbook.
    write_excel_report(
        reports=report,
        output_path=report_path,
    )

    logger.info(
        "Data-quality reporting completed: worksheets=%s path=%s duration_seconds=%.2f",
        len(report),
        report_path,
        perf_counter() - start,
    )

    return report_path
