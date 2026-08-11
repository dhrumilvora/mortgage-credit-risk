"""Reporting pipeline for mortgage credit-risk data quality."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

from credit_risk.reporting.data_quality import (
    build_data_quality_report,
    load_reporting_data,
)
from credit_risk.reporting.excel import write_excel_report
from credit_risk.utils.config import create_path
from credit_risk.reporting.distribution import build_numerical_distribution_report

logger = logging.getLogger(__name__)


def run_reporting_pipeline(
    config: dict,
) -> Path | None:
    """
    Build and persist the data-quality reporting workbook.

    Reporting is generated from persisted pipeline outputs and therefore
    can be run independently of ingestion and preprocessing.
    """

    parameters = config["parameters"]
    if parameters.get("reporting", {}).get("skip", False):
        logger.info("Data-quality reporting skipped by configuration")
        return None

    catalog = config["catalog"]
    start = perf_counter()
    logger.info(
        "Data-quality reporting started: vintage=%s", parameters["reporting"]["vintage"]
    )

    data = load_reporting_data(config)

    # Build all QC/reporting tables.
    report = build_data_quality_report(data, config)

    # Resolve reporting output path.
    report_path = create_path(
        catalog["base"],
        catalog,
        "pipeline_qc",
        parameters["reporting"]["vintage"],
        must_exist=False,
    )

    # Write workbook.
    write_excel_report(
        reports=report,
        output_path=report_path,
    )

    numerical_report_path = create_path(
        catalog["base"],
        catalog,
        "numerical_distribution_report",
        parameters["reporting"]["vintage"],
        must_exist=False,
    )

    build_numerical_distribution_report(
        df=data["model_input"],
        output_path=numerical_report_path,
        exclude={
            parameters["data"]["id_col"],
            parameters["target"]["name"],
        },
    )
    logger.info(
        "Reporting completed: worksheets=%s excel_path=%s numerical_pdf=%s "
        "duration_seconds=%.2f",
        len(report),
        report_path,
        numerical_report_path,
        perf_counter() - start,
    )

    return report_path
