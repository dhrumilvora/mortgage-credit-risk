"""Reporting pipeline for mortgage credit-risk data quality."""

from __future__ import annotations

from pathlib import Path

from credit_risk.reporting.data_quality import build_data_quality_report
from credit_risk.reporting.excel import write_excel_report
from credit_risk.utils.config import create_path


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

    return report_path
