"""Main entry point for the mortgage credit-risk pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from credit_risk.pipelines.ingest import ingest
from credit_risk.pipelines.data_preprocess import build_modeling_dataset


def run_pipeline(year: int) -> pd.DataFrame:
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
    origination_path:
        Path to the Freddie Mac origination file.

    performance_path:
        Path to the Freddie Mac monthly performance file.

    Returns
    -------
    pd.DataFrame
        Final loan-level modelling dataset containing origination
        features and the target ``ever_90dpd_24m``.
    """

    origination, performance = ingest(
        year, Path("data/01_raw/freddie_mac"), Path("data/02_interim")
    )

    modeling_df = build_modeling_dataset(
        origination=origination,
        performance=performance,
    )

    return modeling_df
