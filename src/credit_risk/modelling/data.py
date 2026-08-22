from __future__ import annotations

import logging

import pandas as pd

from credit_risk.utils.config import create_path

logger = logging.getLogger(__name__)


def load_modelling_vintage(
    config: dict,
    vintages: list[int],
) -> pd.DataFrame:
    """
    Load modelling datasets for the configured modelling approach.

    Expected grain:
        origination -> loan_id
        behavioral  -> loan_id x observation_age
    """
    approach = config["parameters"]["modelling_approach"]

    if not vintages:
        raise ValueError("At least one modelling vintage must be provided.")

    data_config = config["parameters"]["data"]
    provider = data_config["data_provider"]

    frames = []

    for vintage in vintages:

        path = create_path(
            config["catalog"]["base"],
            config["catalog"],
            "model_input_path",
            approach,
            provider,
            vintage,
        )

        df = pd.read_parquet(
            path,
        )

        df["vintage"] = vintage

        frames.append(df)

    final_df = pd.concat(
        frames,
        ignore_index=True,
    )

    # ------------------------------------------------------------------
    # Validate modelling grain by approach.
    # ------------------------------------------------------------------

    if approach == "origination":

        duplicate_rows = final_df.duplicated(
            subset=["loan_id"],
        )

        if duplicate_rows.any():
            raise ValueError(
                "Duplicate loan_id values detected across "
                "origination modelling vintages: "
                f"{int(duplicate_rows.sum()):,}"
            )

    elif approach == "behavioral":

        required_columns = [
            "loan_id",
            "observation_age",
        ]

        missing_columns = sorted(set(required_columns) - set(final_df.columns))

        if missing_columns:
            raise ValueError(
                "Missing behavioral grain columns: " + ", ".join(missing_columns)
            )

        duplicate_rows = final_df.duplicated(
            subset=[
                "loan_id",
                "observation_age",
            ],
        )

        if duplicate_rows.any():
            raise ValueError(
                "Duplicate loan_id x observation_age values detected "
                "across behavioral modelling vintages: "
                f"{int(duplicate_rows.sum()):,}"
            )

    else:
        raise ValueError(f"Unsupported modelling approach: {approach}")

    return final_df
