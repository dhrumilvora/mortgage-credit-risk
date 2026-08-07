from __future__ import annotations
import logging
import pandas as pd
from credit_risk.utils.config import create_path

logger = logging.getLogger(__name__)


def load_modelling_vintage(config: dict, vintages: list[int]) -> pd.DataFrame:
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
            provider,
            vintage,
        )
        logger.info(
            "Loading model input: provider=%s vintage=%s path=%s",
            provider,
            vintage,
            path,
        )
        df = pd.read_parquet(path)
        df["vintage"] = vintage
        frames.append(df)
    final_df = pd.concat(frames, ignore_index=True)
    duplicate_loans = final_df["loan_id"].duplicated()

    if duplicate_loans.any():
        raise ValueError(
            "Duplicate loan_id values detected across modelling vintages: "
            f"{int(duplicate_loans.sum()):,}"
        )
    logger.info(
        "Modelling vintages loaded: vintages=%s rows=%s columns=%s",
        vintages,
        f"{len(final_df):,}",
        final_df.shape[1],
    )

    return final_df
