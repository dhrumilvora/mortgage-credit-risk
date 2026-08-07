from __future__ import annotations
import logging
from time import perf_counter
import pandas as pd

from credit_risk.modelling.data import load_modelling_vintage
from credit_risk.modelling.split import stratified_data_split
from credit_risk.utils.config import create_path
from credit_risk.data.writers import write_parquet

logger = logging.getLogger(__name__)


def run_modelling_pipeline(config: dict) -> None:

    modelling_config = config["parameters"]["modelling"]
    if modelling_config["skip"]:
        logger.info("Modelling pipeline skipped by configuration")
        return
    start = perf_counter()
    development_vintages = modelling_config["vintages_train"]
    development_df = load_modelling_vintage(config, development_vintages)
    train_df, test_df = stratified_data_split(development_df, config)
    train_path = create_path(
        config["catalog"]["base"], config["catalog"], "train_df", must_exist=False
    )
    test_path = create_path(
        config["catalog"]["base"], config["catalog"], "validation_df", must_exist=False
    )

    write_parquet(train_df, train_path)
    write_parquet(test_df, test_path)

    logger.info(
        "Modelling pipeline completed: "
        "train_rows=%s validation_rows=%s duration_seconds=%.2f",
        f"{len(train_df):,}",
        f"{len(test_df):,}",
        perf_counter() - start,
    )
