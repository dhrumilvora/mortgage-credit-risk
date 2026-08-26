from __future__ import annotations
import logging
from pyspark.sql import DataFrame, functions as F
from credit_risk.utils.config import create_path

logger = logging.getLogger(__name__)


def load_modelling_vintage_spark(spark, config: dict, vintages: list[int]) -> DataFrame:
    """
    Load Spark modelling datasets for the configured modelling approach.

    Expected grain:
        origination -> loan_id
        behavioral  -> loan_id x observation_age

    Returns:
        One Spark DataFrame containing all requested vintages.

    Notes:
        - No conversion to Pandas.
        - No collect().
        - Vintages are combined lazily using unionByName().
    """

    approach = config["parameters"]["modelling_approach"]
    if not vintages:
        raise ValueError("At least one modelling vintage must be provided.")

    if approach not in {
        "origination",
        "behavioral",
    }:
        raise ValueError(f"Unsupported modelling approach: {approach}")

    data_config = config["parameters"]["data"]
    provider = data_config["data_provider"]

    frames: list[DataFrame] = []

    for vintage in vintages:

        path = create_path(
            config["catalog"]["base"],
            config["catalog"],
            "model_input_path",
            approach,
            provider,
            vintage,
        )

        logger.info(
            "Spark reading modelling dataset: " "approach=%s vintage=%s path=%s",
            approach,
            vintage,
            path,
        )

        df = spark.read.parquet(str(path)).withColumn(
            "vintage",
            F.lit(vintage).cast("int"),
        )

        frames.append(df)

    # --------------------------------------------------------------
    # Combine vintages lazily.
    # --------------------------------------------------------------

    final_df = frames[0]

    for df in frames[1:]:
        final_df = final_df.unionByName(
            df,
        )

    # --------------------------------------------------------------
    # Validate modelling grain.
    #
    # Only a one-row existence check is collected to Python.
    # The full dataset is never collected.
    # --------------------------------------------------------------

    if approach == "origination":

        duplicate_rows = (
            final_df.groupBy("loan_id")
            .count()
            .filter(
                F.col("count") > 1,
            )
            .limit(1)
        )

        if duplicate_rows.count() > 0:
            raise ValueError(
                "Duplicate loan_id values detected across "
                "origination modelling vintages."
            )

    elif approach == "behavioral":

        required_columns = {
            "loan_id",
            "observation_age",
        }

        missing_columns = sorted(required_columns - set(final_df.columns))

        if missing_columns:
            raise ValueError(
                "Missing behavioral grain columns: " + ", ".join(missing_columns)
            )

        duplicate_rows = (
            final_df.groupBy(
                "loan_id",
                "observation_age",
            )
            .count()
            .filter(
                F.col("count") > 1,
            )
            .limit(1)
        )

        if duplicate_rows.count() > 0:
            raise ValueError(
                "Duplicate loan_id x observation_age values "
                "detected across behavioral modelling vintages."
            )

    logger.info(
        "Spark modelling datasets loaded: " "approach=%s vintages=%s columns=%s",
        approach,
        vintages,
        len(final_df.columns),
    )

    return final_df
