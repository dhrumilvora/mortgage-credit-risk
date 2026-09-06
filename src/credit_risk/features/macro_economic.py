import logging

from pyspark.sql import DataFrame, functions as F

from credit_risk.utils.config import create_path


logger = logging.getLogger(__name__)


def add_macro_economic_features(
    df: DataFrame,
    config: dict,
    spark,
) -> DataFrame:
    """
    Add monthly macroeconomic features to the loan-month dataset.

    The macro period is converted to the same monthly ordinal representation
    used by the Freddie Mac master dataset.
    """

    macro_config = config["parameters"]['data']["preprocess"]["macro-economic"]

    if not macro_config["enabled"]:
        logger.info(
            "Macro Economic Features skipped by configuration"
        )
        return df

    path = create_path(
        config["catalog"]["base"],
        config["catalog"],
        "raw_macro_data",
    )

    macro_features = macro_config["features"]

    macro_df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(str(path))
        .select(
            "period",
            *macro_features,
        )
    )

    macro_spark = (
        macro_df
        .withColumn(
            "_macro_period",
            F.to_date(
                F.col("period"),
                "yyyy-MM-dd",
            ),
        )
        .withColumn(
            "period",
            (
                (F.year("_macro_period") - F.lit(1970)) * F.lit(12)
                + (F.month("_macro_period") - F.lit(1))
            ).cast("long"),
        )
        .drop("_macro_period")
    )

    for feature in macro_features:
        macro_spark = macro_spark.withColumn(
            feature,
            F.col(feature).cast("double"),
        )

    result = df.join(
        macro_spark,
        on="period",
        how="left",
    )

    return result