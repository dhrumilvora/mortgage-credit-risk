from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def build_early_intervention_flags(
    df: DataFrame,
    threshold: float,
    pd_column: str = "p1_pd_3m",
) -> DataFrame:
    """
    Add an operational early-intervention flag.

    A loan is flagged when its predicted 3-month PD is greater than
    or equal to the configured intervention threshold.

    Parameters
    ----------
    df:
        Scored P1 population.

    threshold:
        PD threshold used to trigger intervention.

    pd_column:
        P1 predicted probability column.
    """

    if not 0 < threshold < 1:
        raise ValueError("Early-intervention threshold must be between 0 and 1.")

    if pd_column not in df.columns:
        raise ValueError(f"Required PD column '{pd_column}' not found.")

    return df.withColumn(
        "intervention_flag",
        F.when(
            F.col(pd_column) >= F.lit(threshold),
            F.lit(1),
        )
        .otherwise(F.lit(0))
        .cast("byte"),
    ).withColumn(
        "intervention_band",
        F.when(
            F.col(pd_column) >= F.lit(threshold),
            F.lit("FLAG"),
        ).otherwise(F.lit("NO_FLAG")),
    )
