from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def rank_early_intervention_population(
    df: DataFrame,
    pd_column: str = "p1_pd_3m",
) -> DataFrame:
    """
    Rank currently flagged loans for intervention.

    Ranking is performed independently within each reporting period.

    Highest predicted PD receives rank 1.

    Ties are resolved deterministically using:
        1. predicted PD descending
        2. UPB descending
        3. loan_id ascending
    """

    required_columns = {
        "loan_id",
        "period",
        "intervention_flag",
        pd_column,
    }

    missing_columns = sorted(required_columns - set(df.columns))

    if missing_columns:
        raise ValueError(
            "Missing columns required for intervention ranking: "
            + ", ".join(missing_columns)
        )

    if "current_actual_upb" not in df.columns:
        raise ValueError("Early-intervention ranking requires " "'current_actual_upb'.")

    ranking_window = Window.partitionBy("period").orderBy(
        F.col("intervention_flag").desc(),
        F.col(pd_column).desc(),
        F.col("current_actual_upb").desc(),
        F.col("loan_id").asc(),
    )

    return df.withColumn(
        "intervention_rank",
        F.when(
            F.col("intervention_flag") == 1,
            F.row_number().over(ranking_window),
        ).otherwise(F.lit(None).cast("long")),
    )
