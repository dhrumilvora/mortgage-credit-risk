from __future__ import annotations
from pyspark.sql import DataFrame, Window, functions as F


def validate_ranking_input(df: DataFrame, score_column: str) -> None:
    required_columns = {
        "loan_id",
        "snapshot_age",
        score_column,
    }

    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Ranking input is missing required columns: " f"{sorted(missing_columns)}"
        )

    if df.limit(1).count() == 0:
        raise ValueError("Ranking input DataFrame is empty.")

    if df.filter(F.col(score_column).isNull()).limit(1).count() > 0:
        raise ValueError(f"Ranking score column '{score_column}' contains null values.")

    duplicate_count = (
        df.groupBy("loan_id", "snapshot_age")
        .count()
        .filter(F.col("count") > 1)
        .limit(1)
        .count()
    )

    if duplicate_count > 0:
        raise ValueError("Ranking input must have unique loan_id × snapshot_age grain.")


def add_risk_rank(df: DataFrame, score_column: str) -> DataFrame:
    """Add risk percentile within each snapshot age."""
    window = Window.partitionBy("snapshot_age").orderBy(F.col(score_column).desc())
    return df.withColumn(
        "risk_percentile",
        1 - F.percent_rank().over(window),
    )


def add_risk_decile(df: DataFrame, score_column: str) -> DataFrame:
    window = Window.partitionBy("snapshot_age").orderBy(
        F.col(score_column).desc(),
        F.col("loan_id").asc(),
    )

    return df.withColumn(
        "risk_decile",
        F.ntile(10).over(window),
    )


def add_risk_percentile(
    df: DataFrame,
    score_column: str,
) -> DataFrame:
    """Add risk percentile within each snapshot age."""
    window = Window.partitionBy("snapshot_age").orderBy(
        F.col(score_column).desc(),
    )

    return df.withColumn(
        "risk_percentile",
        1 - F.percent_rank().over(window),
    )


def build_risk_ranking(
    df: DataFrame,
    score_column: str,
) -> DataFrame:
    """Build deterministic risk ranking from loan-level PD scores."""
    validate_ranking_input(df, score_column)

    df = add_risk_rank(df, score_column)
    df = add_risk_percentile(df, score_column)
    df = add_risk_decile(df, score_column)

    return df
