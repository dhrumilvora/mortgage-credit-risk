from pathlib import Path
import pandas as pd
import logging
from pyspark.sql import DataFrame
logger = logging.getLogger(__name__)


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)


def write_spark_parquet(
    df: DataFrame,
    path: Path,
) -> None:
    """Write a Spark DataFrame as Parquet."""

    logger.info(
        "Spark writing parquet: %s",
        path,
    )

    (
        df.write.mode("overwrite").parquet(
            str(path),
        )
    )
