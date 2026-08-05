from pathlib import Path
import pandas as pd
from credit_risk.data.schemas import PERFORMANCE_SCHEMA, get_column_names


def read_performance(path: str | Path, nrows: int | None = None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Performance file not found: {path}")
    columns = get_column_names(PERFORMANCE_SCHEMA)
    df = pd.read_csv(
        path, sep="|", header=None, names=columns, nrows=nrows, low_memory=False
    )
    if df.shape[1] != len(PERFORMANCE_SCHEMA):
        raise ValueError(
            f"Performance schema mismatch. "
            f"Expected {len(PERFORMANCE_SCHEMA)} columns, "
            f"received {df.shape[1]}."
        )
    return df
