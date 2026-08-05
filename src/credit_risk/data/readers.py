from pathlib import Path
import pandas as pd
from credit_risk.data.schemas import (
    PERFORMANCE_SCHEMA,
    ORIGINATION_SCHEMA,
    Field,
    get_column_names,
)


def _read_pipe_delimited(
    path: str | Path, schema: tuple[Field, ...], nrows: int | None = None
) -> pd.DataFrame:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    columns = get_column_names(schema)

    df = pd.read_csv(
        path,
        sep="|",
        header=None,
        names=columns,
        nrows=nrows,
        low_memory=False,
    )

    if df.shape[1] != len(schema):
        raise ValueError(
            f"Schema mismatch for {path.name}: "
            f"expected {len(schema)} fields, "
            f"received {df.shape[1]}."
        )

    return df


def read_performance(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    return _read_pipe_delimited(path=path, schema=PERFORMANCE_SCHEMA, nrows=nrows)


def read_origination(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    return _read_pipe_delimited(path=path, schema=ORIGINATION_SCHEMA, nrows=nrows)
