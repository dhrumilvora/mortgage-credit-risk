from pathlib import Path
import pandas as pd
from credit_risk.data.schemas import (
    PERFORMANCE_SCHEMA,
    ORIGINATION_SCHEMA,
    ORIGINATION_RAW_DTYPES,
    PERFORMANCE_RAW_DTYPES,
    Field,
    get_column_names,
)
from collections.abc import Iterator


def _read_pipe_delimited(
    path: str | Path,
    schema: tuple[Field, ...],
    dtypes: dict[str, str] | None = None,
    nrows: int | None = None,
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
        dtype=dtypes,
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


def iter_performance(
    path: str | Path, chunksize: int = 250000
) -> Iterator[pd.DataFrame]:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    yield from pd.read_csv(
        path,
        sep="|",
        header=None,
        names=get_column_names(PERFORMANCE_SCHEMA),
        dtype=PERFORMANCE_RAW_DTYPES,
        chunksize=chunksize,
        low_memory=False,
    )


def read_performance(
    path: str | Path,
    nrows: int | None = None,
) -> pd.DataFrame:

    return _read_pipe_delimited(
        path=path,
        schema=PERFORMANCE_SCHEMA,
        dtypes=PERFORMANCE_RAW_DTYPES,
        nrows=nrows,
    )


def read_origination(
    path: str | Path,
    nrows: int | None = None,
) -> pd.DataFrame:

    return _read_pipe_delimited(
        path=path,
        schema=ORIGINATION_SCHEMA,
        dtypes=ORIGINATION_RAW_DTYPES,
        nrows=nrows,
    )
