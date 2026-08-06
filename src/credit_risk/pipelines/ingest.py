from pathlib import Path
from credit_risk.data.readers import iter_performance, read_origination
from credit_risk.data.validation import validate_origination, validate_performance
from credit_risk.data.writers import write_parquet
from credit_risk.data.transformers import transform_origination, transform_performance
import pandas as pd


def ingest(
    year: int, raw_root: Path, interim_root: Path, chunksize: int = 250000
) -> pd.DataFrame:
    raw_dir = raw_root / str(year)
    orig_path = raw_dir / f"sample_orig_{year}.txt"
    perf_path = raw_dir / f"sample_perf_{year}.txt"

    output_dir = interim_root / "freddie_mac" / str(year)

    ## Origination ##
    orig = read_origination(orig_path)
    orig = transform_origination(orig)
    orig_validation = validate_origination(orig)
    orig_validation.raise_if_invalid()

    write_parquet(orig, output_dir / "origination.parquet")

    ## Performance ##

    perf_output_dir = output_dir / "performance"
    perf_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_rows = 0

    for chunk_number, chunk in enumerate(
        iter_performance(
            perf_path,
            chunksize=chunksize,
        )
    ):
        chunk = transform_performance(chunk)
        validation = validate_performance(chunk)
        validation.raise_if_invalid()

        write_parquet(
            chunk,
            perf_output_dir / f"part-{chunk_number:05d}.parquet",
        )

        total_rows += len(chunk)

    print(f"Year {year} ingestion complete. " f"Performance rows: {total_rows:,}")
    return output_dir / "origination.parquet", perf_output_dir
