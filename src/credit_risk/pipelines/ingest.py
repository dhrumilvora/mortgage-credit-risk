import shutil
from pathlib import Path

from credit_risk.data.readers import iter_performance, read_origination
from credit_risk.data.transformers import transform_origination, transform_performance
from credit_risk.data.validation import validate_origination, validate_performance
from credit_risk.data.writers import write_parquet
from credit_risk.utils.config import create_path


def ingest(config: dict) -> None:
    if config["parameters"]["data"]["ingestion"]["skip"]:
        return
    raw_dir = Path(config["catalog"]["base"])
    orig_path = create_path(
        raw_dir,
        config["catalog"],
        "raw_origination",
        config["parameters"]["data"]["data_provider"],
        config["parameters"]["data"]["vintage"],
    )
    perf_path = create_path(
        raw_dir,
        config["catalog"],
        "raw_performance",
        config["parameters"]["data"]["data_provider"],
        config["parameters"]["data"]["vintage"],
    )

    output_origination = create_path(
        raw_dir,
        config["catalog"],
        "origination_path",
        config["parameters"]["data"]["data_provider"],
        config["parameters"]["data"]["vintage"],
        must_exist=False,
    )

    ## Origination ##
    orig = read_origination(orig_path)
    orig = transform_origination(orig)
    orig_validation = validate_origination(orig)
    orig_validation.raise_if_invalid()

    write_parquet(orig, output_origination)

    ## Performance ##

    perf_output_dir = create_path(
        raw_dir,
        config["catalog"],
        "performance_path",
        config["parameters"]["data"]["data_provider"],
        config["parameters"]["data"]["vintage"],
        must_exist=False,
    )
    if perf_output_dir.exists():
        shutil.rmtree(perf_output_dir)
    perf_output_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0

    for chunk_number, chunk in enumerate(
        iter_performance(
            perf_path,
            chunksize=config["parameters"]["data"]["ingestion"]["chunksize"],
        )
    ):
        chunk = transform_performance(chunk)
        validation = validate_performance(chunk)
        validation.raise_if_invalid()

        write_parquet(chunk, perf_output_dir / f"part-{chunk_number:05d}.parquet")

        total_rows += len(chunk)

    print(
        f"Year {config['parameters']['data']['vintage']} ingestion complete. "
        f"Performance rows: {total_rows:,}"
    )
