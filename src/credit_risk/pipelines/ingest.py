import logging
import shutil
from pathlib import Path
from time import perf_counter

from credit_risk.data.readers import iter_performance, read_origination
from credit_risk.data.transformers import transform_origination, transform_performance
from credit_risk.data.validation import validate_origination, validate_performance
from credit_risk.data.writers import write_parquet
from credit_risk.utils.config import create_path

logger = logging.getLogger(__name__)


def ingest(config: dict) -> None:
    if config["parameters"]["data"]["ingestion"]["skip"]:
        logger.info("Ingestion skipped by configuration")
        return

    start = perf_counter()
    data_config = config["parameters"]["data"]
    provider = data_config["data_provider"]
    for vintage in data_config["all_vintages"]:

        logger.info("Ingestion started: provider=%s vintage=%s", provider, vintage)

        raw_dir = Path(config["catalog"]["base"])
        orig_path = create_path(
            raw_dir,
            config["catalog"],
            "raw_origination",
            provider,
            vintage,
        )
        perf_path = create_path(
            raw_dir,
            config["catalog"],
            "raw_performance",
            provider,
            vintage,
        )

        output_origination = create_path(
            raw_dir,
            config["catalog"],
            "origination_path",
            provider,
            vintage,
            must_exist=False,
        )

        ## Origination ##
        orig = read_origination(orig_path)
        orig = transform_origination(orig)
        orig_validation = validate_origination(orig)
        orig_validation.raise_if_invalid()

        write_parquet(orig, output_origination)
        logger.info(
            "Origination data written: rows=%s path=%s",
            f"{len(orig):,}",
            output_origination,
        )

        ## Performance ##

        perf_output_dir = create_path(
            raw_dir,
            config["catalog"],
            "performance_path",
            provider,
            vintage,
            must_exist=False,
        )
        if perf_output_dir.exists():
            logger.info(
                "Removing existing performance output: path=%s", perf_output_dir
            )
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

        logger.info(
            "Ingestion completed: vintage=%s performance_rows=%s chunks=%s duration_seconds=%.2f",
            vintage,
            f"{total_rows:,}",
            chunk_number + 1 if total_rows else 0,
            perf_counter() - start,
        )
