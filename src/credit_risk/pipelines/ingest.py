from __future__ import annotations

import logging
import shutil

# 1. Switched from ThreadPoolExecutor to ProcessPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import pandas as pd

from credit_risk.data.readers import iter_performance, read_origination
from credit_risk.data.transformers import (
    transform_origination,
    transform_performance,
)
from credit_risk.data.validation import (
    validate_origination,
    validate_performance,
)
from credit_risk.data.writers import write_parquet
from credit_risk.utils.config import create_path

logger = logging.getLogger(__name__)


def _process_performance_quarter(
    *,
    raw_dir: Path,
    catalog: dict,
    provider: str,
    vintage: int,
    quarter: str,
    performance_output_dir: Path,
    chunksize: int,
    quarter_index: int,
) -> tuple[int, int]:
    """
    Process one performance quarter.

    Returns:
        total_rows: Number of processed rows.
        chunk_count: Number of parquet chunks written.
    """

    quarter_start = perf_counter()

    perf_path = create_path(
        raw_dir,
        catalog,
        "raw_performance",
        provider,
        vintage,
        quarter,
    )

    logger.info(
        "Performance quarter started: vintage=%s quarter=%s",
        vintage,
        quarter,
    )

    total_rows = 0
    chunk_count = 0

    for chunk_number, chunk in enumerate(
        iter_performance(
            perf_path,
            chunksize=chunksize,
        )
    ):
        chunk = transform_performance(chunk)

        validation = validate_performance(chunk)
        validation.raise_if_invalid()

        # Make chunk names unique across quarters.
        output_path = (
            performance_output_dir
            / f"part-q{quarter_index:02d}-{chunk_number:05d}.parquet"
        )

        write_parquet(
            chunk,
            output_path,
        )

        total_rows += len(chunk)
        chunk_count += 1

    logger.info(
        "Performance quarter completed: "
        "vintage=%s quarter=%s rows=%s chunks=%s duration_seconds=%.2f",
        vintage,
        quarter,
        f"{total_rows:,}",
        chunk_count,
        perf_counter() - quarter_start,
    )

    return total_rows, chunk_count


def ingest(config: dict) -> None:
    """Ingest all configured Freddie Mac vintages."""

    if config["parameters"]["data"]["ingestion"]["skip"]:
        logger.info("Ingestion skipped by configuration")
        return

    ingestion_start = perf_counter()

    data_config = config["parameters"]["data"]
    ingestion_config = data_config["ingestion"]

    provider = data_config["data_provider"]
    quarters = data_config["quarters"]

    chunksize = ingestion_config["chunksize"]

    # Config-driven performance parallelism.
    max_workers = ingestion_config.get(
        "performance_workers",
        2,
    )

    raw_dir = Path(config["catalog"]["base"])

    logger.info(
        "Ingestion started: provider=%s vintages=%s quarters=%s "
        "performance_workers=%s chunksize=%s",
        provider,
        data_config["all_vintages"],
        quarters,
        max_workers,
        chunksize,
    )

    for vintage in data_config["all_vintages"]:

        vintage_start = perf_counter()

        logger.info(
            "Vintage ingestion started: provider=%s vintage=%s",
            provider,
            vintage,
        )

        # ============================================================
        # Origination
        # ============================================================

        output_origination = create_path(
            raw_dir,
            config["catalog"],
            "origination_path",
            provider,
            vintage,
            must_exist=False,
        )

        orig_parts: list[pd.DataFrame] = []

        for quarter in quarters:

            quarter_start = perf_counter()

            orig_path = create_path(
                raw_dir,
                config["catalog"],
                "raw_origination",
                provider,
                vintage,
                quarter,
            )

            logger.info(
                "Origination quarter started: " "vintage=%s quarter=%s",
                vintage,
                quarter,
            )

            orig = read_origination(orig_path)
            orig = transform_origination(orig)

            validation = validate_origination(orig)
            validation.raise_if_invalid()

            orig_parts.append(orig)

            logger.info(
                "Origination quarter completed: "
                "vintage=%s quarter=%s rows=%s duration_seconds=%.2f",
                vintage,
                quarter,
                f"{len(orig):,}",
                perf_counter() - quarter_start,
            )

        if not orig_parts:
            raise ValueError(f"No origination data found for vintage={vintage}")

        orig = pd.concat(
            orig_parts,
            ignore_index=True,
        )

        write_parquet(
            orig,
            output_origination,
        )

        logger.info(
            "Origination data written: " "vintage=%s rows=%s path=%s",
            vintage,
            f"{len(orig):,}",
            output_origination,
        )

        # Release quarterly origination DataFrames before performance
        # processing to avoid retaining unnecessary memory.
        del orig_parts
        del orig

        # ============================================================
        # Performance
        # ============================================================

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
                "Removing existing performance output: path=%s",
                perf_output_dir,
            )
            shutil.rmtree(perf_output_dir)

        perf_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        total_rows = 0
        total_chunks = 0

        # 2. Swapped to ProcessPoolExecutor for true CPU multiprocessing
        with ProcessPoolExecutor(
            max_workers=max_workers,
        ) as executor:

            futures = {
                executor.submit(
                    _process_performance_quarter,
                    raw_dir=raw_dir,
                    catalog=config["catalog"],
                    provider=provider,
                    vintage=vintage,
                    quarter=quarter,
                    performance_output_dir=perf_output_dir,
                    chunksize=chunksize,
                    quarter_index=quarter_index,
                ): quarter
                for quarter_index, quarter in enumerate(quarters, start=1)
            }

            for future in as_completed(futures):

                quarter = futures[future]

                try:
                    quarter_rows, quarter_chunks = future.result()

                except Exception:
                    logger.exception(
                        "Performance quarter failed: " "vintage=%s quarter=%s",
                        vintage,
                        quarter,
                    )
                    raise

                total_rows += quarter_rows
                total_chunks += quarter_chunks

        logger.info(
            "Vintage ingestion completed: "
            "vintage=%s performance_rows=%s performance_chunks=%s "
            "duration_seconds=%.2f",
            vintage,
            f"{total_rows:,}",
            total_chunks,
            perf_counter() - vintage_start,
        )

    logger.info(
        "Ingestion completed: provider=%s " "duration_seconds=%.2f",
        provider,
        perf_counter() - ingestion_start,
    )
