from __future__ import annotations

import logging

from pyspark.sql import SparkSession
import os, sys

logger = logging.getLogger(__name__)


def create_spark_session(
    config: dict,
) -> SparkSession:
    """
    Create the Spark session used by preprocessing.

    Spark configuration is environment-driven so the same code can run
    locally and on the GCP VM.
    """

    spark_config = config["parameters"]["spark"]
    python_executable = sys.executable

    os.environ["PYSPARK_PYTHON"] = python_executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_executable
    master = spark_config.get(
        "master",
        "local[*]",
    )

    driver_memory = spark_config.get(
        "driver_memory",
        "12g",
    )

    shuffle_partitions = int(
        spark_config.get(
            "shuffle_partitions",
            16,
        )
    )

    logger.info(
        "Creating Spark session: " "master=%s driver_memory=%s shuffle_partitions=%s",
        master,
        driver_memory,
        shuffle_partitions,
    )

    spark = (
        SparkSession.builder.appName(
            "mortgage-credit-risk-preprocessing",
        )
        .master(
            master,
        )
        .config(
            "spark.driver.memory",
            driver_memory,
        )
        .config(
            "spark.sql.shuffle.partitions",
            shuffle_partitions,
        )
        .config(
            "spark.default.parallelism",
            shuffle_partitions,
        )
        .config(
            "spark.sql.adaptive.enabled",
            "true",
        )
        .config(
            "spark.sql.adaptive.coalescePartitions.enabled",
            "true",
        )
        .config(
            "spark.sql.adaptive.skewJoin.enabled",
            "true",
        )
        .config(
            "spark.pyspark.python",
            python_executable,
        )
        .config(
            "spark.pyspark.driver.python",
            python_executable,
        )
        .config(
            "spark.sql.execution.pyspark.udf.faulthandler.enabled",
            "true",
        )
        .config(
            "spark.python.worker.faulthandler.enabled",
            "true",
        )
        .getOrCreate()
    )

    logger.info(
        "Spark session created: " "version=%s default_parallelism=%s",
        spark.version,
        spark.sparkContext.defaultParallelism,
    )

    return spark
