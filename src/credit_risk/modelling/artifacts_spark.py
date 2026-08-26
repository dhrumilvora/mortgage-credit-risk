from __future__ import annotations

import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml
from pyspark.ml.pipeline import PipelineModel

from credit_risk.utils.config import create_path

logger = logging.getLogger(__name__)


def _save_spark_artifact(
    artifact: Any,
    path: str | Path,
) -> None:
    """Persist a Spark ML artifact using Spark's native format."""

    start = perf_counter()

    artifact.write().overwrite().save(
        str(path),
    )

    logger.info(
        "Spark artifact saved: path=%s duration_seconds=%.2f",
        path,
        perf_counter() - start,
    )


def _load_spark_artifact(
    loader,
    path: str | Path,
) -> Any:
    """Load a Spark ML artifact using its corresponding reader."""

    start = perf_counter()

    artifact = loader.load(
        str(path),
    )

    logger.info(
        "Spark artifact loaded: path=%s duration_seconds=%.2f",
        path,
        perf_counter() - start,
    )

    return artifact


def _save_json(
    artifact: dict,
    path: str | Path,
) -> None:
    """Save JSON metadata."""

    start = perf_counter()

    path = Path(path)

    with path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            artifact,
            file,
            indent=4,
            sort_keys=True,
        )

    logger.info(
        "JSON artifact saved: path=%s duration_seconds=%.2f",
        path,
        perf_counter() - start,
    )


def _save_yaml(
    artifact: dict,
    path: str | Path,
) -> None:
    """Save YAML configuration."""

    start = perf_counter()

    path = Path(path)

    with path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            artifact,
            file,
            sort_keys=False,
        )

    logger.info(
        "YAML artifact saved: path=%s duration_seconds=%.2f",
        path,
        perf_counter() - start,
    )


def _load_yaml(
    path: str | Path,
) -> dict:
    """Load YAML configuration."""

    start = perf_counter()

    path = Path(path)

    with path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        artifact = yaml.safe_load(file)

    logger.info(
        "YAML artifact loaded: path=%s duration_seconds=%.2f",
        path,
        perf_counter() - start,
    )

    return artifact


def save_artifacts_spark(
    model,
    preprocessor: PipelineModel,
    training_metadata: dict,
    config: dict,
) -> None:
    """Save Spark model, preprocessor, metadata and configuration."""

    approach = config["parameters"]["modelling_approach"]
    modelling = config["parameters"]["modelling"]

    version = modelling["version"]
    model_type = modelling["algorithm"]

    catalog = config["catalog"]
    base_path = catalog["base"]

    start = perf_counter()

    model_path = create_path(
        base_path,
        catalog,
        "model",
        approach,
        version,
        model_type,
        must_exist=False,
    )

    preprocessor_path = create_path(
        base_path,
        catalog,
        "preprocessor",
        approach,
        version,
        model_type,
        must_exist=False,
    )

    training_metadata_path = create_path(
        base_path,
        catalog,
        "training_metadata",
        approach,
        version,
        model_type,
        must_exist=False,
    )

    training_config_path = create_path(
        base_path,
        catalog,
        "training_config",
        approach,
        version,
        model_type,
        must_exist=False,
    )

    # --------------------------------------------------------------
    # Spark-native model
    # --------------------------------------------------------------

    _save_spark_artifact(
        model,
        model_path,
    )

    # --------------------------------------------------------------
    # Spark-native fitted preprocessing pipeline
    # --------------------------------------------------------------

    _save_spark_artifact(
        preprocessor,
        preprocessor_path,
    )

    # --------------------------------------------------------------
    # Driver-side metadata/config
    # --------------------------------------------------------------

    _save_json(
        training_metadata,
        training_metadata_path,
    )

    _save_yaml(
        modelling,
        training_config_path,
    )

    logger.info(
        "Spark training artifacts saved: "
        "version=%s model_type=%s duration_seconds=%.2f",
        version,
        model_type,
        perf_counter() - start,
    )


def load_spark_model_artifacts(
    config: dict,
):
    """Load a trained Spark model and fitted preprocessor."""

    approach = config["parameters"]["modelling_approach"]
    modelling = config["parameters"]["modelling"]

    version = modelling["version"]
    model_type = modelling["algorithm"]

    catalog = config["catalog"]
    base_path = catalog["base"]

    model_path = create_path(
        base_path,
        catalog,
        "model",
        approach,
        version,
        model_type,
        must_exist=True,
    )

    preprocessor_path = create_path(
        base_path,
        catalog,
        "preprocessor",
        approach,
        version,
        model_type,
        must_exist=True,
    )

    preprocessor = _load_spark_artifact(
        PipelineModel,
        preprocessor_path,
    )

    if model_type == "logistic_regression":

        from pyspark.ml.classification import (
            LogisticRegressionModel,
        )

        model_loader = LogisticRegressionModel

    elif model_type == "random_forest":

        from pyspark.ml.classification import (
            RandomForestClassificationModel,
        )

        model_loader = RandomForestClassificationModel

    elif model_type == "xgboost":

        from xgboost.spark import (
            SparkXGBClassifierModel,
        )

        model_loader = SparkXGBClassifierModel

    elif model_type == "lightgbm":

        raise NotImplementedError(
            "Spark LightGBM artifact loading is not implemented yet."
        )

    else:
        raise ValueError(f"Unsupported Spark modelling algorithm: {model_type}")

    model = _load_spark_artifact(
        model_loader,
        model_path,
    )

    return model, preprocessor


def load_training_config_spark(
    config: dict,
) -> dict:
    """Load the modelling configuration saved with a Spark model."""

    approach = config["parameters"]["modelling_approach"]
    modelling = config["parameters"]["modelling"]
    catalog = config["catalog"]

    training_config_path = create_path(
        catalog["base"],
        catalog,
        "training_config",
        approach,
        modelling["version"],
        modelling["algorithm"],
        must_exist=True,
    )

    return _load_yaml(
        training_config_path,
    )
