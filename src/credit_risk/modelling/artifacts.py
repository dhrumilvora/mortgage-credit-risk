from __future__ import annotations
from credit_risk.utils.config import create_path
from pathlib import Path
from time import perf_counter
import logging
import joblib
import json
import yaml

logger = logging.getLogger(__name__)


def _save_joblib(artifact: Any, path: Path) -> None:
    start = perf_counter()
    joblib.dump(artifact, path)

    logger.info(
        "Joblib artifact saved: path=%s duration_seconds=%.2f",
        path,
        perf_counter() - start,
    )


def _load_joblib(path: Path) -> Any:
    start = perf_counter()
    artifact = joblib.load(path)

    logger.info(
        "Joblib artifact loaded: path=%s duration_seconds=%.2f",
        path,
        perf_counter() - start,
    )

    return artifact


def _save_json(artifact: dict, path: Path) -> None:
    start = perf_counter()

    with path.open(mode="w", encoding="utf-8") as file:
        json.dump(artifact, file, indent=4, sort_keys=True)
    logger.info(
        "JSON artifact saved: path=%s duration_seconds=%.2f",
        path,
        perf_counter() - start,
    )


def _load_json(path: Path) -> dict:
    start = perf_counter()

    with path.open(mode="r", encoding="utf-8") as file:
        artifact = json.load(file)

    logger.info(
        "JSON artifact loaded: path=%s duration_seconds=%.2f",
        path,
        perf_counter() - start,
    )
    return artifact


def _save_yaml(artifact: dict, path: Path) -> None:
    start = perf_counter()
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
    """
    Load a YAML artifact.

    Parameters
    ----------
    path
        YAML artifact path.

    Returns
    -------
    dict
        Loaded YAML artifact.
    """

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


def save_artifacts(model, preprocessor, training_metadata, config: dict) -> None:
    start = perf_counter()
    modelling = config["parameters"]["modelling"]
    version = modelling["version"]
    model_type = modelling["algorithm"]
    catalog = config["catalog"]
    base_path = catalog["base"]

    model_path = create_path(
        base_path, catalog, "model", version, model_type, must_exist=False
    )

    preprocessor_path = create_path(
        base_path,
        catalog,
        "preprocessor",
        version,
        model_type,
        must_exist=False,
    )

    training_metadata_path = create_path(
        base_path,
        catalog,
        "training_metadata",
        version,
        model_type,
        must_exist=False,
    )

    training_config_path = create_path(
        base_path,
        catalog,
        "training_config",
        version,
        model_type,
        must_exist=False,
    )

    _save_joblib(
        model,
        model_path,
    )

    _save_joblib(
        preprocessor,
        preprocessor_path,
    )

    _save_json(
        training_metadata,
        training_metadata_path,
    )

    _save_yaml(
        modelling,
        training_config_path,
    )
    logger.info(
        "Training artifacts saved: " "version=%s model_type=%s duration_seconds=%.2f",
        version,
        model_type,
        perf_counter() - start,
    )


def load_model_artifacts(
    config: dict,
):
    """
    Load the trained model and fitted preprocessor.

    Parameters
    ----------
    config
        Project configuration.

    Returns
    -------
    tuple
        Trained model and fitted preprocessor.
    """

    modelling = config["parameters"]["modelling"]

    version = modelling["version"]
    model_type = modelling["algorithm"]

    catalog = config["catalog"]
    base_path = catalog["base"]

    model_path = create_path(
        base_path,
        catalog,
        "model",
        version,
        model_type,
        must_exist=True,
    )

    preprocessor_path = create_path(
        base_path,
        catalog,
        "preprocessor",
        version,
        model_type,
        must_exist=True,
    )

    model = _load_joblib(model_path)
    preprocessor = _load_joblib(preprocessor_path)

    return model, preprocessor
