from __future__ import annotations

import logging
from time import perf_counter

import pandas as pd

from credit_risk.evaluations.evaluations import (
    evaluate_dataset,
    generate_predictions,
)
from credit_risk.evaluations.reporting import save_evaluation_results
from credit_risk.modelling.artifacts import load_model_artifacts
from credit_risk.modelling.preprocessing import split_features_target
from credit_risk.utils.config import create_path

logger = logging.getLogger(__name__)


def evaluate_split(
    model,
    preprocessor,
    df,
    config,
) -> dict:
    """
    Generate predictions and evaluate a single dataset split.
    """

    X, y = split_features_target(
        df,
        config,
    )

    evaluation_config = config["parameters"]["evaluation"]

    y_pred, y_proba = generate_predictions(
        model=model,
        preprocessor=preprocessor,
        X=X,
        threshold=evaluation_config["classification"]["threshold"],
    )

    return evaluate_dataset(
        y_true=y,
        y_pred=y_pred,
        y_proba=y_proba,
        n_deciles=evaluation_config["risk"]["n_deciles"],
        calibration_bins=evaluation_config["calibration"]["bins"],
    )


def run_evaluation_pipeline(
    config: dict,
) -> None:
    """
    Run model evaluation for the configured datasets and
    persist evaluation artifacts.
    """
    model = None
    preprocessor = None
    start = perf_counter()

    evaluation_config = config["parameters"]["evaluation"]

    if evaluation_config["skip"]:
        logger.info("Evaluation pipeline skipped by configuration")
        return

    mode = evaluation_config["mode"]

    if mode == "same_run":
        model, preprocessor = load_model_artifacts(config)
    elif mode == "existing_model":
        config_copy = {**config}
        config_copy["parameters"]["modelling"]["version"] = config["parameters"][
            "evaluation"
        ]["model"]["version"]

        config_copy["parameters"]["modelling"]["algorithm"] = config["parameters"][
            "evaluation"
        ]["model"]["type"]
        model, preprocessor = load_model_artifacts(config_copy)

    else:

        raise ValueError(f"Unsupported evaluation mode: {mode}")

    logger.info(
        "Evaluation model loaded: mode=%s",
        mode,
    )

    validation_evaluation = None
    oot_evaluation = None

    if evaluation_config["datasets"]["validation"]:

        validation_path = create_path(
            config["catalog"]["base"],
            config["catalog"],
            "validation_df",
        )

        validation_df = pd.read_parquet(validation_path)

        validation_evaluation = evaluate_split(
            model=model,
            preprocessor=preprocessor,
            df=validation_df,
            config=config,
        )

        logger.info("Validation evaluation completed.")

    if evaluation_config["datasets"]["oot"]:

        oot_path = create_path(
            config["catalog"]["base"],
            config["catalog"],
            "oot_df",
        )

        oot_df = pd.read_parquet(oot_path)

        oot_evaluation = evaluate_split(
            model=model,
            preprocessor=preprocessor,
            df=oot_df,
            config=config,
        )

        logger.info("OOT evaluation completed.")

    save_evaluation_results(
        validation_evaluation=validation_evaluation,
        oot_evaluation=oot_evaluation,
        config=config,
    )

    logger.info(
        "Evaluation pipeline completed: " "duration_seconds=%.2f",
        perf_counter() - start,
    )
