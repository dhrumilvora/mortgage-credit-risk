from __future__ import annotations

import logging
from copy import deepcopy
from time import perf_counter

import pandas as pd

from credit_risk.evaluations.evaluations import (
    evaluate_dataset,
    generate_predictions,
)
from credit_risk.evaluations.reporting import save_evaluation_results
from credit_risk.evaluations.shap import evaluate_shap
from credit_risk.modelling.artifacts import load_model_artifacts, load_training_config
from credit_risk.modelling.preprocessing import split_features_target
from credit_risk.utils.config import create_path

logger = logging.getLogger(__name__)


def evaluate_split(
    model,
    preprocessor,
    df: pd.DataFrame,
    config: dict,
) -> dict:
    """Generate predictions and evaluate a single dataset split."""

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
    """Run model evaluation for configured datasets and persist results."""

    start = perf_counter()
    approach = config["parameters"]["modelling_approach"]
    evaluation_config = config["parameters"]["evaluation"]

    if evaluation_config["skip"]:
        logger.info("Evaluation pipeline skipped by configuration")
        return

    mode = evaluation_config["mode"]

    # --------------------------------------------------------------
    # Resolve model configuration
    # --------------------------------------------------------------

    if mode == "same_run":
        model_config = config
        scoring_config = config

    elif mode == "existing_model":
        model_config = deepcopy(config)

        model_config["parameters"]["modelling"]["version"] = evaluation_config["model"][
            "version"
        ]

        model_config["parameters"]["modelling"]["algorithm"] = evaluation_config[
            "model"
        ]["type"]

        training_config = load_training_config(model_config)
        scoring_config = deepcopy(config)
        scoring_config["parameters"]["modelling"]["features"] = training_config[
            "features"
        ]

    else:
        raise ValueError(f"Unsupported evaluation mode: {mode}")

    # --------------------------------------------------------------
    # Load model artifacts
    # --------------------------------------------------------------

    model, preprocessor = load_model_artifacts(model_config)

    logger.info(
        "Evaluation model loaded: mode=%s version=%s algorithm=%s",
        mode,
        model_config["parameters"]["modelling"]["version"],
        model_config["parameters"]["modelling"]["algorithm"],
    )

    validation_evaluation = None
    oot_evaluation = None

    # --------------------------------------------------------------
    # Validation evaluation
    # --------------------------------------------------------------

    if evaluation_config["datasets"]["validation"]:

        validation_path = create_path(
            config["catalog"]["base"], config["catalog"], "validation_df", approach
        )

        validation_df = pd.read_parquet(validation_path)

        validation_evaluation = evaluate_split(
            model=model,
            preprocessor=preprocessor,
            df=validation_df,
            config=scoring_config,
        )
        evaluate_shap(
            model=model,
            preprocessor=preprocessor,
            df=validation_df,
            config=scoring_config,
            dataset_name="validation",
        )

        logger.info("Validation evaluation completed.")

    # --------------------------------------------------------------
    # OOT evaluation
    # --------------------------------------------------------------

    if evaluation_config["datasets"]["oot"]:

        oot_path = create_path(
            config["catalog"]["base"], config["catalog"], "oot_df", approach
        )

        oot_df = pd.read_parquet(oot_path)

        oot_evaluation = evaluate_split(
            model=model,
            preprocessor=preprocessor,
            df=oot_df,
            config=scoring_config,
        )
        evaluate_shap(
            model=model,
            preprocessor=preprocessor,
            df=oot_df,
            config=scoring_config,
            dataset_name="oot",
        )
        logger.info("OOT evaluation completed.")

    # --------------------------------------------------------------
    # Persist evaluation results
    # --------------------------------------------------------------

    save_evaluation_results(
        validation_evaluation=validation_evaluation,
        oot_evaluation=oot_evaluation,
        config=config,
    )

    logger.info(
        "Evaluation pipeline completed: duration_seconds=%.2f",
        perf_counter() - start,
    )
