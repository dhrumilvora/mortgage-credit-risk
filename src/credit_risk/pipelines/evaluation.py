from __future__ import annotations

import logging
from copy import deepcopy
from time import perf_counter

import pandas as pd

from credit_risk.evaluations.evaluations import (
    evaluate_dataset,
    generate_predictions,
    calculate_top_k_metrics,
    fit_calibration,
    apply_calibration,
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
    return_predictions: Bool = False,
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
    evaluations_results = evaluate_dataset(
        y_true=y,
        y_pred=y_pred,
        y_proba=y_proba,
        n_deciles=evaluation_config["risk"]["n_deciles"],
        calibration_bins=evaluation_config["calibration"]["bins"],
    )
    evaluations_results["top_k_metrics"] = calculate_top_k_metrics(
        y_true=y,
        y_pred=y_proba,
    ).to_dict(orient="records")
    if return_predictions:
        return evaluations_results, y, y_proba

    return evaluations_results


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

        validation_evaluation, y_validation, y_validation_proba = evaluate_split(
            model=model,
            preprocessor=preprocessor,
            df=validation_df,
            config=scoring_config,
            return_predictions=True,
        )

        evaluate_shap(
            model=model,
            preprocessor=preprocessor,
            df=validation_df,
            config=scoring_config,
            dataset_name="validation",
        )
        calibration_intercept, calibration_slope = fit_calibration(
            y_true=y_validation,
            y_proba=y_validation_proba,
        )

        logger.info(
            "Calibration fitted on validation: " "intercept=%.6f slope=%.6f",
            calibration_intercept,
            calibration_slope,
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

        oot_evaluation, y_oot, y_oot_proba = evaluate_split(
            model=model,
            preprocessor=preprocessor,
            df=oot_df,
            config=scoring_config,
            return_predictions=True,
        )
        evaluate_shap(
            model=model,
            preprocessor=preprocessor,
            df=oot_df,
            config=scoring_config,
            dataset_name="oot",
        )
        y_oot_calibrated_proba = apply_calibration(
            y_proba=y_oot_proba,
            intercept=calibration_intercept,
            slope=calibration_slope,
        )
        calibrated_threshold = scoring_config["parameters"]["evaluation"][
            "classification"
        ]["threshold"]

        y_oot_calibrated_pred = (y_oot_calibrated_proba >= calibrated_threshold).astype(
            int
        )
        calibrated_oot_evaluation = evaluate_dataset(
            y_true=y_oot,
            y_pred=y_oot_calibrated_pred,
            y_proba=y_oot_calibrated_proba,
            n_deciles=scoring_config["parameters"]["evaluation"]["risk"]["n_deciles"],
            calibration_bins=scoring_config["parameters"]["evaluation"]["calibration"][
                "bins"
            ],
        )

        calibrated_oot_evaluation["top_k_metrics"] = calculate_top_k_metrics(
            y_true=y_oot,
            y_pred=y_oot_calibrated_proba,
        ).to_dict(orient="records")

        calibrated_oot_evaluation["calibration_applied"] = {
            "intercept": calibration_intercept,
            "slope": calibration_slope,
        }
        logger.info("OOT evaluation completed.")

    # --------------------------------------------------------------
    # Persist evaluation results
    # --------------------------------------------------------------

    save_evaluation_results(
        validation_evaluation=validation_evaluation,
        oot_evaluation=oot_evaluation,
        oot_calibration=calibrated_oot_evaluation,
        config=config,
    )

    logger.info(
        "Evaluation pipeline completed: duration_seconds=%.2f",
        perf_counter() - start,
    )
