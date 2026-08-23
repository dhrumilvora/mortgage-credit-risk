from __future__ import annotations

import logging
from copy import deepcopy
from time import perf_counter

import pandas as pd

from credit_risk.evaluations.evaluations import (
    apply_calibration,
    calculate_top_k_metrics,
    evaluate_dataset,
    evaluate_thresholds,
    fit_calibration,
    generate_predictions,
)
from credit_risk.evaluations.reporting import (
    _get_evaluation_dir,
    save_evaluation_results,
)
from credit_risk.evaluations.shap import evaluate_shap
from credit_risk.modelling.artifacts import (
    load_model_artifacts,
    load_training_config,
)
from credit_risk.modelling.preprocessing import (
    split_features_target,
)
from credit_risk.utils.config import create_path

logger = logging.getLogger(__name__)


def evaluate_split(
    model,
    preprocessor,
    df: pd.DataFrame,
    config: dict,
    return_predictions: bool = False,
):
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

    evaluation_results = evaluate_dataset(
        y_true=y,
        y_pred=y_pred,
        y_proba=y_proba,
        n_deciles=evaluation_config["risk"]["n_deciles"],
        calibration_bins=evaluation_config["calibration"]["bins"],
    )

    evaluation_results["top_k_metrics"] = calculate_top_k_metrics(
        y_true=y,
        y_pred=y_proba,
    ).to_dict(orient="records")

    if return_predictions:
        return (
            evaluation_results,
            y,
            y_proba,
        )

    return evaluation_results


def _select_validation_threshold(
    y_validation: pd.Series,
    y_validation_proba,
    config: dict,
) -> tuple[float | None, pd.DataFrame | None, dict | None]:
    """
    Evaluate configured thresholds on validation data and select
    the threshold using the configured optimization metric.

    OOT data is never used for threshold selection.
    """

    threshold_config = config["parameters"]["evaluation"]["threshold_selection"]

    if not threshold_config["enabled"]:
        return None, None, None

    threshold_results = evaluate_thresholds(
        y_true=y_validation,
        y_proba=y_validation_proba,
        config=config,
    )

    if threshold_results.empty:
        raise ValueError("Threshold evaluation returned no results.")

    optimization_metric = threshold_config.get(
        "optimization_metric",
        threshold_config.get("optimisation_metric"),
    )

    if not optimization_metric:
        raise ValueError(
            "Missing threshold optimization metric. "
            "Expected 'optimization_metric' under "
            "evaluation.threshold_selection."
        )

    if optimization_metric not in threshold_results.columns:
        raise ValueError(
            "Unsupported threshold optimization metric: " f"{optimization_metric}"
        )

    best_threshold_row = threshold_results.sort_values(
        by=[
            optimization_metric,
            "precision",
            "recall",
        ],
        ascending=False,
    ).iloc[0]

    selected_threshold = float(best_threshold_row["threshold"])

    threshold_summary = {
        "optimization_metric": optimization_metric,
        "selected_threshold": selected_threshold,
        "validation_precision": float(best_threshold_row["precision"]),
        "validation_recall": float(best_threshold_row["recall"]),
        "validation_f1": float(best_threshold_row["f1"]),
        "population_flagged": int(best_threshold_row["population_flagged"]),
        "population_flagged_pct": float(best_threshold_row["population_flagged_pct"]),
        "event_capture_rate": float(best_threshold_row["event_capture_rate"]),
    }
    logger.info(f"Selected Threshold: {selected_threshold}")
    return (
        selected_threshold,
        threshold_results,
        threshold_summary,
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

        # Keep scoring configuration separate so threshold selection
        # does not mutate the master configuration.
        scoring_config = deepcopy(config)

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

        # Existing model must always be scored with the exact feature
        # definition used during model training.
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
        "Evaluation model loaded: " "mode=%s version=%s algorithm=%s",
        mode,
        model_config["parameters"]["modelling"]["version"],
        model_config["parameters"]["modelling"]["algorithm"],
    )

    validation_evaluation = None
    oot_evaluation = None
    calibrated_oot_evaluation = None

    calibration_intercept = None
    calibration_slope = None

    selected_threshold = None
    threshold_results = None
    threshold_summary = None

    # --------------------------------------------------------------
    # Validation evaluation
    # --------------------------------------------------------------

    if evaluation_config["datasets"]["validation"]:

        validation_path = create_path(
            config["catalog"]["base"],
            config["catalog"],
            "validation_df",
            approach,
        )

        validation_df = pd.read_parquet(validation_path)

        (
            validation_evaluation,
            y_validation,
            y_validation_proba,
        ) = evaluate_split(
            model=model,
            preprocessor=preprocessor,
            df=validation_df,
            config=scoring_config,
            return_predictions=True,
        )

        # ----------------------------------------------------------
        # Threshold selection
        #
        # Threshold selection is performed ONLY on validation.
        # ----------------------------------------------------------

        (
            selected_threshold,
            threshold_results,
            threshold_summary,
        ) = _select_validation_threshold(
            y_validation=y_validation,
            y_validation_proba=y_validation_proba,
            config=scoring_config,
        )

        if selected_threshold is not None:

            scoring_config["parameters"]["evaluation"]["classification"][
                "threshold"
            ] = selected_threshold

            validation_evaluation["threshold_selection"] = threshold_summary

            threshold_summary_path = _get_evaluation_dir(
                config,
                "validation",
            )

            threshold_results.to_csv(
                threshold_summary_path / "threshold_summary.csv",
                index=False,
            )

            logger.info(
                "Validation threshold selected: "
                "threshold=%.6f metric=%s "
                "precision=%.6f recall=%.6f f1=%.6f "
                "population_flagged_pct=%.6f",
                threshold_summary["selected_threshold"],
                threshold_summary["optimization_metric"],
                threshold_summary["validation_precision"],
                threshold_summary["validation_recall"],
                threshold_summary["validation_f1"],
                threshold_summary["population_flagged_pct"],
            )

        # ----------------------------------------------------------
        # Calibration
        #
        # Calibration is fitted ONLY on validation.
        # It remains a separate diagnostic from threshold selection.
        # ----------------------------------------------------------

        calibration_intercept, calibration_slope = fit_calibration(
            y_true=y_validation,
            y_proba=y_validation_proba,
        )

        logger.info(
            "Calibration fitted on validation: " "intercept=%.6f slope=%.6f",
            calibration_intercept,
            calibration_slope,
        )
        if config["parameters"]["evaluation"]["shap"]["enabled"]:
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
            config["catalog"]["base"],
            config["catalog"],
            "oot_df",
            approach,
        )

        oot_df = pd.read_parquet(oot_path)

        # ----------------------------------------------------------
        # Raw OOT evaluation using the threshold selected from
        # validation.
        #
        # scoring_config now contains the frozen validation threshold.
        # ----------------------------------------------------------

        (
            oot_evaluation,
            y_oot,
            y_oot_proba,
        ) = evaluate_split(
            model=model,
            preprocessor=preprocessor,
            df=oot_df,
            config=scoring_config,
            return_predictions=True,
        )

        if selected_threshold is not None:
            oot_evaluation["threshold_applied"] = selected_threshold

        # ----------------------------------------------------------
        # Apply validation-fitted calibration to OOT.
        #
        # Calibration does NOT affect ranking and is kept separate
        # from the raw threshold-based OOT evaluation.
        # ----------------------------------------------------------

        if calibration_intercept is not None and calibration_slope is not None:

            y_oot_calibrated_proba = apply_calibration(
                y_proba=y_oot_proba,
                intercept=calibration_intercept,
                slope=calibration_slope,
            )

            # Apply the same threshold selected from validation.
            if selected_threshold is not None:
                calibrated_threshold = selected_threshold
            else:
                calibrated_threshold = scoring_config["parameters"]["evaluation"][
                    "classification"
                ]["threshold"]

            y_oot_calibrated_pred = (
                y_oot_calibrated_proba >= calibrated_threshold
            ).astype("int8")

            calibrated_oot_evaluation = evaluate_dataset(
                y_true=y_oot,
                y_pred=y_oot_calibrated_pred,
                y_proba=y_oot_calibrated_proba,
                n_deciles=(
                    scoring_config["parameters"]["evaluation"]["risk"]["n_deciles"]
                ),
                calibration_bins=(
                    scoring_config["parameters"]["evaluation"]["calibration"]["bins"]
                ),
            )

            calibrated_oot_evaluation["top_k_metrics"] = calculate_top_k_metrics(
                y_true=y_oot,
                y_pred=y_oot_calibrated_proba,
            ).to_dict(orient="records")

            calibrated_oot_evaluation["calibration_applied"] = {
                "intercept": calibration_intercept,
                "slope": calibration_slope,
            }

            calibrated_oot_evaluation["threshold_applied"] = calibrated_threshold
        if config["parameters"]["evaluation"]["shap"]["enabled"]:
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
        oot_calibration=calibrated_oot_evaluation,
        config=config,
    )

    logger.info(
        "Evaluation pipeline completed: duration_seconds=%.2f",
        perf_counter() - start,
    )
