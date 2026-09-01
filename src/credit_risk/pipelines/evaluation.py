from __future__ import annotations

import logging
from copy import deepcopy
from time import perf_counter

import numpy as np
import pandas as pd
from pyspark.ml.functions import vector_to_array

from credit_risk.evaluations.evaluations import (
    calculate_top_k_metrics,
    evaluate_dataset,
    evaluate_thresholds,
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
from credit_risk.modelling.preprocessing_spark import (
    split_features_target_spark,
)
from credit_risk.utils.config import create_path
from credit_risk.evaluations.calibration import apply_calibration, fit_calibration

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Split evaluation
# ---------------------------------------------------------------------


def evaluate_split(
    model,
    preprocessor,
    df,
    config: dict,
    return_predictions: bool = False,
    engine: str = "pandas",
):
    """
    Generate predictions and evaluate a single dataset split.

    Prediction generation is engine-specific.

    All evaluation metrics remain Pandas-based so that the Pandas
    and PySpark engines use the same evaluation methodology.
    """

    evaluation_config = config["parameters"]["evaluation"]

    threshold = evaluation_config["classification"]["threshold"]

    # -----------------------------------------------------------------
    # Pandas
    # -----------------------------------------------------------------

    if engine == "pandas":

        X, y = split_features_target(
            df,
            config,
        )

        y_pred, y_proba = generate_predictions(
            model=model,
            preprocessor=preprocessor,
            X=X,
            threshold=threshold,
        )

    # -----------------------------------------------------------------
    # PySpark
    # -----------------------------------------------------------------

    elif engine == "pyspark":

        X, y = split_features_target_spark(
            df,
            config,
        )

        # -------------------------------------------------------------
        # Apply the fitted Spark preprocessing pipeline.
        #
        # IMPORTANT:
        # preprocessor.transform() returns the Spark ML feature vector
        # in the "features" column.
        # -------------------------------------------------------------

        X_transformed = preprocessor.transform(
            X,
        )

        # -------------------------------------------------------------
        # Generate Spark predictions.
        #
        # Spark ML binary classifiers generally produce:
        #
        #     prediction
        #     probability
        #
        # where probability is a Spark Vector UDT.
        # -------------------------------------------------------------

        predictions = model.transform(
            X_transformed,
        )

        # -------------------------------------------------------------
        # Convert Spark ML Vector -> Spark Array.
        #
        # probability is NOT a normal Spark array. Therefore:
        #
        #     probability[1]
        #
        # causes:
        #
        #     INVALID_EXTRACT_BASE_FIELD_TYPE
        #
        # vector_to_array() converts:
        #
        #     [P(0), P(1)]
        #
        # into an actual Spark array.
        # -------------------------------------------------------------

        predictions = predictions.withColumn(
            "__prediction_probability",
            vector_to_array(
                "probability",
            )[1],
        )

        # -------------------------------------------------------------
        # Validate prediction columns.
        # -------------------------------------------------------------

        required_prediction_columns = {
            "prediction",
            "__prediction_probability",
        }

        missing_prediction_columns = sorted(
            required_prediction_columns - set(predictions.columns)
        )

        if missing_prediction_columns:
            raise ValueError(
                "Spark model predictions are missing required columns: "
                + ", ".join(missing_prediction_columns)
            )

        # -------------------------------------------------------------
        # Collect ONLY predictions.
        #
        # We deliberately do not convert X_transformed to Pandas.
        #
        # The large feature matrix remains in Spark.
        # -------------------------------------------------------------

        prediction_pd = predictions.select(
            "prediction",
            "__prediction_probability",
        ).toPandas()

        y_pred = prediction_pd["prediction"].to_numpy().astype(int)

        y_proba = prediction_pd["__prediction_probability"].to_numpy().astype(float)

        # -------------------------------------------------------------
        # Collect ONLY the target.
        # -------------------------------------------------------------

        target = config["parameters"]["target"]["name"]

        y = (
            y.select(
                target,
            )
            .toPandas()
            .iloc[:, 0]
        )

        # -------------------------------------------------------------
        # Validate row alignment.
        # -------------------------------------------------------------

        if len(y) != len(y_pred):
            raise ValueError(
                "Spark target and prediction outputs contain "
                "different numbers of rows: "
                f"target={len(y):,}, "
                f"predictions={len(y_pred):,}"
            )

    else:

        raise ValueError(f"Unsupported evaluation engine: {engine}")

    # -----------------------------------------------------------------
    # Common prediction validation
    # -----------------------------------------------------------------

    if len(y) == 0:
        raise ValueError("Evaluation dataset is empty.")

    if len(y) != len(y_proba):
        raise ValueError(
            "Evaluation target and predicted probabilities contain "
            "different numbers of rows."
        )

    if len(y) != len(y_pred):
        raise ValueError(
            "Evaluation target and predictions contain " "different numbers of rows."
        )

    if not np.isfinite(y_proba).all():
        raise ValueError("Predicted probabilities contain non-finite values.")

    if ((y_proba < 0) | (y_proba > 1)).any():
        raise ValueError("Predicted probabilities must be between 0 and 1.")

    # -----------------------------------------------------------------
    # Shared Pandas evaluation
    # -----------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------


def _select_validation_threshold(
    y_validation: pd.Series,
    y_validation_proba,
    config: dict,
) -> tuple[
    float | None,
    pd.DataFrame | None,
    dict | None,
]:
    """
    Evaluate configured thresholds on validation data and select
    the threshold using the configured optimization metric.

    OOT data is never used for threshold selection.
    """

    threshold_config = config["parameters"]["evaluation"]["threshold_selection"]

    if not threshold_config["enabled"]:

        return (
            None,
            None,
            None,
        )

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

    logger.info(
        "Selected Threshold: %s",
        selected_threshold,
    )

    return (
        selected_threshold,
        threshold_results,
        threshold_summary,
    )


def _map_threshold_to_calibrated_scale(
    raw_threshold: float,
    calibration_model,
    config: dict,
) -> float:
    """Map a decision cutoff from raw to calibrated probability space.

    Threshold selection is performed on the model's raw validation
    probabilities.  Calibration changes the probability scale, so that same
    numeric cutoff must not be applied directly to calibrated probabilities.
    """

    calibrated_threshold = apply_calibration(
        y_proba=np.asarray([raw_threshold], dtype=float),
        calibration_model=calibration_model,
        config=config,
    )

    return float(calibrated_threshold[0])


# ---------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------


def _resolve_model_configuration(
    config: dict,
) -> tuple[str, dict, dict]:
    """
    Resolve model configuration for the configured evaluation mode.

    Returns
    -------
    tuple
        mode,
        model_config,
        scoring_config
    """

    evaluation_config = config["parameters"]["evaluation"]

    mode = evaluation_config["mode"]

    # -----------------------------------------------------------------
    # Same run
    # -----------------------------------------------------------------

    if mode == "same_run":

        model_config = config

        scoring_config = deepcopy(config)

        return (
            mode,
            model_config,
            scoring_config,
        )

    # -----------------------------------------------------------------
    # Existing model
    # -----------------------------------------------------------------

    if mode == "existing_model":

        model_config = deepcopy(config)

        model_config["parameters"]["modelling"]["version"] = evaluation_config["model"][
            "version"
        ]

        model_config["parameters"]["modelling"]["algorithm"] = evaluation_config[
            "model"
        ]["type"]

        engine = config["parameters"]["engine"]

        # -------------------------------------------------------------
        # Load persisted training configuration.
        # -------------------------------------------------------------

        if engine == "pyspark":

            from credit_risk.modelling.artifacts_spark import (
                load_training_config_spark,
            )

            training_config = load_training_config_spark(
                model_config,
            )

        elif engine == "pandas":

            training_config = load_training_config(
                model_config,
            )

        else:

            raise ValueError(f"Unsupported modelling engine: {engine}")

        scoring_config = deepcopy(config)

        scoring_config["parameters"]["modelling"]["features"] = training_config[
            "features"
        ]

        return (
            mode,
            model_config,
            scoring_config,
        )

    raise ValueError(f"Unsupported evaluation mode: {mode}")


# ---------------------------------------------------------------------
# Spark dataset loading
# ---------------------------------------------------------------------


def _load_spark_dataset(
    spark,
    path,
):
    """Load a persisted modelling dataset with Spark."""

    logger.info(
        "Spark reading evaluation dataset: %s",
        path,
    )

    return spark.read.parquet(
        str(path),
    )


# ---------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------


def run_evaluation_pipeline(
    config: dict,
    spark=None,
) -> None:
    """
    Run model evaluation for the configured datasets and persist results.

    Prediction generation is engine-specific:

        Pandas
            sklearn preprocessing + sklearn model

        PySpark
            Spark PipelineModel + Spark ML model

    All downstream evaluation, threshold selection, calibration,
    top-k metrics, and reporting remain shared Pandas logic.
    """

    start = perf_counter()

    approach = config["parameters"]["modelling_approach"]

    evaluation_config = config["parameters"]["evaluation"]

    modelling_config = config["parameters"]["modelling"]

    engine = config["parameters"]["engine"]

    # -----------------------------------------------------------------
    # Skip
    # -----------------------------------------------------------------

    if evaluation_config["skip"]:

        logger.info("Evaluation pipeline skipped by configuration")

        return

    # -----------------------------------------------------------------
    # Resolve model configuration
    # -----------------------------------------------------------------

    (
        mode,
        model_config,
        scoring_config,
    ) = _resolve_model_configuration(
        config,
    )

    # -----------------------------------------------------------------
    # Load model artifacts
    # -----------------------------------------------------------------

    if engine == "pandas":

        model, preprocessor = load_model_artifacts(
            model_config,
        )

    elif engine == "pyspark":

        if spark is None:

            raise ValueError(
                "SparkSession is required when " "modelling engine is 'pyspark'."
            )

        from credit_risk.modelling.artifacts_spark import (
            load_spark_model_artifacts,
        )

        model, preprocessor = load_spark_model_artifacts(
            model_config,
        )

    else:

        raise ValueError(f"Unsupported modelling engine: {engine}")

    logger.info(
        "Evaluation model loaded: " "engine=%s mode=%s version=%s algorithm=%s",
        engine,
        mode,
        model_config["parameters"]["modelling"]["version"],
        model_config["parameters"]["modelling"]["algorithm"],
    )

    # -----------------------------------------------------------------
    # Result containers
    # -----------------------------------------------------------------

    validation_evaluation = None
    oot_evaluation = None
    calibrated_oot_evaluation = None

    calibration_model = None

    selected_threshold = None
    threshold_results = None
    threshold_summary = None

    # -----------------------------------------------------------------
    # Validation evaluation
    # -----------------------------------------------------------------

    if evaluation_config["datasets"]["validation"]:

        validation_path = create_path(
            config["catalog"]["base"],
            config["catalog"],
            "validation_df",
            approach,
        )

        if engine == "pandas":

            validation_df = pd.read_parquet(
                validation_path,
            )

        else:

            validation_df = _load_spark_dataset(
                spark,
                validation_path,
            )

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
            engine=engine,
        )

        # -------------------------------------------------------------
        # Threshold selection
        # -------------------------------------------------------------

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

        # -------------------------------------------------------------
        # Calibration
        # -------------------------------------------------------------

        calibration_model = fit_calibration(
            y_true=y_validation, y_proba=y_validation_proba, config=scoring_config
        )

        # logger.info(
        #     "Calibration fitted on validation: " "intercept=%.6f slope=%.6f",
        #     calibration_model.intercept_[0],
        #     calibration_model.coef_[0][0],
        # )

        # -------------------------------------------------------------
        # SHAP
        # -------------------------------------------------------------

        if evaluation_config["shap"]["enabled"]:

            if engine == "pandas":

                evaluate_shap(
                    model=model,
                    preprocessor=preprocessor,
                    df=validation_df,
                    config=scoring_config,
                    dataset_name="validation",
                )

            else:

                logger.warning(
                    "SHAP evaluation skipped for PySpark engine. "
                    "The existing SHAP implementation is Pandas-based."
                )

        logger.info("Validation evaluation completed.")

    # -----------------------------------------------------------------
    # OOT evaluation
    # -----------------------------------------------------------------

    if evaluation_config["datasets"]["oot"]:

        oot_path = create_path(
            config["catalog"]["base"],
            config["catalog"],
            "oot_df",
            approach,
        )

        if engine == "pandas":

            oot_df = pd.read_parquet(
                oot_path,
            )

        else:

            oot_df = _load_spark_dataset(
                spark,
                oot_path,
            )

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
            engine=engine,
        )

        if selected_threshold is not None:

            oot_evaluation["threshold_applied"] = selected_threshold

        # -------------------------------------------------------------
        # Apply validation-fitted calibration to OOT.
        # -------------------------------------------------------------

        if calibration_model is not None:

            y_oot_calibrated_proba = apply_calibration(
                y_proba=y_oot_proba,
                calibration_model=calibration_model,
                config=scoring_config,
            )

            raw_threshold = (
                selected_threshold
                if selected_threshold is not None
                else scoring_config["parameters"]["evaluation"]["classification"][
                    "threshold"
                ]
            )

            calibrated_threshold = _map_threshold_to_calibrated_scale(
                raw_threshold=raw_threshold,
                calibration_model=calibration_model,
                config=scoring_config,
            )

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
                "method": scoring_config["parameters"]["evaluation"][
                    "calibration"
                ]["method"],
                "raw_threshold": raw_threshold,
                "calibrated_threshold": calibrated_threshold,
            }
            calibrated_oot_evaluation["threshold_applied"] = calibrated_threshold

        # -------------------------------------------------------------
        # SHAP
        # -------------------------------------------------------------

        if evaluation_config["shap"]["enabled"]:

            if engine == "pandas":

                evaluate_shap(
                    model=model,
                    preprocessor=preprocessor,
                    df=oot_df,
                    config=scoring_config,
                    dataset_name="oot",
                )

            else:

                logger.warning(
                    "SHAP evaluation skipped for PySpark engine. "
                    "The existing SHAP implementation is Pandas-based."
                )

        logger.info("OOT evaluation completed.")

    # -----------------------------------------------------------------
    # Persist evaluation results
    # -----------------------------------------------------------------

    save_evaluation_results(
        validation_evaluation=validation_evaluation,
        oot_evaluation=oot_evaluation,
        oot_calibration=calibrated_oot_evaluation,
        config=config,
    )

    logger.info(
        "Evaluation pipeline completed: " "duration_seconds=%.2f",
        perf_counter() - start,
    )
