from __future__ import annotations

import logging
from copy import deepcopy
from time import perf_counter

import numpy as np
import pandas as pd
from pyspark.ml.functions import vector_to_array

from credit_risk.evaluations.evaluations import (
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
from credit_risk.evaluations.calibration import (
    apply_calibration,
    calculate_calibration_summary,
    fit_calibration,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Evaluation population aggregation
# ---------------------------------------------------------------------


def aggregate_evaluation_predictions(
    predictions: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Aggregate model predictions to the configured evaluation grain.

    Point-in-time evaluation:
        One row per loan_id × observation_age.

    Loan-level evaluation:
        One row per loan_id using configurable aggregations for the
        target and predicted probability.

    The underlying modelling population is not changed.
    """

    evaluation_config = config["parameters"]["evaluation"]
    grain = evaluation_config.get("grain", "point_in_time")
    
    if grain == "point_in_time":
        return predictions

    if grain != "loan_level":
        raise ValueError(
            "Unsupported evaluation grain: "
            f"{grain}. Expected 'point_in_time' or 'loan_level'."
        )

    loan_level_config = evaluation_config["loan_level"]

    probability_aggregation = loan_level_config.get(
        "probability_aggregation",
        "max",
    )
    target_aggregation = loan_level_config.get(
        "target_aggregation",
        "max",
    )

    required_columns = {
        "loan_id",
        "target",
        "__prediction_probability",
    }

    missing_columns = sorted(
        required_columns - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Loan-level evaluation requires columns: "
            + ", ".join(missing_columns)
        )

    return (
        predictions
        .groupby("loan_id", as_index=False)
        .agg(
            target=("target", target_aggregation),
            __prediction_probability=(
                "__prediction_probability",
                probability_aggregation,
            ),
        )
    )


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

    Evaluation can be performed at either:
        - point_in_time: loan_id × observation_age
        - loan_level: one row per loan_id

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

        prediction_pd = pd.DataFrame(
            {
                "target": y.to_numpy(),
                "__prediction_probability": np.asarray(
                    y_proba,
                    dtype=float,
                ),
                "prediction": np.asarray(
                    y_pred,
                    dtype=int,
                ),
            }
        )

        # Retain identifiers for loan-level aggregation.
        if "loan_id" in df.columns:
            prediction_pd["loan_id"] = df["loan_id"].to_numpy()

    # -----------------------------------------------------------------
    # PySpark
    # -----------------------------------------------------------------

    elif engine == "pyspark":

        X, y = split_features_target_spark(
            df,
            config,
        )

        # GAM already contains its complete Spark preprocessing
        # pipeline. Do not apply the outer preprocessor again.
        X_transformed = (
            X
            if config["parameters"]["modelling"]["algorithm"] == "gam"
            else preprocessor.transform(X)
        )

        predictions = model.transform(
            X_transformed,
        )

        predictions = predictions.withColumn(
            "__prediction_probability",
            vector_to_array(
                "probability",
            )[1],
        )

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
        # Collect only prediction-related columns.
        # -------------------------------------------------------------

        prediction_columns = [
            "prediction",
            "__prediction_probability",
        ]

        if "loan_id" in predictions.columns:
            prediction_columns.append("loan_id")

        prediction_pd = (
            predictions
            .select(*prediction_columns)
            .toPandas()
        )

        y_pred = prediction_pd["prediction"].to_numpy().astype(int)

        y_proba = (
            prediction_pd["__prediction_probability"]
            .to_numpy()
            .astype(float)
        )

        # -------------------------------------------------------------
        # Collect target.
        # -------------------------------------------------------------

        target = config["parameters"]["target"]["name"]

        y = (
            y
            .select(target)
            .toPandas()
            .iloc[:, 0]
        )

        if len(y) != len(y_pred):
            raise ValueError(
                "Spark target and prediction outputs contain "
                "different numbers of rows: "
                f"target={len(y):,}, "
                f"predictions={len(y_pred):,}"
            )

        prediction_pd["target"] = y.to_numpy()

    else:

        raise ValueError(
            f"Unsupported evaluation engine: {engine}"
        )

    # -----------------------------------------------------------------
    # Common prediction validation
    # -----------------------------------------------------------------

    if len(prediction_pd) == 0:
        raise ValueError(
            "Evaluation dataset is empty."
        )

    if not np.isfinite(
        prediction_pd["__prediction_probability"]
    ).all():
        raise ValueError(
            "Predicted probabilities contain non-finite values."
        )

    if (
        (prediction_pd["__prediction_probability"] < 0)
        | (prediction_pd["__prediction_probability"] > 1)
    ).any():
        raise ValueError(
            "Predicted probabilities must be between 0 and 1."
        )

    # -----------------------------------------------------------------
    # Apply configured evaluation grain
    # -----------------------------------------------------------------

    evaluation_predictions = aggregate_evaluation_predictions(
        predictions=prediction_pd,
        config=config,
    )

    y = evaluation_predictions["target"].to_numpy()

    y_proba = (
        evaluation_predictions[
            "__prediction_probability"
        ]
        .to_numpy()
        .astype(float)
    )

    # Recompute classification prediction after aggregation.
    y_pred = (
        y_proba >= threshold
    ).astype(int)

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

    # Record evaluation grain in the results.
    evaluation_results["evaluation_grain"] = (
        evaluation_config.get(
            "grain",
            "point_in_time",
        )
    )

    if return_predictions:

        return (
            evaluation_results,
            pd.Series(y),
            y_proba,
        )

    return evaluation_results


# ---------------------------------------------------------------------
# Calibration metadata
# ---------------------------------------------------------------------


def _get_calibration_model_metadata(
    calibration_model,
    config: dict,
) -> dict:
    """Return serializable metadata for the fitted calibration model."""

    method = (
        config["parameters"]["evaluation"]["calibration"]["method"]
        .strip()
        .lower()
    )

    return calculate_calibration_summary(
        calibration_model=calibration_model,
        method=method,
    )


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

    threshold_config = (
        config["parameters"]["evaluation"]["threshold_selection"]
    )

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
        raise ValueError(
            "Threshold evaluation returned no results."
        )

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
            "Unsupported threshold optimization metric: "
            f"{optimization_metric}"
        )

    best_threshold_row = (
        threshold_results
        .sort_values(
            by=[
                optimization_metric,
                "precision",
                "recall",
            ],
            ascending=False,
        )
        .iloc[0]
    )

    selected_threshold = float(
        best_threshold_row["threshold"]
    )

    threshold_summary = {
        "optimization_metric": optimization_metric,
        "selected_threshold": selected_threshold,
        "validation_precision": float(
            best_threshold_row["precision"]
        ),
        "validation_recall": float(
            best_threshold_row["recall"]
        ),
        "validation_f1": float(
            best_threshold_row["f1"]
        ),
        "population_flagged": int(
            best_threshold_row["population_flagged"]
        ),
        "population_flagged_pct": float(
            best_threshold_row["population_flagged_pct"]
        ),
        "event_capture_rate": float(
            best_threshold_row["event_capture_rate"]
        ),
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


# ---------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------


def _resolve_model_configuration(
    config: dict,
) -> tuple[str, dict, dict]:
    """
    Resolve model configuration for the configured evaluation mode.
    """

    evaluation_config = config["parameters"]["evaluation"]

    mode = evaluation_config["mode"]

    if mode == "same_run":

        model_config = config

        scoring_config = deepcopy(config)

        return (
            mode,
            model_config,
            scoring_config,
        )

    if mode == "existing_model":

        model_config = deepcopy(config)

        model_config["parameters"]["modelling"]["version"] = (
            evaluation_config["model"]["version"]
        )

        model_config["parameters"]["modelling"]["algorithm"] = (
            evaluation_config["model"]["type"]
        )

        engine = config["parameters"]["engine"]

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

            raise ValueError(
                f"Unsupported modelling engine: {engine}"
            )

        scoring_config = deepcopy(config)

        scoring_config["parameters"]["modelling"]["features"] = (
            training_config["features"]
        )

        return (
            mode,
            model_config,
            scoring_config,
        )

    raise ValueError(
        f"Unsupported evaluation mode: {mode}"
    )


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

    Prediction generation is engine-specific.

    Evaluation grain is configuration-driven:
        point_in_time
        loan_level

    All downstream evaluation, threshold selection, calibration,
    top-k metrics, and reporting use the selected evaluation grain.
    """

    start = perf_counter()

    approach = config["parameters"]["modelling_approach"]

    evaluation_config = config["parameters"]["evaluation"]

    modelling_config = config["parameters"]["modelling"]

    engine = config["parameters"]["engine"]

    if evaluation_config["skip"]:

        logger.info(
            "Evaluation pipeline skipped by configuration"
        )

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
                "SparkSession is required when "
                "modelling engine is 'pyspark'."
            )

        from credit_risk.modelling.artifacts_spark import (
            load_spark_model_artifacts,
        )

        model, preprocessor = load_spark_model_artifacts(
            model_config,
        )

    else:

        raise ValueError(
            f"Unsupported evaluation engine: {engine}"
        )

    logger.info(
        "Evaluation model loaded: "
        "engine=%s mode=%s version=%s algorithm=%s "
        "grain=%s",
        engine,
        mode,
        model_config["parameters"]["modelling"]["version"],
        model_config["parameters"]["modelling"]["algorithm"],
        evaluation_config.get(
            "grain",
            "point_in_time",
        ),
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
        # Calibration
        # -------------------------------------------------------------

        calibration_model = fit_calibration(
            y_true=y_validation,
            y_proba=y_validation_proba,
            config=scoring_config,
        )

        calibration_model_metadata = (
            _get_calibration_model_metadata(
                calibration_model=calibration_model,
                config=scoring_config,
            )
        )

        calibration_model_metadata["method"] = (
            calibration_model_metadata[
                "calibration_method"
            ]
        )

        y_val_calibrated_proba = apply_calibration(
            y_proba=y_validation_proba,
            calibration_model=calibration_model,
            config=scoring_config,
        )

        validation_evaluation["calibration_model"] = (
            calibration_model_metadata
        )

        validation_evaluation["calibration_summary"] = (
            calibration_model_metadata
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
            y_validation_proba=y_val_calibrated_proba,
            config=scoring_config,
        )

        if selected_threshold is not None:

            scoring_config["parameters"]["evaluation"][
                "classification"
            ]["threshold"] = selected_threshold

            validation_evaluation["threshold_selection"] = (
                threshold_summary
            )

            threshold_summary_path = _get_evaluation_dir(
                config,
                "validation",
            )

            threshold_results.to_csv(
                threshold_summary_path
                / "threshold_summary.csv",
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

        logger.info(
            "Validation evaluation completed: grain=%s",
            evaluation_config.get(
                "grain",
                "point_in_time",
            ),
        )

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

            oot_evaluation["threshold_applied"] = (
                selected_threshold
            )

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
                scoring_config["parameters"]["evaluation"][
                    "classification"
                ]["threshold"]
            )

            y_oot_calibrated_pred = (
                y_oot_calibrated_proba >= raw_threshold
            ).astype("int8")

            calibrated_oot_evaluation = evaluate_dataset(
                y_true=y_oot,
                y_pred=y_oot_calibrated_pred,
                y_proba=y_oot_calibrated_proba,
                n_deciles=(
                    scoring_config["parameters"]["evaluation"][
                        "risk"
                    ]["n_deciles"]
                ),
                calibration_bins=(
                    scoring_config["parameters"]["evaluation"][
                        "calibration"
                    ]["bins"]
                ),
            )

            calibrated_oot_evaluation[
                "calibration_applied"
            ] = {
                **calibration_model_metadata,
                "raw_threshold": raw_threshold,
            }

            calibrated_oot_evaluation[
                "calibration_summary"
            ] = calculate_calibration_summary(
                calibration_model=calibration_model,
                method=scoring_config["parameters"][
                    "evaluation"
                ]["calibration"]["method"],
            )

            calibrated_oot_evaluation[
                "threshold_applied"
            ] = raw_threshold

            calibrated_oot_evaluation[
                "evaluation_grain"
            ] = evaluation_config.get(
                "grain",
                "point_in_time",
            )

        logger.info(
            "OOT evaluation completed: grain=%s",
            evaluation_config.get(
                "grain",
                "point_in_time",
            ),
        )

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
        "Evaluation pipeline completed: "
        "duration_seconds=%.2f",
        perf_counter() - start,
    )