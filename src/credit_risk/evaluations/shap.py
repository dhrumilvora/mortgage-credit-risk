"""Model-agnostic SHAP analysis for the credit-risk modelling pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

from credit_risk.modelling.preprocessing import split_features_target
from credit_risk.utils.config import create_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


def _get_shap_config(config: dict) -> dict:
    """Return SHAP configuration with sensible defaults."""

    defaults = {
        "enabled": True,
        "background_size": 1000,
        "max_samples": 5000,
        "random_state": 42,
    }
    parameters = config["parameters"]
    return parameters.get("evaluation", {}).get(
        "shap", parameters.get("shap", defaults)
    )


def _load_shap() -> ModuleType:
    """Import the optional SHAP dependency only when analysis is enabled."""

    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP analysis requires the optional 'shap' dependency. "
            "Install the project dependencies before enabling evaluation.shap."
        ) from exc

    return shap


# ---------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------


def _get_transformed_feature_names(
    preprocessor,
) -> list[str]:
    """
    Return feature names produced by the fitted preprocessor.

    The model receives these transformed features rather than the
    original raw modelling columns.
    """

    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError as exc:
        raise ValueError(
            "The fitted preprocessor does not expose " "get_feature_names_out()."
        ) from exc

    return [str(name) for name in feature_names]


# ---------------------------------------------------------------------
# Prediction wrapper
# ---------------------------------------------------------------------


def _build_prediction_function(model):
    """
    Build a generic prediction function for an already-transformed
    model input.

    The function intentionally uses predict_proba rather than any
    model-specific SHAP explainer so the evaluator remains model
    agnostic.
    """

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            f"Model of type {type(model).__name__} does not expose " "predict_proba()."
        )

    def predict_probability(X_transformed):
        probabilities = model.predict_proba(X_transformed)

        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            raise ValueError(
                "Expected predict_proba() to return a 2D array with "
                "at least two probability columns."
            )

        return probabilities[:, 1]

    return predict_probability


# ---------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------


def compute_shap_values(
    model,
    preprocessor,
    X: pd.DataFrame,
    background_size: int = 1000,
    max_samples: int = 5000,
    random_state: int = 42,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Compute SHAP values on transformed model inputs.

    Parameters
    ----------
    model:
        Fitted classification model.

    preprocessor:
        Fitted sklearn preprocessor used by the model.

    X:
        Raw modelling features.

    background_size:
        Number of transformed observations used as SHAP background.

    max_samples:
        Maximum number of observations for which SHAP values are
        calculated.

    random_state:
        Random seed used when sampling observations.

    Returns
    -------
    shap_values:
        Array with shape (n_samples, n_transformed_features).

    X_transformed:
        Transformed feature matrix used for SHAP.
    """

    if X.empty:
        raise ValueError("Cannot compute SHAP values on an empty dataset.")

    if background_size < 1:
        raise ValueError("SHAP background_size must be at least 1.")

    if max_samples < 1:
        raise ValueError("SHAP max_samples must be at least 1.")

    rng = np.random.default_rng(random_state)

    # --------------------------------------------------------------
    # Transform using the already-fitted preprocessing pipeline.
    # --------------------------------------------------------------

    X_transformed = preprocessor.transform(X)

    # Convert sparse output to dense because SHAP's generic
    # permutation/exact explainers work more reliably with dense
    # matrices and because we need the transformed feature values
    # for the output artifact.
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    X_transformed = np.asarray(X_transformed)

    if X_transformed.ndim != 2:
        raise ValueError("Expected transformed feature matrix to be 2-dimensional.")

    n_samples = X_transformed.shape[0]

    # --------------------------------------------------------------
    # Sample rows for SHAP analysis.
    # --------------------------------------------------------------

    if n_samples > max_samples:
        sample_indices = rng.choice(
            n_samples,
            size=max_samples,
            replace=False,
        )
        sample_indices = np.sort(sample_indices)
    else:
        sample_indices = np.arange(n_samples)

    X_explain = X_transformed[sample_indices]

    # --------------------------------------------------------------
    # Build background sample.
    # --------------------------------------------------------------

    if n_samples > background_size:
        background_indices = rng.choice(
            n_samples,
            size=background_size,
            replace=False,
        )
        background_indices = np.sort(background_indices)
    else:
        background_indices = np.arange(n_samples)

    background = X_transformed[background_indices]

    # --------------------------------------------------------------
    # Generic prediction function.
    # --------------------------------------------------------------

    predict_probability = _build_prediction_function(model)

    # --------------------------------------------------------------
    # Use TreeExplainer for XGBoost. XGBoost models containing categorical
    # splits cannot be explained in SHAP's probability mode; the supported
    # tree-path-dependent mode produces exact contributions on the raw-margin
    # (log-odds) scale. The generic permutation explainer requires many model
    # evaluations and is impractical for the configured XGBoost analysis size.
    # Other classifiers retain the model-agnostic probability explanation path.
    # --------------------------------------------------------------

    shap_module = _load_shap()

    if model.__class__.__module__.startswith("xgboost"):
        explainer = shap_module.TreeExplainer(
            model,
            feature_perturbation="tree_path_dependent",
            model_output="raw",
        )
    else:
        explainer = shap_module.Explainer(
            predict_probability,
            background,
            algorithm="permutation",
        )

    shap_result = explainer(X_explain)

    shap_values = np.asarray(shap_result.values)

    # Some explainers may return an additional output dimension.
    # We expect a single scalar prediction: probability of event.
    if shap_values.ndim == 3:
        if shap_values.shape[-1] != 1:
            raise ValueError("Expected a single-output SHAP explanation.")

        shap_values = shap_values[:, :, 0]

    if shap_values.ndim != 2:
        raise ValueError("Unexpected SHAP output shape: " f"{shap_values.shape}")

    logger.info(
        "SHAP values computed: samples=%d features=%d",
        shap_values.shape[0],
        shap_values.shape[1],
    )

    return shap_values, pd.DataFrame(
        X_explain,
        columns=_get_transformed_feature_names(preprocessor),
    )


# ---------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------


def build_shap_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Build global SHAP feature importance.

    Importance is mean absolute SHAP value.
    """

    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            "Number of SHAP columns does not match feature names: "
            f"{shap_values.shape[1]} vs {len(feature_names)}"
        )

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.mean(
                np.abs(shap_values),
                axis=0,
            ),
            "mean_shap": np.mean(
                shap_values,
                axis=0,
            ),
        }
    )

    total_mean_abs_shap = importance["mean_abs_shap"].sum()
    importance["mean_abs_shap_pct"] = (
        importance["mean_abs_shap"] / total_mean_abs_shap
        if total_mean_abs_shap > 0
        else 0.0
    )

    importance = importance.sort_values(
        "mean_abs_shap",
        ascending=False,
    ).reset_index(drop=True)

    importance.insert(
        0,
        "rank",
        np.arange(1, len(importance) + 1),
    )

    return importance


# ---------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------


def _get_shap_dir(
    config: dict,
    dataset_name: str,
) -> Path:
    """Resolve the SHAP artifact directory."""
    approach = config["parameters"]["modelling_approach"]
    if dataset_name not in {"validation", "oot"}:
        raise ValueError(f"Unsupported SHAP dataset: {dataset_name}")

    parameters = config["parameters"]
    evaluation_config = parameters["evaluation"]

    if evaluation_config["mode"] == "same_run":
        model_version = parameters["modelling"]["version"]
        model_type = parameters["modelling"]["algorithm"]
    elif evaluation_config["mode"] == "existing_model":
        model_version = evaluation_config["model"]["version"]
        model_type = evaluation_config["model"]["type"]
    else:
        raise ValueError(f"Unsupported evaluation mode: {evaluation_config['mode']}")

    evaluation_root = create_path(
        config["catalog"]["base"],
        config["catalog"],
        "model_evaluation",
        approach,
        model_version,
        model_type,
        must_exist=False,
    )

    shap_dir = evaluation_root / "shap" / dataset_name

    shap_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return shap_dir


# ---------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------


def save_shap_results(
    shap_values: np.ndarray,
    X_transformed: pd.DataFrame,
    importance: pd.DataFrame,
    config: dict,
    dataset_name: str,
) -> None:
    """Persist SHAP values and feature importance artifacts."""

    shap_dir = _get_shap_dir(
        config=config,
        dataset_name=dataset_name,
    )

    # --------------------------------------------------------------
    # SHAP values
    # --------------------------------------------------------------

    shap_values_df = pd.DataFrame(
        shap_values,
        columns=X_transformed.columns,
    )

    shap_values_df.to_parquet(
        shap_dir / "shap_values.parquet",
        index=False,
    )

    # --------------------------------------------------------------
    # Transformed feature values
    # --------------------------------------------------------------

    X_transformed.to_parquet(
        shap_dir / "transformed_features.parquet",
        index=False,
    )

    # --------------------------------------------------------------
    # Global importance
    # --------------------------------------------------------------

    importance.to_csv(
        shap_dir / "feature_importance.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Metadata
    # --------------------------------------------------------------

    metadata = {
        "dataset": dataset_name,
        "n_samples": int(shap_values.shape[0]),
        "n_features": int(shap_values.shape[1]),
        "top_features": importance.head(20)[
            [
                "rank",
                "feature",
                "mean_abs_shap",
                "mean_shap",
                "mean_abs_shap_pct",
            ]
        ].to_dict(orient="records"),
    }

    with open(
        shap_dir / "metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    logger.info(
        "SHAP artifacts saved: dataset=%s path=%s",
        dataset_name,
        shap_dir,
    )


# ---------------------------------------------------------------------
# Public evaluation function
# ---------------------------------------------------------------------


def evaluate_shap(
    model,
    preprocessor,
    df: pd.DataFrame,
    config: dict,
    dataset_name: str,
) -> dict:
    """
    Run SHAP analysis for one evaluation dataset.

    The raw dataset is passed through the same fitted preprocessor
    used by the model. For XGBoost, SHAP values explain the raw model
    margin (log-odds); other supported classifiers use event probability.
    """

    shap_config = _get_shap_config(config)

    if not shap_config.get("enabled", True):
        logger.info(
            "SHAP evaluation disabled: dataset=%s",
            dataset_name,
        )
        return {}

    X, _ = split_features_target(
        df,
        config,
    )

    shap_values, X_transformed = compute_shap_values(
        model=model,
        preprocessor=preprocessor,
        X=X,
        background_size=shap_config.get(
            "background_size",
            1000,
        ),
        max_samples=shap_config.get(
            "max_samples",
            5000,
        ),
        random_state=shap_config.get(
            "random_state",
            42,
        ),
    )

    feature_names = list(X_transformed.columns)

    importance = build_shap_importance(
        shap_values=shap_values,
        feature_names=feature_names,
    )

    save_shap_results(
        shap_values=shap_values,
        X_transformed=X_transformed,
        importance=importance,
        config=config,
        dataset_name=dataset_name,
    )

    return {
        "dataset": dataset_name,
        "n_samples": int(shap_values.shape[0]),
        "n_features": int(shap_values.shape[1]),
        "feature_importance": importance,
    }
