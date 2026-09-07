"""Spark GAM V2 modelling utilities.

GAM V2 is an additive Generalized Additive Model with optional interactions.
The same pipeline can optionally be switched to a hierarchical GAM (HGAM):

    logit(PD) = beta_0 + sum_j f_j(X_j)

Selected numerical features are represented by cubic B-spline bases.
Remaining numerical features enter linearly. Categorical variables use
Spark StringIndexer/OneHotEncoder.

The complete model is one Spark PipelineModel. Preprocessing, fitted
spline knots, feature assembly, and logistic-regression coefficients are
therefore persisted with the PipelineModel.

V1 behavior is preserved exactly when interactions are disabled.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from pyspark import keyword_only
from pyspark.ml import Estimator, Pipeline, Transformer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler, VectorSizeHint
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.column import Column


# ======================================================================
# SPLINE SPECIFICATION
# ======================================================================


@dataclass
class GAMSplineSpec:
    """Fitted B-spline specification for one logical GAM feature."""

    feature: str
    degree: int
    knots: list[float]
    lower_bound: float
    upper_bound: float

@dataclass
class GAMInteractionSpec:
    left: str
    right: str
    interaction_type:str
    
def _resolve_interaction_type(
    left: str,
    right: str,
    numerical_features: set[str],
    categorical_features: set[str],
) -> str:

    left_type = "numeric" if left in numerical_features else "categorical"
    right_type = "numeric" if right in numerical_features else "categorical"

    if left_type == "categorical" and right_type == "categorical":
        raise ValueError(
            f"Categorical × categorical interactions are not supported: "
            f"'{left}' × '{right}'."
        )

    if left_type == "numeric" and right_type == "numeric":
        return "numeric_numeric"

    return "numeric_categorical"


def _get_gam_interaction_specs(
    config: dict,
) -> list[GAMInteractionSpec]:
    """Read and validate configured GAM interactions.

    Numeric features do not need to be configured as spline features.
    Their fitted representation is resolved later from the actual fitted
    spline basis. This means a numeric feature can participate in an
    interaction as linear or spline.
    """

    features = config["parameters"]["modelling"]["features"]
    numerical_features = set(features.get("numerical_features", []))
    categorical_features = set(features.get("categorical_features", []))
    all_features = numerical_features | categorical_features

    gam_config = config["parameters"]["modelling"]["gam"]
    interaction_config = gam_config.get("interactions", {})

    if not interaction_config.get("enabled", False):
        return []

    pairs = interaction_config.get("pairs", [])
    if not isinstance(pairs, list):
        raise ValueError("GAM interactions.pairs must be a list.")

    specs: list[GAMInteractionSpec] = []
    seen_pairs: set[frozenset[str]] = set()

    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(
                "Each GAM interaction must contain exactly two feature names."
            )

        left, right = pair

        if not isinstance(left, str) or not isinstance(right, str):
            raise ValueError("GAM interaction feature names must be strings.")

        if left == right:
            raise ValueError(
                f"Self-interaction is not supported: '{left}'."
            )

        unknown = sorted(
            {feature for feature in (left, right) if feature not in all_features}
        )
        if unknown:
            raise ValueError(
                "GAM interaction contains unknown features: "
                + ", ".join(unknown)
            )

        pair_key = frozenset((left, right))
        if pair_key in seen_pairs:
            raise ValueError(
                f"Duplicate GAM interaction: '{left}' × '{right}'."
            )
        seen_pairs.add(pair_key)

        interaction_type = _resolve_interaction_type(
            left=left,
            right=right,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        )

        specs.append(
            GAMInteractionSpec(
                left=left,
                right=right,
                interaction_type=interaction_type,
            )
        )

    return specs


def _bspline_interaction_expression(
    left: Column,
    right: Column,
    left_knots: list[float],
    right_knots: list[float],
    degree: int,
    left_basis_index: int,
    right_basis_index: int,
) -> Column:
    """Build one tensor-product B-spline interaction basis."""

    left_basis = _bspline_basis_expression(
        x=left,
        knots=left_knots,
        degree=degree,
        basis_index=left_basis_index,
    )

    right_basis = _bspline_basis_expression(
        x=right,
        knots=right_knots,
        degree=degree,
        basis_index=right_basis_index,
    )

    return left_basis * right_basis


def _encode_interaction_specs(
    specs: list[GAMInteractionSpec],
) -> str:
    """Serialize validated interaction specifications to JSON."""

    return json.dumps(
        [asdict(spec) for spec in specs],
        sort_keys=True,
    )


def _decode_interaction_specs(
    value: str,
) -> list[GAMInteractionSpec]:
    """Deserialize interaction specifications."""

    payload = json.loads(value)

    if not isinstance(payload, list):
        raise ValueError(
            "Serialized GAM interaction specifications must be a list."
        )

    specs: list[GAMInteractionSpec] = []

    for data in payload:

        if not isinstance(data, dict):
            raise ValueError(
                "Invalid GAM interaction specification."
            )

        required_fields = {
            "left",
            "right",
            "interaction_type",
        }

        missing = required_fields - set(data)

        if missing:
            raise ValueError(
                "GAM interaction specification is missing: "
                + ", ".join(sorted(missing))
            )

        specs.append(
            GAMInteractionSpec(
                left=str(data["left"]),
                right=str(data["right"]),
                interaction_type=str(data["interaction_type"]),
            )
        )

    return specs


class GAMInteractionEstimator(
    Estimator,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """Fit representation-aware GAM interaction transformations."""

    interactionSpecs = Param(
        Params._dummy(),
        "interactionSpecs",
        "JSON-encoded configured GAM interaction specifications.",
        TypeConverters.toString,
    )
    numericalFeatures = Param(
        Params._dummy(),
        "numericalFeatures",
        "Canonical numerical GAM features.",
        TypeConverters.toList,
    )
    categoricalFeatures = Param(
        Params._dummy(),
        "categoricalFeatures",
        "Canonical categorical GAM features.",
        TypeConverters.toList,
    )

    @keyword_only
    def __init__(
        self,
        interactionSpecs: str = "[]",
        numericalFeatures: list[str] | None = None,
        categoricalFeatures: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._setDefault(
            interactionSpecs="[]",
            numericalFeatures=[],
            categoricalFeatures=[],
        )
        self.setParams(
            interactionSpecs=interactionSpecs,
            numericalFeatures=numericalFeatures or [],
            categoricalFeatures=categoricalFeatures or [],
        )

    @keyword_only
    def setParams(
        self,
        interactionSpecs: str | None = None,
        numericalFeatures: list[str] | None = None,
        categoricalFeatures: list[str] | None = None,
    ) -> "GAMInteractionEstimator":
        return self._set(**self._input_kwargs)

    @staticmethod
    def _basis_columns(
        dataset: DataFrame,
        feature: str,
    ) -> list[str]:
        prefix = f"{feature}_spline_"
        columns = [
            column
            for column in dataset.columns
            if column.startswith(prefix)
            and column[len(prefix):].isdigit()
        ]
        return sorted(
            columns,
            key=lambda column: int(column.rsplit("_", 1)[1]),
        )

    @staticmethod
    def _encoded_dimension(
        dataset: DataFrame,
        feature: str,
    ) -> int:
        encoded = f"__encoded_{feature}"
        if encoded not in dataset.columns:
            raise ValueError(
                f"Expected encoded categorical column '{encoded}' not found."
            )

        field = dataset.schema[encoded]
        dimension = field.metadata.get("ml_attr", {}).get("num_attrs")

        if dimension is None or int(dimension) < 1:
            raise ValueError(
                f"Unable to determine encoded dimensionality for "
                f"'{feature}'."
            )

        return int(dimension)

    def _fit(self, dataset: DataFrame) -> "GAMInteractionModel":
        specs = _decode_interaction_specs(
            self.getOrDefault(self.interactionSpecs)
        )
        numerical_features = set(
            self.getOrDefault(self.numericalFeatures)
        )
        categorical_features = set(
            self.getOrDefault(self.categoricalFeatures)
        )

        if not specs:
            return GAMInteractionModel(
                interactionSpecs="[]",
                linearMeans="{}",
                splineMeans="{}",
                categoricalDimensions="{}",
                interactionSizes="{}",
            )

        basis_columns: dict[str, list[str]] = {}
        linear_features: set[str] = set()
        categorical_dimensions: dict[str, int] = {}
        resolved_specs: list[GAMInteractionSpec] = []
        interaction_sizes: dict[str, int] = {}

        def representation(feature: str) -> str:
            if feature in numerical_features:
                if self._basis_columns(dataset, feature):
                    return "spline"
                return "linear"

            if feature in categorical_features:
                return "categorical"

            raise ValueError(
                f"Unknown GAM interaction feature '{feature}'."
            )

        for spec in specs:
            left_rep = representation(spec.left)
            right_rep = representation(spec.right)

            if (
                left_rep == "categorical"
                and right_rep == "categorical"
            ):
                raise ValueError(
                    "Categorical × categorical interactions are not supported: "
                    f"'{spec.left}' × '{spec.right}'."
                )

            if (
                left_rep == "categorical"
                or right_rep == "categorical"
            ):
                if left_rep == "categorical":
                    categorical_feature = spec.left
                    numeric_feature = spec.right
                else:
                    categorical_feature = spec.right
                    numeric_feature = spec.left

                numeric_rep = (
                    right_rep if left_rep == "categorical" else left_rep
                )
                resolved_type = f"{numeric_rep}_categorical"

                categorical_dimensions[categorical_feature] = (
                    self._encoded_dimension(
                        dataset,
                        categorical_feature,
                    )
                )
            else:
                resolved_type = f"{left_rep}_{right_rep}"

            resolved_spec = GAMInteractionSpec(
                left=spec.left,
                right=spec.right,
                interaction_type=resolved_type,
            )
            resolved_specs.append(resolved_spec)

            for feature, feature_rep in (
                (spec.left, left_rep),
                (spec.right, right_rep),
            ):
                if feature_rep == "linear":
                    linear_features.add(feature)
                elif feature_rep == "spline":
                    columns = self._basis_columns(dataset, feature)
                    if not columns:
                        raise ValueError(
                            f"No fitted spline basis found for '{feature}'."
                        )
                    basis_columns[feature] = columns

            if resolved_type == "linear_linear":
                interaction_size = 1
            elif resolved_type == "spline_linear":
                interaction_size = len(basis_columns[spec.left])
            elif resolved_type == "linear_spline":
                interaction_size = len(basis_columns[spec.right])
            elif resolved_type == "spline_spline":
                interaction_size = (
                    len(basis_columns[spec.left])
                    * len(basis_columns[spec.right])
                )
            elif resolved_type == "linear_categorical":
                categorical_feature = (
                    spec.right
                    if spec.left in numerical_features
                    else spec.left
                )
                interaction_size = categorical_dimensions[categorical_feature]
            elif resolved_type == "spline_categorical":
                categorical_feature = (
                    spec.right
                    if spec.left in numerical_features
                    else spec.left
                )
                numeric_feature = (
                    spec.left
                    if spec.left in numerical_features
                    else spec.right
                )
                interaction_size = (
                    len(basis_columns[numeric_feature])
                    * categorical_dimensions[categorical_feature]
                )
            else:
                raise ValueError(
                    f"Unsupported resolved GAM interaction type: {resolved_type!r}"
                )

            interaction_sizes[
                f"__interaction__{spec.left}__{spec.right}"
            ] = int(interaction_size)

        mean_expressions = []

        for feature in sorted(linear_features):
            mean_expressions.append(
                F.avg(
                    F.col(f"__imputed_{feature}")
                ).alias(
                    f"__mean_linear_{feature}"
                )
            )

        for feature in sorted(basis_columns):
            for column in basis_columns[feature]:
                mean_expressions.append(
                    F.avg(
                        F.col(column)
                    ).alias(
                        f"__mean_{column}"
                    )
                )

        linear_means: dict[str, float] = {}
        spline_means: dict[str, list[float]] = {}

        if mean_expressions:
            row = dataset.select(*mean_expressions).first()
            if row is None:
                raise ValueError(
                    "Unable to fit GAM interaction state: "
                    "training DataFrame is empty."
                )

            for feature in sorted(linear_features):
                value = row[f"__mean_linear_{feature}"]
                linear_means[feature] = (
                    float(value) if value is not None else 0.0
                )

            for feature in sorted(basis_columns):
                spline_means[feature] = [
                    (
                        float(row[f"__mean_{column}"])
                        if row[f"__mean_{column}"] is not None
                        else 0.0
                    )
                    for column in basis_columns[feature]
                ]

        return GAMInteractionModel(
            interactionSpecs=_encode_interaction_specs(resolved_specs),
            linearMeans=json.dumps(
                linear_means,
                sort_keys=True,
            ),
            splineMeans=json.dumps(
                spline_means,
                sort_keys=True,
            ),
            categoricalDimensions=json.dumps(
                categorical_dimensions,
                sort_keys=True,
            ),
            interactionSizes=json.dumps(
                interaction_sizes,
                sort_keys=True,
            ),
        )


class GAMInteractionModel(
    Transformer,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """Apply fitted linear/spline/categorical GAM interactions."""

    interactionSpecs = Param(
        Params._dummy(),
        "interactionSpecs",
        "JSON-encoded resolved interaction specifications.",
        TypeConverters.toString,
    )
    linearMeans = Param(
        Params._dummy(),
        "linearMeans",
        "JSON-encoded training means for linear numeric features.",
        TypeConverters.toString,
    )
    splineMeans = Param(
        Params._dummy(),
        "splineMeans",
        "JSON-encoded training means for spline basis columns.",
        TypeConverters.toString,
    )
    categoricalDimensions = Param(
        Params._dummy(),
        "categoricalDimensions",
        "JSON-encoded encoded categorical vector dimensions.",
        TypeConverters.toString,
    )
    interactionSizes = Param(
        Params._dummy(),
        "interactionSizes",
        "JSON-encoded output vector sizes for interaction columns.",
        TypeConverters.toString,
    )

    @keyword_only
    def __init__(
        self,
        interactionSpecs: str = "[]",
        linearMeans: str = "{}",
        splineMeans: str = "{}",
        categoricalDimensions: str = "{}",
        interactionSizes: str = "{}",
    ) -> None:
        super().__init__()
        self._setDefault(
            interactionSpecs="[]",
            linearMeans="{}",
            splineMeans="{}",
            categoricalDimensions="{}",
            interactionSizes="{}",
        )
        self.setParams(
            interactionSpecs=interactionSpecs,
            linearMeans=linearMeans,
            splineMeans=splineMeans,
            categoricalDimensions=categoricalDimensions,
            interactionSizes=interactionSizes,
        )

    @keyword_only
    def setParams(
        self,
        interactionSpecs: str | None = None,
        linearMeans: str | None = None,
        splineMeans: str | None = None,
        categoricalDimensions: str | None = None,
        interactionSizes: str | None = None,
    ) -> "GAMInteractionModel":
        return self._set(**self._input_kwargs)

    @staticmethod
    def _hint_vector_size(
        dataset: DataFrame,
        column: str,
        size: int,
    ) -> DataFrame:
        """Attach explicit Spark ML vector-size metadata."""

        hinted = VectorSizeHint(
            inputCol=column,
            size=int(size),
            handleInvalid="error",
        ).transform(dataset)
        return hinted

    def _transform(self, dataset: DataFrame) -> DataFrame:
        specs = _decode_interaction_specs(
            self.getOrDefault(self.interactionSpecs)
        )
        if not specs:
            return dataset

        linear_means = json.loads(
            self.getOrDefault(self.linearMeans)
        )
        spline_means = json.loads(
            self.getOrDefault(self.splineMeans)
        )
        categorical_dimensions = json.loads(
            self.getOrDefault(self.categoricalDimensions)
        )
        interaction_sizes = json.loads(
            self.getOrDefault(self.interactionSizes)
        )

        result = dataset

        def linear_col(feature: str) -> Column:
            if feature not in linear_means:
                raise ValueError(
                    f"Missing fitted linear mean for '{feature}'."
                )
            return (
                F.col(f"__imputed_{feature}")
                - F.lit(linear_means[feature])
            )

        def spline_cols(feature: str) -> list[Column]:
            if feature not in spline_means:
                raise ValueError(
                    f"Missing fitted spline means for '{feature}'."
                )
            return [
                F.col(f"{feature}_spline_{index}")
                - F.lit(mean)
                for index, mean in enumerate(spline_means[feature])
            ]

        for spec in specs:
            left = spec.left
            right = spec.right
            output_column = (
                f"__interaction__{left}__{right}"
            )

            interaction_type = spec.interaction_type

            if interaction_type == "linear_linear":
                terms = [
                    linear_col(left) * linear_col(right)
                ]

            elif interaction_type == "spline_linear":
                terms = [
                    basis * linear_col(right)
                    for basis in spline_cols(left)
                ]

            elif interaction_type == "linear_spline":
                terms = [
                    linear_col(left) * basis
                    for basis in spline_cols(right)
                ]

            elif interaction_type == "spline_spline":
                left_basis = spline_cols(left)
                right_basis = spline_cols(right)
                terms = [
                    left_term * right_term
                    for left_term in left_basis
                    for right_term in right_basis
                ]

            elif interaction_type in {
                "linear_categorical",
                "spline_categorical",
            }:
                if left in linear_means or left in spline_means:
                    numeric_feature = left
                    categorical_feature = right
                else:
                    numeric_feature = right
                    categorical_feature = left

                encoded_column = (
                    f"__encoded_{categorical_feature}"
                )

                if encoded_column not in dataset.columns:
                    raise ValueError(
                        f"Expected encoded categorical column "
                        f"'{encoded_column}'."
                    )

                encoded = vector_to_array(
                    F.col(encoded_column)
                )
                dimension = int(
                    categorical_dimensions[categorical_feature]
                )

                numeric_terms = (
                    [linear_col(numeric_feature)]
                    if interaction_type == "linear_categorical"
                    else spline_cols(numeric_feature)
                )

                terms = [
                    term * encoded.getItem(category_index)
                    for category_index in range(dimension)
                    for term in numeric_terms
                ]

            else:
                raise ValueError(
                    f"Unsupported GAM interaction type: "
                    f"{interaction_type!r}"
                )

            expected_size = int(
                interaction_sizes[output_column]
            )
            if len(terms) != expected_size:
                raise ValueError(
                    f"GAM interaction '{output_column}' produced "
                    f"{len(terms)} terms but fitted state expects "
                    f"{expected_size}."
                )

            result = result.withColumn(
                output_column,
                array_to_vector(
                    F.array(*terms)
                ),
            )

            # VectorAssembler with handleInvalid='keep' requires vector
            # length metadata. array_to_vector does not reliably preserve
            # that metadata, so make the fitted size explicit.
            result = self._hint_vector_size(
                result,
                output_column,
                expected_size,
            )

        return result


# ======================================================================
# GAM / HGAM MODEL TYPE
# ======================================================================


def _get_gam_model_type(config: dict) -> str:
    """Return the configured GAM model family.

    Supported values are ``gam`` and ``hgam``. Missing configuration
    defaults to ``gam`` so existing configurations remain unchanged.
    """

    gam_config = config["parameters"]["modelling"]["gam"]
    model_type = gam_config.get("model_type", "gam")

    if model_type not in {"gam", "hgam"}:
        raise ValueError(
            "gam.model_type must be one of 'gam' or 'hgam'. "
            f"Got: {model_type!r}"
        )

    return model_type


def _get_hgam_config(config: dict) -> dict:
    """Return and validate the HGAM-specific configuration."""

    gam_config = config["parameters"]["modelling"]["gam"]
    hgam_config = gam_config.get("hgam", {})

    if not isinstance(hgam_config, dict):
        raise ValueError("gam.hgam must be a mapping.")

    grouping_config = hgam_config.get("grouping", {})
    if not isinstance(grouping_config, dict):
        raise ValueError("gam.hgam.grouping must be a mapping.")

    grouping_feature = grouping_config.get("feature")
    if not grouping_feature:
        raise ValueError(
            "HGAM requires gam.hgam.grouping.feature."
        )

    features = config["parameters"]["modelling"]["features"]
    numerical_features = set(features.get("numerical_features", []))
    categorical_features = set(features.get("categorical_features", []))

    if grouping_feature not in categorical_features:
        if grouping_feature in numerical_features:
            raise ValueError(
                "HGAM grouping feature must be categorical. "
                f"'{grouping_feature}' is configured as numerical."
            )
        raise ValueError(
            "HGAM grouping feature is not configured as a categorical "
            "modelling feature. "
            f"Unknown feature: {grouping_feature!r}"
        )

    varying_config = hgam_config.get("varying_smooths", {})
    if not isinstance(varying_config, dict):
        raise ValueError(
            "gam.hgam.varying_smooths must be a mapping."
        )

    varying_enabled = bool(varying_config.get("enabled", False))
    varying_features = list(varying_config.get("features", []))

    if len(varying_features) != len(set(varying_features)):
        raise ValueError(
            "HGAM varying smooth features must be unique."
        )

    unknown_varying = sorted(
        set(varying_features) - numerical_features
    )
    if unknown_varying:
        raise ValueError(
            "HGAM varying smooth features must be numerical modelling "
            f"features. Unknown features: {unknown_varying}"
        )

    if varying_enabled and not varying_features:
        raise ValueError(
            "HGAM varying_smooths.enabled=true requires at least one feature."
        )

    # A varying smooth must actually be spline-backed. The final fitted
    # representation is checked again by HGAMGroupEffectEstimator.
    spline_features = set(
        gam_config.get("feature_transform", {}).get("spline", [])
    )
    unsupported_varying = sorted(
        set(varying_features) - spline_features
    )
    if unsupported_varying:
        raise ValueError(
            "HGAM varying smooth features must also be configured as GAM "
            "spline features. "
            f"Missing from gam.feature_transform.spline: {unsupported_varying}"
        )

    intercept_config = hgam_config.get("intercept", {})
    if not isinstance(intercept_config, dict):
        raise ValueError("gam.hgam.intercept must be a mapping.")

    shrinkage_config = hgam_config.get("shrinkage", {})
    if not isinstance(shrinkage_config, dict):
        raise ValueError("gam.hgam.shrinkage must be a mapping.")

    if shrinkage_config.get("enabled", True):
        group_penalty = float(
            shrinkage_config.get("group_intercept_penalty", 1.0)
        )
        smooth_penalty = float(
            shrinkage_config.get("varying_smooth_penalty", 1.0)
        )

        if group_penalty <= 0.0:
            raise ValueError(
                "HGAM group_intercept_penalty must be greater than 0."
            )
        if smooth_penalty <= 0.0:
            raise ValueError(
                "HGAM varying_smooth_penalty must be greater than 0."
            )

    regularization = hgam_config.get("regularization", {})
    if not isinstance(regularization, dict):
        raise ValueError("gam.hgam.regularization must be a mapping.")

    reg_param = float(regularization.get("reg_param", 0.01))
    if reg_param < 0.0:
        raise ValueError(
            "HGAM regularization.reg_param must be non-negative."
        )

    return hgam_config


# ======================================================================
# B-SPLINE BASIS
# ======================================================================


def _bspline_basis_expression(
    x: Column,
    knots: list[float],
    degree: int,
    basis_index: int,
) -> Column:
    """
    Build one clamped B-spline basis function using Cox-de Boor recursion.

    `knots` must be a complete, non-decreasing knot vector.
    """

    if degree < 0:
        raise ValueError(
            f"B-spline degree must be non-negative. Got {degree}."
        )

    if len(knots) < degree + 2:
        raise ValueError(
            "Knot vector is too short for the requested spline degree."
        )

    if any(
        right < left
        for left, right in zip(knots, knots[1:])
    ):
        raise ValueError(
            "B-spline knot vector must be non-decreasing."
        )

    num_basis = len(knots) - degree - 1

    if not 0 <= basis_index < num_basis:
        raise ValueError(
            f"Invalid B-spline basis index {basis_index}. "
            f"Expected 0 <= index < {num_basis}."
        )

    def basis_zero(index: int) -> Column:
        left = float(knots[index])
        right = float(knots[index + 1])

        if index == len(knots) - 2:
            condition = (
                (x >= F.lit(left))
                & (x <= F.lit(right))
            )
        else:
            condition = (
                (x >= F.lit(left))
                & (x < F.lit(right))
            )

        return condition.cast("double")

    def basis(index: int, current_degree: int) -> Column:
        if current_degree == 0:
            return basis_zero(index)

        left_knot = float(knots[index])
        left_next_knot = float(
            knots[index + current_degree]
        )

        right_knot = float(
            knots[index + current_degree + 1]
        )
        right_next_knot = float(
            knots[index + 1]
        )

        left_basis = basis(
            index,
            current_degree - 1,
        )
        right_basis = basis(
            index + 1,
            current_degree - 1,
        )

        if left_next_knot == left_knot:
            left_term = F.lit(0.0)
        else:
            left_term = (
                (x - F.lit(left_knot))
                / F.lit(left_next_knot - left_knot)
                * left_basis
            )

        if right_knot == right_next_knot:
            right_term = F.lit(0.0)
        else:
            right_term = (
                (F.lit(right_knot) - x)
                / F.lit(right_knot - right_next_knot)
                * right_basis
            )

        return left_term + right_term

    return basis(
        basis_index,
        degree,
    )


def build_full_knot_vector(
    spec: GAMSplineSpec,
) -> list[float]:
    """Build a clamped knot vector."""

    repetitions = spec.degree + 1

    return (
        [spec.lower_bound] * repetitions
        + spec.knots
        + [spec.upper_bound] * repetitions
    )


def _number_of_basis_functions(
    spec: GAMSplineSpec,
) -> int:
    """Return the number of B-spline basis functions."""

    full_knots = build_full_knot_vector(spec)

    return len(full_knots) - spec.degree - 1


# ======================================================================
# SPLINE SPECIFICATION FITTING
# ======================================================================


def _fit_gam_spline_specs_spark(
    df: DataFrame,
    features: list[str],
    degree: int,
    num_knots: int,
) -> tuple[dict[str, GAMSplineSpec], dict[str, dict[str, str]]]:
    """
    Fit optional spline specifications using observed training values only.

    Every numerical feature remains valid as a linear feature. A configured
    spline feature is upgraded to a spline only when the requested quantile
    knots are sufficiently distinct to define a valid spline. Otherwise the
    feature automatically remains linear.

    Knot estimation is performed on the original feature columns.
    Null/NaN values are excluded from knot estimation.

    Missing values are handled separately by the GAM preparation/imputation
    stage during transformation. This prevents median imputation from
    artificially creating repeated quantile knots.
    """

    if not features:
        return {}, {}

    if len(features) != len(set(features)):
        raise ValueError(
            "GAM spline features must be unique."
        )

    if degree < 1:
        raise ValueError(
            f"Spline degree must be at least 1. Got: {degree}"
        )

    if num_knots < 2:
        raise ValueError(
            f"Number of spline knots must be at least 2. Got: {num_knots}"
        )

    missing = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            "GAM spline fitting requires the following columns: "
            + ", ".join(missing)
        )

    positions = np.linspace(
        0.0,
        1.0,
        num_knots,
    ).tolist()

    # Fit knots on observed values, not on __imputed_* columns.
    aggregation_expressions = []

    for feature in features:
        column = F.col(feature).cast("double")
        valid_column = F.when(
            column.isNotNull() & ~F.isnan(column),
            column,
        )

        aggregation_expressions.extend(
            [
                F.count(valid_column).alias(
                    f"__valid_count_{feature}"
                ),
                F.percentile_approx(
                    valid_column,
                    positions,
                    10000,
                ).alias(
                    f"__quantiles_{feature}"
                ),
            ]
        )

    # One Spark action for all configured spline-able features.
    row = df.select(*aggregation_expressions).first()

    if row is None:
        raise ValueError(
            "Unable to fit GAM spline specifications: "
            "training DataFrame is empty."
        )

    specs: dict[str, GAMSplineSpec] = {}
    representations: dict[str, dict[str, str]] = {}

    for feature in features:
        valid_count = row[f"__valid_count_{feature}"]

        if valid_count is None or valid_count == 0:
            raise ValueError(
                f"Spline-able feature '{feature}' has no valid observed "
                "training values, so it cannot be represented linearly "
                "or by a spline."
            )

        quantiles = row[f"__quantiles_{feature}"]
        reason: str | None = None

        if quantiles is None:
            reason = "quantile estimation returned no values"
        else:
            quantiles = [
                float(value)
                for value in quantiles
                if value is not None
            ]

            if len(quantiles) != num_knots:
                reason = (
                    f"only {len(quantiles)} of {num_knots} requested "
                    "quantiles were available"
                )
            elif not np.isfinite(quantiles).all():
                reason = "quantile estimation produced non-finite values"
            else:
                lower_bound = quantiles[0]
                upper_bound = quantiles[-1]
                all_knots = [
                    lower_bound,
                    *quantiles[1:-1],
                    upper_bound,
                ]

                if lower_bound >= upper_bound:
                    reason = (
                        "insufficient observed variation for the requested "
                        "spline knots"
                    )
                elif any(
                    right <= left
                    for left, right in zip(
                        all_knots,
                        all_knots[1:],
                    )
                ):
                    reason = (
                        "insufficient unique quantiles for the requested "
                        "spline knots"
                    )

        if reason is not None:
            # Linear is the baseline representation for every numerical
            # feature. A spline request is therefore opportunistic rather
            # than a hard requirement.
            representations[feature] = {
                "representation": "linear",
                "reason": reason,
            }
            continue

        internal_knots = quantiles[1:-1]
        specs[feature] = GAMSplineSpec(
            feature=feature,
            degree=degree,
            knots=internal_knots,
            lower_bound=quantiles[0],
            upper_bound=quantiles[-1],
        )
        representations[feature] = {
            "representation": "spline",
            "reason": "",
        }

    return specs, representations


# ======================================================================
# SERIALIZATION
# ======================================================================


def _encode_spline_specs(
    specs: dict[str, GAMSplineSpec],
) -> str:
    """Serialize spline specifications to JSON."""

    return json.dumps(
        {
            feature: asdict(spec)
            for feature, spec in specs.items()
        },
        sort_keys=True,
    )


def _decode_spline_specs(
    value: str,
) -> dict[str, GAMSplineSpec]:
    """Deserialize and validate fitted spline specifications."""

    payload = json.loads(value)

    if not isinstance(payload, dict):
        raise ValueError(
            "Serialized GAM spline specifications must be a JSON object."
        )

    specs: dict[str, GAMSplineSpec] = {}

    required_fields = {
        "feature",
        "degree",
        "knots",
        "lower_bound",
        "upper_bound",
    }

    for feature, data in payload.items():
        if not isinstance(data, dict):
            raise ValueError(
                f"Invalid spline specification for '{feature}'."
            )

        missing = required_fields - set(data)

        if missing:
            raise ValueError(
                f"Spline specification for '{feature}' is missing: "
                + ", ".join(sorted(missing))
            )

        if data["feature"] != feature:
            raise ValueError(
                f"Spline specification key mismatch for '{feature}': "
                f"stored feature={data['feature']!r}."
            )

        degree = int(data["degree"])
        knots = [
            float(knot)
            for knot in data["knots"]
        ]
        lower_bound = float(data["lower_bound"])
        upper_bound = float(data["upper_bound"])

        if degree < 1:
            raise ValueError(
                f"Invalid spline degree for '{feature}': {degree}"
            )

        if not knots:
            raise ValueError(
                f"Spline specification for '{feature}' "
                "has no internal knots."
            )

        all_knots = [
            lower_bound,
            *knots,
            upper_bound,
        ]

        if not np.isfinite(all_knots).all():
            raise ValueError(
                f"Spline specification for '{feature}' "
                "contains non-finite knots."
            )

        if any(
            right <= left
            for left, right in zip(
                all_knots,
                all_knots[1:],
            )
        ):
            raise ValueError(
                f"Spline specification for '{feature}' "
                "contains non-increasing knots."
            )

        specs[feature] = GAMSplineSpec(
            feature=feature,
            degree=degree,
            knots=knots,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    return specs


# ======================================================================
# GAM PREPARATION TRANSFORMER
# ======================================================================


class GAMPreparationTransformer(
    Transformer,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """
    Deterministic preprocessing that belongs inside the persisted GAM
    PipelineModel.
    """

    numericalNullColumns = Param(
        Params._dummy(),
        "numericalNullColumns",
        "Numerical columns whose nulls become -1.",
        TypeConverters.toList,
    )

    categoricalColumns = Param(
        Params._dummy(),
        "categoricalColumns",
        "Categorical columns whose nulls become Unknown.",
        TypeConverters.toList,
    )

    @keyword_only
    def __init__(
        self,
        numericalNullColumns: list[str] | None = None,
        categoricalColumns: list[str] | None = None,
    ) -> None:

        super().__init__()

        self._setDefault(
            numericalNullColumns=[],
            categoricalColumns=[],
        )

        self.setParams(
            numericalNullColumns=numericalNullColumns or [],
            categoricalColumns=categoricalColumns or [],
        )

    @keyword_only
    def setParams(
        self,
        numericalNullColumns: list[str] | None = None,
        categoricalColumns: list[str] | None = None,
    ) -> "GAMPreparationTransformer":

        return self._set(
            **self._input_kwargs,
        )

    def _transform(
        self,
        dataset: DataFrame,
    ) -> DataFrame:

        result = dataset

        # Preserve the existing special handling used by the modelling
        # pipeline for this feature.
        for column in self.getOrDefault(
            self.numericalNullColumns
        ):

            if column in result.columns:

                result = result.withColumn(
                    column,
                    F.coalesce(
                        F.col(column).cast("double"),
                        F.lit(-1.0),
                    ),
                )

        # Preserve categorical null handling.
        for column in self.getOrDefault(
            self.categoricalColumns
        ):

            if column in result.columns:

                result = result.withColumn(
                    column,
                    F.coalesce(
                        F.col(column).cast("string"),
                        F.lit("Unknown"),
                    ),
                )

        return result


# ======================================================================
# SPLINE ESTIMATOR
# ======================================================================


class GAMSplineEstimator(
    Estimator,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """
    Spark Estimator that learns spline knots from training data.

    The fitted estimator produces GAMSplineModel, which stores the
    frozen spline specifications.
    """

    inputCols = Param(
        Params._dummy(),
        "inputCols",
        "Logical numerical features to spline-transform.",
        TypeConverters.toList,
    )

    degree = Param(
        Params._dummy(),
        "degree",
        "B-spline degree.",
        TypeConverters.toInt,
    )

    numKnots = Param(
        Params._dummy(),
        "numKnots",
        "Number of quantile knots including boundaries.",
        TypeConverters.toInt,
    )

    @keyword_only
    def __init__(
        self,
        inputCols: list[str] | None = None,
        degree: int = 3,
        numKnots: int = 6,
    ) -> None:

        super().__init__()

        self._setDefault(
            inputCols=[],
            degree=3,
            numKnots=6,
        )

        self.setParams(
            inputCols=inputCols or [],
            degree=degree,
            numKnots=numKnots,
        )

    @keyword_only
    def setParams(
        self,
        inputCols: list[str] | None = None,
        degree: int = 3,
        numKnots: int = 6,
    ) -> "GAMSplineEstimator":

        return self._set(
            **self._input_kwargs,
        )

    def _fit(
        self,
        dataset: DataFrame,
    ) -> "GAMSplineModel":

        features = list(
            self.getOrDefault(
                self.inputCols
            )
        )

        degree = int(
            self.getOrDefault(
                self.degree
            )
        )

        num_knots = int(
            self.getOrDefault(
                self.numKnots
            )
        )

        specs, representations = _fit_gam_spline_specs_spark(
            df=dataset,
            features=features,
            degree=degree,
            num_knots=num_knots,
        )

        return GAMSplineModel(
            splineSpecs=_encode_spline_specs(specs),
            representations=json.dumps(
                representations,
                sort_keys=True,
            ),
            inputCols=features,
        )


# ======================================================================
# SPLINE MODEL
# ======================================================================


class GAMSplineModel(
    Transformer,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """
    Apply frozen spline specifications.

    The constructor deliberately supports zero-argument construction so
    Spark can recreate the Python stage during PipelineModel loading.
    """

    inputCols = Param(
        Params._dummy(),
        "inputCols",
        "Logical numerical features to spline-transform.",
        TypeConverters.toList,
    )

    splineSpecs = Param(
        Params._dummy(),
        "splineSpecs",
        "JSON-encoded fitted spline specifications.",
        TypeConverters.toString,
    )

    representations = Param(
        Params._dummy(),
        "representations",
        "JSON-encoded fitted numerical representations.",
        TypeConverters.toString,
    )

    @keyword_only
    def __init__(
        self,
        splineSpecs: str | None = None,
        representations: str | None = None,
        inputCols: list[str] | None = None,
    ) -> None:

        super().__init__()

        self._setDefault(
            splineSpecs="{}",
            representations="{}",
            inputCols=[],
        )

        self.setParams(
            splineSpecs=splineSpecs or "{}",
            representations=representations or "{}",
            inputCols=inputCols or [],
        )

    @keyword_only
    def setParams(
        self,
        splineSpecs: str | None = None,
        representations: str | None = None,
        inputCols: list[str] | None = None,
    ) -> "GAMSplineModel":
        return self._set(
            **self._input_kwargs,
        )

    def _transform(
        self,
        dataset: DataFrame,
    ) -> DataFrame:

        specs = _decode_spline_specs(
            self.getOrDefault(
                self.splineSpecs
            )
        )
        representations = json.loads(
            self.getOrDefault(
                self.representations
            )
        )

        features = list(
            self.getOrDefault(
                self.inputCols
            )
        )

        if not features:
            return dataset

        result = dataset

        for feature in features:
            representation = representations.get(
                feature,
                {"representation": "spline"},
            ).get("representation")

            if representation == "linear":
                # The imputed numerical column is already present and is
                # consumed by the final GAM assembly stage.
                continue

            if representation != "spline":
                raise ValueError(
                    f"Unsupported fitted representation for '{feature}': "
                    f"{representation!r}."
                )

            if feature not in specs:
                raise ValueError(
                    f"No fitted spline specification found for "
                    f"'{feature}'."
                )

            spec = specs[feature]

            if spec.feature != feature:
                raise ValueError(
                    f"Spline specification mismatch for '{feature}'. "
                    f"Stored feature is '{spec.feature}'."
                )

            if spec.degree < 1:
                raise ValueError(
                    f"Invalid spline degree for '{feature}': "
                    f"{spec.degree}"
                )

            if spec.lower_bound >= spec.upper_bound:
                raise ValueError(
                    f"Invalid spline bounds for '{feature}': "
                    f"{spec.lower_bound} >= {spec.upper_bound}"
                )

            transformed_feature = (
                f"__imputed_{feature}"
            )

            if transformed_feature not in result.columns:
                raise ValueError(
                    f"Expected imputed spline feature "
                    f"'{transformed_feature}' not found in input DataFrame."
                )

            # Clamp scoring values to the support observed during
            # training. This gives stable basis functions for future
            # observations outside the original training range.
            x = F.least(
                F.greatest(
                    F.col(
                        transformed_feature
                    ).cast("double"),
                    F.lit(
                        spec.lower_bound
                    ),
                ),
                F.lit(
                    spec.upper_bound
                ),
            )

            full_knots = build_full_knot_vector(
                spec
            )

            num_basis = _number_of_basis_functions(
                spec
            )

            for basis_index in range(num_basis):

                result = result.withColumn(
                    f"{feature}_spline_{basis_index}",
                    _bspline_basis_expression(
                        x=x,
                        knots=full_knots,
                        degree=spec.degree,
                        basis_index=basis_index,
                    ),
                )

        return result


# ======================================================================
# FEATURE CONFIGURATION
# ======================================================================


def _get_gam_feature_groups(
    config: dict,
) -> tuple[
    list[str],
    list[str],
    list[str],
    list[str],
]:
    """
    Determine GAM feature treatment while preserving V1 main effects.

    Returns:
        spline_features
        linear_numerical_features
        categorical_features
        engineered_features
    """

    features = config[
        "parameters"
    ][
        "modelling"
    ][
        "features"
    ]

    numerical_features = list(
        features.get(
            "numerical_features",
            [],
        )
    )

    categorical_features = list(
        features.get(
            "categorical_features",
            [],
        )
    )

    engineered_features = list(
        features.get(
            "engineered_features",
            [],
        )
    )

    gam_config = config[
        "parameters"
    ][
        "modelling"
    ][
        "gam"
    ]

    feature_transform = gam_config[
        "feature_transform"
    ]

    default_numerical = feature_transform.get(
        "default_numerical",
        "linear",
    )

    if default_numerical != "linear":
        raise ValueError(
            "GAM V1 only supports "
            "default_numerical='linear'. "
            f"Got: {default_numerical!r}"
        )

    knot_strategy = gam_config[
        "spline"
    ].get(
        "knot_strategy",
        "quantile",
    )

    if knot_strategy != "quantile":
        raise ValueError(
            "GAM V1 only supports "
            "knot_strategy='quantile'. "
            f"Got: {knot_strategy!r}"
        )

    spline_features = list(
        feature_transform.get(
            "spline",
            [],
        )
    )

    if len(spline_features) != len(
        set(spline_features)
    ):
        raise ValueError(
            "GAM spline feature configuration "
            "contains duplicates."
        )

    numerical_set = set(
        numerical_features
    )

    spline_set = set(
        spline_features
    )

    unknown = sorted(
        spline_set - numerical_set
    )

    if unknown:
        raise ValueError(
            "GAM spline features must be configured "
            "numerical features. "
            f"Unknown features: {unknown}"
        )

    engineered_overlap = sorted(
        spline_set.intersection(
            engineered_features
        )
    )

    if engineered_overlap:
        raise ValueError(
            "GAM spline features cannot also be "
            "engineered features: "
            + ", ".join(engineered_overlap)
        )

    linear_numerical_features = [
        f"__imputed_{feature}"
        for feature in numerical_features
        if feature not in spline_set
    ]

    return (
        spline_features,
        linear_numerical_features,
        categorical_features,
        engineered_features,
    )


# ======================================================================
# FINAL MODEL FEATURES
# ======================================================================


def _get_gam_model_features(
    config: dict,
) -> list[str]:

    (
        spline_features,
        linear_numerical_features,
        categorical_features,
        engineered_features,
    ) = _get_gam_feature_groups(
        config
    )

    spline_config = config[
        "parameters"
    ][
        "modelling"
    ][
        "gam"
    ][
        "spline"
    ]

    degree = int(
        spline_config["degree"]
    )

    num_knots = int(
        spline_config["num_knots"]
    )

    if degree < 1:
        raise ValueError(
            f"GAM spline degree must be >= 1. "
            f"Got: {degree}"
        )

    if num_knots < 2:
        raise ValueError(
            f"GAM num_knots must be >= 2. "
            f"Got: {num_knots}"
        )

    # K quantile knots + d+1 repeated boundaries
    # produce K+d-1 basis functions.
    num_basis = (
        num_knots
        + degree
        - 1
    )

    spline_basis_features = [
        f"{feature}_spline_{basis_index}"
        for feature in spline_features
        for basis_index in range(num_basis)
    ]

    encoded_categorical_features = [
        f"__encoded_{feature}"
        for feature in categorical_features
    ]

    model_features = (
        linear_numerical_features
        + spline_basis_features
        + encoded_categorical_features
        + engineered_features
    )

    interaction_specs = _get_gam_interaction_specs(config)

    model_features += [
        f"__interaction__{spec.left}__{spec.right}"
        for spec in interaction_specs
    ]

    if not model_features:
        raise ValueError(
            "GAM has no model features after "
            "feature treatment."
        )

    if len(model_features) != len(
        set(model_features)
    ):
        raise ValueError(
            "GAM model feature columns contain duplicates."
        )

    return model_features


# ======================================================================
# GAM INTERNAL PREPROCESSING
# ======================================================================


def _build_gam_preparation_stages(
    config: dict,
) -> list[Any]:
    """
    Build all preprocessing stages owned by GAM.

    The outer modelling preprocessor is deliberately a no-op for GAM.
    Therefore GAM must perform its own imputation and categorical
    encoding inside this persisted Pipeline.
    """

    features = config[
        "parameters"
    ][
        "modelling"
    ][
        "features"
    ]

    numerical_features = list(
        features.get(
            "numerical_features",
            [],
        )
    )

    categorical_features = list(
        features.get(
            "categorical_features",
            [],
        )
    )

    stages: list[Any] = []

    # Preserve existing deterministic preprocessing semantics.
    null_numerical_columns = []

    if (
        "months_since_last_delinquency"
        in numerical_features
    ):
        null_numerical_columns.append(
            "months_since_last_delinquency"
        )

    stages.append(
        GAMPreparationTransformer(
            numericalNullColumns=(
                null_numerical_columns
            ),
            categoricalColumns=(
                categorical_features
            ),
        )
    )

    # --------------------------------------------------------------
    # Numerical imputation.
    # --------------------------------------------------------------

    if numerical_features:

        stages.append(
            Imputer(
                inputCols=numerical_features,
                outputCols=[
                    f"__imputed_{feature}"
                    for feature in numerical_features
                ],
                strategy="median",
            )
        )

    # --------------------------------------------------------------
    # Categorical indexing.
    # --------------------------------------------------------------

    indexed_columns = []

    for feature in categorical_features:

        indexed = (
            f"__indexed_{feature}"
        )

        stages.append(
            StringIndexer(
                inputCol=feature,
                outputCol=indexed,
                handleInvalid="keep",
            )
        )

        indexed_columns.append(
            indexed
        )

    # --------------------------------------------------------------
    # One-hot encoding.
    # --------------------------------------------------------------

    if categorical_features:

        stages.append(
            OneHotEncoder(
                inputCols=indexed_columns,
                outputCols=[
                    f"__encoded_{feature}"
                    for feature in categorical_features
                ],
                handleInvalid="keep",
            )
        )

    return stages


# ======================================================================
# HGAM GROUP EFFECTS
# ======================================================================


class HGAMGroupEffectEstimator(
    Estimator,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """Build frozen HGAM group-intercept and varying-smooth columns.

    The global GAM effects remain unchanged. Hierarchical effects are
    represented as group indicators and group-by-global-spline deviations.
    Their feature scaling controls their relative L2 shrinkage when the
    downstream Spark LogisticRegression is fitted.
    """

    groupingFeature = Param(
        Params._dummy(),
        "groupingFeature",
        "Categorical feature defining HGAM groups.",
        TypeConverters.toString,
    )
    groupInterceptEnabled = Param(
        Params._dummy(),
        "groupInterceptEnabled",
        "Whether group-level intercept deviations are included.",
        TypeConverters.toBoolean,
    )
    varyingSmoothFeatures = Param(
        Params._dummy(),
        "varyingSmoothFeatures",
        "Numerical spline features with group-varying deviations.",
        TypeConverters.toList,
    )
    groupInterceptPenalty = Param(
        Params._dummy(),
        "groupInterceptPenalty",
        "Relative shrinkage strength for group intercepts.",
        TypeConverters.toFloat,
    )
    varyingSmoothPenalty = Param(
        Params._dummy(),
        "varyingSmoothPenalty",
        "Relative shrinkage strength for varying smooths.",
        TypeConverters.toFloat,
    )
    shrinkageEnabled = Param(
        Params._dummy(),
        "shrinkageEnabled",
        "Whether hierarchical feature scaling is enabled.",
        TypeConverters.toBoolean,
    )

    @keyword_only
    def __init__(
        self,
        groupingFeature: str = "",
        groupInterceptEnabled: bool = True,
        varyingSmoothFeatures: list[str] | None = None,
        groupInterceptPenalty: float = 1.0,
        varyingSmoothPenalty: float = 1.0,
        shrinkageEnabled: bool = True,
    ) -> None:
        super().__init__()
        self._setDefault(
            groupingFeature="",
            groupInterceptEnabled=True,
            varyingSmoothFeatures=[],
            groupInterceptPenalty=1.0,
            varyingSmoothPenalty=1.0,
            shrinkageEnabled=True,
        )
        self.setParams(
            groupingFeature=groupingFeature,
            groupInterceptEnabled=groupInterceptEnabled,
            varyingSmoothFeatures=varyingSmoothFeatures or [],
            groupInterceptPenalty=groupInterceptPenalty,
            varyingSmoothPenalty=varyingSmoothPenalty,
            shrinkageEnabled=shrinkageEnabled,
        )

    @keyword_only
    def setParams(
        self,
        groupingFeature: str | None = None,
        groupInterceptEnabled: bool | None = None,
        varyingSmoothFeatures: list[str] | None = None,
        groupInterceptPenalty: float = 1.0,
        varyingSmoothPenalty: float = 1.0,
        shrinkageEnabled: bool = True,
    ) -> "HGAMGroupEffectEstimator":
        return self._set(**self._input_kwargs)

    @staticmethod
    def _basis_columns(
        dataset: DataFrame,
        feature: str,
    ) -> list[str]:
        prefix = f"{feature}_spline_"
        columns = [
            column
            for column in dataset.columns
            if column.startswith(prefix)
            and column[len(prefix):].isdigit()
        ]
        return sorted(
            columns,
            key=lambda column: int(column.rsplit("_", 1)[1]),
        )

    @staticmethod
    def _encoded_dimension(
        dataset: DataFrame,
        feature: str,
    ) -> int:
        encoded_column = f"__encoded_{feature}"
        if encoded_column not in dataset.columns:
            raise ValueError(
                f"Expected encoded HGAM grouping column "
                f"'{encoded_column}' not found."
            )

        dimension = dataset.schema[encoded_column].metadata.get(
            "ml_attr",
            {},
        ).get("num_attrs")

        if dimension is None:
            raise ValueError(
                "Unable to determine encoded dimensionality for "
                f"HGAM grouping feature '{feature}'."
            )

        dimension = int(dimension)
        if dimension < 1:
            raise ValueError(
                f"Invalid encoded dimensionality for '{feature}': {dimension}"
            )

        return dimension

    def _fit(self, dataset: DataFrame) -> "HGAMGroupEffectModel":
        grouping_feature = self.getOrDefault(self.groupingFeature)
        group_intercept_enabled = self.getOrDefault(
            self.groupInterceptEnabled
        )
        varying_features = list(
            self.getOrDefault(self.varyingSmoothFeatures)
        )
        # These values are fitted from the actual upstream representation.
        # Do not read them as Estimator Params: they are Model outputs and are
        # only known after the spline stage has run.
        group_intercept_penalty = float(
            self.getOrDefault(self.groupInterceptPenalty)
        )
        varying_smooth_penalty = float(
            self.getOrDefault(self.varyingSmoothPenalty)
        )
        shrinkage_enabled = self.getOrDefault(self.shrinkageEnabled)

        encoded_column = f"__encoded_{grouping_feature}"
        if encoded_column not in dataset.columns:
            raise ValueError(
                f"HGAM grouping column '{encoded_column}' is missing "
                "after categorical encoding."
            )

        group_dimension = self._encoded_dimension(
            dataset,
            grouping_feature,
        )

        # Reuse the representation selected by the existing GAM spline stage.
        # A configured spline is opportunistic: if the fitted spline basis is
        # available, HGAM varies that basis by group. If the GAM fell back to
        # linear because the training data did not support the requested knots,
        # HGAM varies the imputed linear representation instead.
        varying_sizes: dict[str, int] = {}
        varying_representations: dict[str, str] = {}

        for feature in varying_features:
            basis_columns = self._basis_columns(dataset, feature)

            if basis_columns:
                varying_representations[feature] = "spline"
                varying_sizes[feature] = len(basis_columns) * group_dimension
                continue

            linear_column = f"__imputed_{feature}"
            if linear_column not in dataset.columns:
                raise ValueError(
                    f"HGAM varying feature '{feature}' has neither a fitted "
                    "spline basis nor an imputed linear representation."
                )

            varying_representations[feature] = "linear"
            varying_sizes[feature] = group_dimension

        return HGAMGroupEffectModel(
            groupingFeature=grouping_feature,
            groupDimension=group_dimension,
            groupInterceptEnabled=group_intercept_enabled,
            varyingSmoothFeatures=varying_features,
            varyingSmoothSizes=json.dumps(varying_sizes, sort_keys=True),
            varyingRepresentations=json.dumps(
                varying_representations,
                sort_keys=True,
            ),
            groupInterceptPenalty=group_intercept_penalty,
            varyingSmoothPenalty=varying_smooth_penalty,
            shrinkageEnabled=shrinkage_enabled,
        )


class HGAMGroupEffectModel(
    Transformer,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """Apply the frozen HGAM group-effect representation."""

    groupingFeature = Param(
        Params._dummy(),
        "groupingFeature",
        "HGAM grouping feature.",
        TypeConverters.toString,
    )
    groupDimension = Param(
        Params._dummy(),
        "groupDimension",
        "Number of fitted encoded group dimensions.",
        TypeConverters.toInt,
    )
    groupInterceptEnabled = Param(
        Params._dummy(),
        "groupInterceptEnabled",
        "Whether group intercept deviations are included.",
        TypeConverters.toBoolean,
    )
    varyingSmoothFeatures = Param(
        Params._dummy(),
        "varyingSmoothFeatures",
        "HGAM varying smooth features.",
        TypeConverters.toList,
    )
    varyingSmoothSizes = Param(
        Params._dummy(),
        "varyingSmoothSizes",
        "JSON encoded fitted varying smooth sizes.",
        TypeConverters.toString,
    )
    varyingRepresentations = Param(
        Params._dummy(),
        "varyingRepresentations",
        "JSON encoded fitted linear/spline representations for varying features.",
        TypeConverters.toString,
    )
    groupInterceptPenalty = Param(
        Params._dummy(),
        "groupInterceptPenalty",
        "Relative group intercept shrinkage.",
        TypeConverters.toFloat,
    )
    varyingSmoothPenalty = Param(
        Params._dummy(),
        "varyingSmoothPenalty",
        "Relative varying smooth shrinkage.",
        TypeConverters.toFloat,
    )
    shrinkageEnabled = Param(
        Params._dummy(),
        "shrinkageEnabled",
        "Whether hierarchical feature scaling is enabled.",
        TypeConverters.toBoolean,
    )

    @keyword_only
    def __init__(
        self,
        groupingFeature: str = "",
        groupDimension: int = 1,
        groupInterceptEnabled: bool = True,
        varyingSmoothFeatures: list[str] | None = None,
        varyingSmoothSizes: str = "{}",
        varyingRepresentations: str = "{}",
        groupInterceptPenalty: float = 1.0,
        varyingSmoothPenalty: float = 1.0,
        shrinkageEnabled: bool = True,
    ) -> None:
        super().__init__()
        self._setDefault(
            groupingFeature="",
            groupDimension=1,
            groupInterceptEnabled=True,
            varyingSmoothFeatures=[],
            varyingSmoothSizes="{}",
            varyingRepresentations="{}",
            groupInterceptPenalty=1.0,
            varyingSmoothPenalty=1.0,
            shrinkageEnabled=True,
        )
        self.setParams(
            groupingFeature=groupingFeature,
            groupDimension=groupDimension,
            groupInterceptEnabled=groupInterceptEnabled,
            varyingSmoothFeatures=varyingSmoothFeatures or [],
            varyingSmoothSizes=varyingSmoothSizes,
            varyingRepresentations=varyingRepresentations,
            groupInterceptPenalty=groupInterceptPenalty,
            varyingSmoothPenalty=varyingSmoothPenalty,
            shrinkageEnabled=shrinkageEnabled,
        )

    @keyword_only
    def setParams(
        self,
        groupingFeature: str | None = None,
        groupDimension: int = 1,
        groupInterceptEnabled: bool = True,
        varyingSmoothFeatures: list[str] | None = None,
        varyingSmoothSizes: str = "{}",
        varyingRepresentations: str = "{}",
        groupInterceptPenalty: float = 1.0,
        varyingSmoothPenalty: float = 1.0,
        shrinkageEnabled: bool = True,
    ) -> "HGAMGroupEffectModel":
        return self._set(**self._input_kwargs)

    @staticmethod
    def _basis_columns(
        dataset: DataFrame,
        feature: str,
    ) -> list[str]:
        prefix = f"{feature}_spline_"
        columns = [
            column
            for column in dataset.columns
            if column.startswith(prefix)
            and column[len(prefix):].isdigit()
        ]
        return sorted(
            columns,
            key=lambda column: int(column.rsplit("_", 1)[1]),
        )

    @staticmethod
    def _hint_vector_size(
        dataset: DataFrame,
        column: str,
        size: int,
    ) -> DataFrame:
        return VectorSizeHint(
            inputCol=column,
            size=int(size),
            handleInvalid="error",
        ).transform(dataset)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        grouping_feature = self.getOrDefault(self.groupingFeature)
        group_dimension = int(self.getOrDefault(self.groupDimension))
        group_intercept_enabled = self.getOrDefault(
            self.groupInterceptEnabled
        )
        varying_features = list(
            self.getOrDefault(self.varyingSmoothFeatures)
        )
        varying_representations = json.loads(
            self.getOrDefault(self.varyingRepresentations)
        )
        varying_sizes = json.loads(
            self.getOrDefault(self.varyingSmoothSizes)
        )
        group_intercept_penalty = float(
            self.getOrDefault(self.groupInterceptPenalty)
        )
        varying_smooth_penalty = float(
            self.getOrDefault(self.varyingSmoothPenalty)
        )
        shrinkage_enabled = self.getOrDefault(self.shrinkageEnabled)

        encoded_column = f"__encoded_{grouping_feature}"
        if encoded_column not in dataset.columns:
            raise ValueError(
                f"HGAM encoded grouping column '{encoded_column}' is missing."
            )

        encoded = vector_to_array(F.col(encoded_column))
        result = dataset

        if group_intercept_enabled:
            scale = (
                1.0 / group_intercept_penalty
                if shrinkage_enabled
                else 1.0
            )
            terms = [
                encoded.getItem(index) * F.lit(scale)
                for index in range(group_dimension)
            ]
            output_column = "__hgam_group_intercept"
            result = result.withColumn(
                output_column,
                array_to_vector(F.array(*terms)),
            )
            result = self._hint_vector_size(
                result,
                output_column,
                group_dimension,
            )

        for feature in varying_features:
            representation = varying_representations.get(feature)

            if representation == "spline":
                basis_columns = self._basis_columns(result, feature)
                if not basis_columns:
                    raise ValueError(
                        f"Fitted HGAM representation for '{feature}' is spline, "
                        "but its spline basis columns are missing."
                    )
                base_terms = [
                    F.col(basis_column)
                    for basis_column in basis_columns
                ]

            elif representation == "linear":
                linear_column = f"__imputed_{feature}"
                if linear_column not in result.columns:
                    raise ValueError(
                        f"Fitted HGAM representation for '{feature}' is linear, "
                        f"but '{linear_column}' is missing."
                    )
                base_terms = [F.col(linear_column)]

            else:
                raise ValueError(
                    f"Unsupported fitted HGAM representation for '{feature}': "
                    f"{representation!r}."
                )

            scale = (
                1.0 / varying_smooth_penalty
                if shrinkage_enabled
                else 1.0
            )

            terms = []
            for group_index in range(group_dimension):
                group_indicator = encoded.getItem(group_index)
                for base_term in base_terms:
                    terms.append(
                        base_term
                        * group_indicator
                        * F.lit(scale)
                    )

            expected_size = int(varying_sizes.get(feature, len(terms)))
            if len(terms) != expected_size:
                raise ValueError(
                    f"HGAM varying feature '{feature}' produced "
                    f"{len(terms)} terms but fitted state expects "
                    f"{expected_size}."
                )

            output_column = f"__hgam_smooth__{feature}"
            result = result.withColumn(
                output_column,
                array_to_vector(F.array(*terms)),
            )
            result = self._hint_vector_size(
                result,
                output_column,
                expected_size,
            )

        return result


# ======================================================================
# FINAL FEATURE ASSEMBLY
# ======================================================================


class GAMFeatureAssemblyEstimator(
    Estimator,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """
    Resolve the actual fitted GAM representation and freeze the final
    VectorAssembler input columns.

    A configured spline-able numerical feature can be either a spline or
    linear after fitting, so the final feature list cannot be determined
    safely from configuration alone.
    """

    numericalFeatures = Param(
        Params._dummy(),
        "numericalFeatures",
        "Canonical numerical GAM features in configured order.",
        TypeConverters.toList,
    )
    categoricalFeatures = Param(
        Params._dummy(),
        "categoricalFeatures",
        "Canonical categorical GAM features in configured order.",
        TypeConverters.toList,
    )
    engineeredFeatures = Param(
        Params._dummy(),
        "engineeredFeatures",
        "Engineered GAM features in configured order.",
        TypeConverters.toList,
    )
    interactionOutputCols = Param(
        Params._dummy(),
        "interactionOutputCols",
        "Fitted interaction vector columns in configured order.",
        TypeConverters.toList,
    )
    hgamOutputCols = Param(
        Params._dummy(),
        "hgamOutputCols",
        "Fitted HGAM hierarchical vector columns in configured order.",
        TypeConverters.toList,
    )

    @keyword_only
    def __init__(
        self,
        numericalFeatures: list[str] | None = None,
        categoricalFeatures: list[str] | None = None,
        engineeredFeatures: list[str] | None = None,
        interactionOutputCols: list[str] | None = None,
        hgamOutputCols: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._setDefault(
            numericalFeatures=[],
            categoricalFeatures=[],
            engineeredFeatures=[],
            interactionOutputCols=[],
            hgamOutputCols=[],
        )
        self.setParams(
            numericalFeatures=numericalFeatures or [],
            categoricalFeatures=categoricalFeatures or [],
            engineeredFeatures=engineeredFeatures or [],
            interactionOutputCols=interactionOutputCols or [],
            hgamOutputCols=hgamOutputCols or [],
        )

    @keyword_only
    def setParams(
        self,
        numericalFeatures: list[str] | None = None,
        categoricalFeatures: list[str] | None = None,
        engineeredFeatures: list[str] | None = None,
        interactionOutputCols: list[str] | None = None,
        hgamOutputCols: list[str] | None = None,
    ) -> "GAMFeatureAssemblyEstimator":
        return self._set(**self._input_kwargs)

    @staticmethod
    def _spline_basis_columns(
        dataset: DataFrame,
        feature: str,
    ) -> list[str]:
        prefix = f"{feature}_spline_"
        columns = [
            column
            for column in dataset.columns
            if column.startswith(prefix)
            and column[len(prefix):].isdigit()
        ]
        return sorted(
            columns,
            key=lambda column: int(column.rsplit("_", 1)[1]),
        )

    def _fit(self, dataset: DataFrame) -> "GAMFeatureAssemblyModel":
        numerical_features = list(
            self.getOrDefault(self.numericalFeatures)
        )
        categorical_features = list(
            self.getOrDefault(self.categoricalFeatures)
        )
        engineered_features = list(
            self.getOrDefault(self.engineeredFeatures)
        )
        interaction_output_cols = list(
            self.getOrDefault(self.interactionOutputCols)
        )
        hgam_output_cols = list(
            self.getOrDefault(self.hgamOutputCols)
        )

        model_features: list[str] = []

        # Every numerical feature has a linear representation available.
        # If the preceding spline stage produced basis columns, use them;
        # otherwise use the original imputed linear column.
        for feature in numerical_features:
            basis_columns = self._spline_basis_columns(
                dataset,
                feature,
            )
            if basis_columns:
                model_features.extend(basis_columns)
            else:
                model_features.append(
                    f"__imputed_{feature}"
                )

        # In HGAM, a hierarchical group intercept represents the group
        # effect itself. Do not also add the same encoded grouping variable
        # as a global categorical main effect, otherwise the design matrix
        # contains duplicate group columns. Other categorical features keep
        # their existing GAM representation unchanged.
        model_categorical_features = categorical_features

        if hgam_output_cols and "__hgam_group_intercept" in hgam_output_cols:
            # The pipeline has already removed the HGAM grouping feature from
            # categoricalFeatures when a hierarchical intercept is enabled.
            # Keep this check only as documentation of that invariant.
            pass

        model_features.extend(
            f"__encoded_{feature}"
            for feature in model_categorical_features
        )
        model_features.extend(engineered_features)
        model_features.extend(interaction_output_cols)
        model_features.extend(hgam_output_cols)

        if not model_features:
            raise ValueError(
                "GAM has no model features after feature treatment."
            )

        missing = [
            column
            for column in model_features
            if column not in dataset.columns
        ]
        if missing:
            raise ValueError(
                "GAM feature assembly requires missing columns: "
                + ", ".join(missing)
            )

        if len(model_features) != len(set(model_features)):
            raise ValueError(
                "GAM model feature columns contain duplicates."
            )

        return GAMFeatureAssemblyModel(
            inputCols=json.dumps(model_features)
        )


class GAMFeatureAssemblyModel(
    Transformer,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """Apply the frozen native Spark VectorAssembler configuration."""

    inputCols = Param(
        Params._dummy(),
        "inputCols",
        "JSON-encoded frozen GAM VectorAssembler input columns.",
        TypeConverters.toString,
    )

    @keyword_only
    def __init__(
        self,
        inputCols: str = "[]",
    ) -> None:
        super().__init__()
        self._setDefault(inputCols="[]")
        self.setParams(inputCols=inputCols)

    @keyword_only
    def setParams(
        self,
        inputCols: str | None = None,
    ) -> "GAMFeatureAssemblyModel":
        return self._set(**self._input_kwargs)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        input_cols = json.loads(
            self.getOrDefault(self.inputCols)
        )
        if not input_cols:
            raise ValueError(
                "GAM feature assembly contains no input columns."
            )

        missing = [
            column
            for column in input_cols
            if column not in dataset.columns
        ]
        if missing:
            raise ValueError(
                "GAM VectorAssembler input columns are missing: "
                + ", ".join(missing)
            )

        # Keep VectorAssembler as the actual Spark ML assembly operation.
        # This model only freezes its fitted input column list so spline
        # fallback decisions remain persisted with the PipelineModel.
        return VectorAssembler(
            inputCols=input_cols,
            outputCol="features",
            handleInvalid="keep",
        ).transform(dataset)


# ======================================================================
# PIPELINE
# ======================================================================


def build_gam_pipeline(
    config: dict,
) -> Pipeline:
    """Build the complete representation-aware GAM/HGAM Spark Pipeline.

    ``model_type: gam`` preserves the existing GAM path. ``model_type: hgam``
    inserts the hierarchical group-effect stage after the existing spline and
    interaction stages.
    """

    features = config["parameters"]["modelling"]["features"]

    numerical_features = list(
        features.get("numerical_features", [])
    )
    categorical_features = list(
        features.get("categorical_features", [])
    )
    engineered_features = list(
        features.get("engineered_features", [])
    )

    gam_config = config["parameters"]["modelling"]["gam"]
    model_type = _get_gam_model_type(config)

    spline_features, _, _, _ = _get_gam_feature_groups(config)
    spline_config = gam_config["spline"]

    stages: list[Any] = _build_gam_preparation_stages(config)

    stages.append(
        GAMSplineEstimator(
            inputCols=spline_features,
            degree=int(spline_config["degree"]),
            numKnots=int(spline_config["num_knots"]),
        )
    )

    interaction_specs = _get_gam_interaction_specs(config)
    interaction_output_cols = [
        f"__interaction__{spec.left}__{spec.right}"
        for spec in interaction_specs
    ]

    if interaction_specs:
        stages.append(
            GAMInteractionEstimator(
                interactionSpecs=_encode_interaction_specs(
                    interaction_specs
                ),
                numericalFeatures=numerical_features,
                categoricalFeatures=categorical_features,
            )
        )

    # --------------------------------------------------------------
    # Optional HGAM extension
    # --------------------------------------------------------------

    hgam_output_cols: list[str] = []

    if model_type == "hgam":
        hgam_config = _get_hgam_config(config)

        grouping_feature = hgam_config["grouping"]["feature"]
        intercept_config = hgam_config.get("intercept", {})
        varying_config = hgam_config.get("varying_smooths", {})
        shrinkage_config = hgam_config.get("shrinkage", {})

        group_intercept_enabled = bool(
            intercept_config.get("enabled", True)
        )
        varying_features = (
            list(varying_config.get("features", []))
            if varying_config.get("enabled", False)
            else []
        )
        shrinkage_enabled = bool(
            shrinkage_config.get("enabled", True)
        )
        group_intercept_penalty = float(
            shrinkage_config.get("group_intercept_penalty", 1.0)
        )
        varying_smooth_penalty = float(
            shrinkage_config.get("varying_smooth_penalty", 1.0)
        )

        if group_intercept_enabled:
            hgam_output_cols.append(
                "__hgam_group_intercept"
            )

        hgam_output_cols.extend(
            f"__hgam_smooth__{feature}"
            for feature in varying_features
        )

        if group_intercept_enabled or varying_features:
            stages.append(
                HGAMGroupEffectEstimator(
                    groupingFeature=grouping_feature,
                    groupInterceptEnabled=group_intercept_enabled,
                    varyingSmoothFeatures=varying_features,
                    groupInterceptPenalty=group_intercept_penalty,
                    varyingSmoothPenalty=varying_smooth_penalty,
                    shrinkageEnabled=shrinkage_enabled,
                )
            )

    # --------------------------------------------------------------
    # Final feature assembly
    # --------------------------------------------------------------

    stages.append(
        GAMFeatureAssemblyEstimator(
            numericalFeatures=numerical_features,
            categoricalFeatures=(
                [
                    feature
                    for feature in categorical_features
                    if not (
                        model_type == "hgam"
                        and "__hgam_group_intercept" in hgam_output_cols
                        and feature == hgam_config["grouping"]["feature"]
                    )
                ]
                if model_type == "hgam"
                else categorical_features
            ),
            engineeredFeatures=engineered_features,
            interactionOutputCols=interaction_output_cols,
            hgamOutputCols=hgam_output_cols,
        )
    )

    # --------------------------------------------------------------
    # Logistic regression / configurable class weighting
    # --------------------------------------------------------------

    weighting_config = gam_config.get(
        "class_weighting",
        {},
    )

    logistic_regression_kwargs = {
        "featuresCol": "features",
        "labelCol": "label",
    }

    if weighting_config.get("enabled", False):
        logistic_regression_kwargs["weightCol"] = "__class_weight__"

    if model_type == "hgam":
        hgam_config = _get_hgam_config(config)
        regularization_config = hgam_config.get(
            "regularization",
            {},
        )
        logistic_regression_kwargs["regParam"] = float(
            regularization_config.get("reg_param", 0.01)
        )
        logistic_regression_kwargs["elasticNetParam"] = 0.0

    stages.append(
        LogisticRegression(
            **logistic_regression_kwargs
        )
    )

    return Pipeline(stages=stages)

def train_gam_spark(
    training_df: DataFrame,
    config: dict,
) -> Any:
    """Train the GAM Spark Pipeline.

    Optional experiment screening is applied ONLY to ``training_df`` before
    Pipeline.fit(). Optional class weighting is then applied to the screened
    training population.

    The resulting PipelineModel therefore transforms the complete
    validation/OOT datasets without sampling or weighting them.
    """

    if "label" not in training_df.columns:
        raise ValueError(
            "GAM training DataFrame must contain "
            "the Spark ML target column 'label'."
        )

    gam_config = (
        config["parameters"]
        ["modelling"]
        ["gam"]
    )

    experiment_config = gam_config.get(
        "experiment",
        {},
    )

    weighting_config = gam_config.get(
        "class_weighting",
        {},
    )

    screening_fraction = float(
        experiment_config.get(
            "screening_sample_fraction",
            1.0,
        )
    )

    screening_seed = int(
        experiment_config.get(
            "screening_seed",
            42,
        )
    )

    if not 0.0 < screening_fraction <= 1.0:
        raise ValueError(
            "gam.experiment.screening_sample_fraction "
            "must be between 0 and 1."
        )

    fit_df = training_df

    # Apply optional training-only screening.
    if screening_fraction < 1.0:
        fit_df = training_df.sample(
            withReplacement=False,
            fraction=screening_fraction,
            seed=screening_seed,
        )

    # Apply optional class weighting to the fitting population.
    if weighting_config.get("enabled", False):
        positive_weight = float(
            weighting_config.get(
                "positive_weight",
                1.0,
            )
        )

        negative_weight = float(
            weighting_config.get(
                "negative_weight",
                1.0,
            )
        )

        if positive_weight <= 0.0:
            raise ValueError(
                "gam.class_weighting.positive_weight "
                "must be greater than 0."
            )

        if negative_weight <= 0.0:
            raise ValueError(
                "gam.class_weighting.negative_weight "
                "must be greater than 0."
            )

        fit_df = fit_df.withColumn(
            "__class_weight__",
            F.when(
                F.col("label") == 1,
                F.lit(positive_weight),
            ).otherwise(
                F.lit(negative_weight)
            ),
        )

    pipeline = build_gam_pipeline(
        config
    )

    return pipeline.fit(
        fit_df
    )
