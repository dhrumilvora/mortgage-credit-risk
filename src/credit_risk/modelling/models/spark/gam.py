"""Spark GAM V1 modelling utilities.

GAM V1 is a strictly additive Generalized Additive Model:

    logit(PD) = beta_0 + sum_j f_j(X_j)

Selected numerical features are represented by cubic B-spline bases.
Remaining numerical features enter linearly. Categorical variables use
the existing Spark StringIndexer/OneHotEncoder treatment.

The complete model is one Spark PipelineModel. All learned state,
including preprocessing state, spline knots, feature assembly, and
logistic-regression coefficients, is therefore persisted with the
PipelineModel.

V1 deliberately contains NO interaction terms.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from pyspark import keyword_only
from pyspark.ml import Estimator, Pipeline, Transformer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.column import Column


@dataclass
class GAMSplineSpec:
    """Fitted B-spline specification for one logical GAM feature."""

    feature: str
    degree: int
    knots: list[float]
    lower_bound: float
    upper_bound: float


def _bspline_basis_expression(
    x: Column,
    knots: list[float],
    degree: int,
    basis_index: int,
) -> Column:
    """Build one B-spline basis function using Cox-de Boor recursion."""

    def basis_zero(index: int) -> Column:
        left = float(knots[index])
        right = float(knots[index + 1])

        if index == len(knots) - 2:
            condition = (x >= F.lit(left)) & (x <= F.lit(right))
        else:
            condition = (x >= F.lit(left)) & (x < F.lit(right))

        return condition.cast("double")

    def basis(index: int, current_degree: int) -> Column:
        if current_degree == 0:
            return basis_zero(index)

        left_knot = float(knots[index])
        left_next_knot = float(knots[index + current_degree])
        right_knot = float(knots[index + current_degree + 1])
        right_next_knot = float(knots[index + 1])

        left_basis = basis(index, current_degree - 1)
        right_basis = basis(index + 1, current_degree - 1)

        if left_next_knot == left_knot:
            left_term = F.lit(0.0)
        else:
            left_term = (
                (x - F.lit(left_knot)) / F.lit(left_next_knot - left_knot) * left_basis
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

    return basis(basis_index, degree)


def build_full_knot_vector(spec: GAMSplineSpec) -> list[float]:
    """Build the clamped knot vector from internal knots and bounds."""

    repetitions = spec.degree + 1

    return (
        [spec.lower_bound] * repetitions + spec.knots + [spec.upper_bound] * repetitions
    )


def _number_of_basis_functions(spec: GAMSplineSpec) -> int:
    """Return the number of basis functions for a fitted spline."""

    return len(build_full_knot_vector(spec)) - spec.degree - 1


def _fit_gam_spline_spec_spark(
    df: DataFrame,
    feature: str,
    transformed_feature: str,
    degree: int,
    num_knots: int,
) -> GAMSplineSpec:
    """Fit one spline specification using training data only."""

    if transformed_feature not in df.columns:
        raise ValueError(
            f"Spline feature '{transformed_feature}' not found in DataFrame."
        )
    if degree < 1:
        raise ValueError(f"Spline degree must be at least 1. Got: {degree}")
    if num_knots < 2:
        raise ValueError(f"Number of spline knots must be at least 2. Got: {num_knots}")

    positions = np.linspace(0.0, 1.0, num_knots).tolist()

    quantiles = df.select(
        F.percentile_approx(
            F.col(transformed_feature),
            positions,
            10000,
        ).alias("quantiles")
    ).first()["quantiles"]

    if quantiles is None or len(quantiles) != num_knots:
        raise ValueError(
            f"Failed to compute {num_knots} spline quantiles for '{feature}'."
        )

    quantiles = [float(value) for value in quantiles if value is not None]

    if len(quantiles) != num_knots:
        raise ValueError(
            f"Feature '{feature}' contains insufficient non-null values "
            "to construct the spline."
        )

    if not np.isfinite(quantiles).all():
        raise ValueError(f"Non-finite spline knot values generated for '{feature}'.")

    lower_bound = quantiles[0]
    upper_bound = quantiles[-1]

    if lower_bound >= upper_bound:
        raise ValueError(
            f"Spline feature '{feature}' has insufficient variation in "
            f"the training data: lower_bound={lower_bound}, "
            f"upper_bound={upper_bound}."
        )

    internal_knots = quantiles[1:-1]

    if any(
        right <= left
        for left, right in zip(
            [lower_bound, *internal_knots],
            [*internal_knots, upper_bound],
        )
    ):
        raise ValueError(
            f"Spline feature '{feature}' has repeated quantile knots. "
            "Reduce num_knots or use a feature with sufficient variation."
        )

    return GAMSplineSpec(
        feature=feature,
        degree=degree,
        knots=internal_knots,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )


def _encode_spline_specs(specs: dict[str, GAMSplineSpec]) -> str:
    """Serialize spline specifications into a Spark Param string."""

    return json.dumps(
        {feature: asdict(spec) for feature, spec in specs.items()},
        sort_keys=True,
    )


def _decode_spline_specs(value: str) -> dict[str, GAMSplineSpec]:
    """Deserialize spline specifications from a Spark Param string."""

    payload = json.loads(value)

    return {
        feature: GAMSplineSpec(
            feature=data["feature"],
            degree=int(data["degree"]),
            knots=[float(knot) for knot in data["knots"]],
            lower_bound=float(data["lower_bound"]),
            upper_bound=float(data["upper_bound"]),
        )
        for feature, data in payload.items()
    }


class GAMPreparationTransformer(
    Transformer,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """Apply deterministic GAM input preparation inside the Pipeline."""

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
        return self._set(**self._input_kwargs)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        result = dataset

        for column in self.getOrDefault(self.numericalNullColumns):
            if column in result.columns:
                result = result.withColumn(
                    column,
                    F.coalesce(F.col(column), F.lit(-1.0)),
                )

        for column in self.getOrDefault(self.categoricalColumns):
            if column in result.columns:
                result = result.withColumn(
                    column,
                    F.coalesce(
                        F.col(column).cast("string"),
                        F.lit("Unknown"),
                    ),
                )

        return result


class GAMSplineEstimator(
    Estimator,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """Fit training-only B-spline knots for configured numerical features."""

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
        "Number of quantile knots including both boundaries.",
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
        return self._set(**self._input_kwargs)

    def _fit(self, dataset: DataFrame) -> "GAMSplineModel":
        features = list(self.getOrDefault(self.inputCols))
        degree = int(self.getOrDefault(self.degree))
        num_knots = int(self.getOrDefault(self.numKnots))

        if not features:
            raise ValueError("GAM requires at least one spline feature.")
        if len(features) != len(set(features)):
            raise ValueError("GAM spline features must be unique.")

        specs = {}

        for feature in features:
            specs[feature] = _fit_gam_spline_spec_spark(
                df=dataset,
                feature=feature,
                transformed_feature=f"__imputed_{feature}",
                degree=degree,
                num_knots=num_knots,
            )

        return GAMSplineModel(
            splineSpecs=_encode_spline_specs(specs),
            inputCols=features,
        )


class GAMSplineModel(
    Transformer,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """Apply frozen spline specifications learned by GAMSplineEstimator."""

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

    @keyword_only
    def __init__(
        self,
        splineSpecs: str,
        inputCols: list[str],
    ) -> None:
        super().__init__()
        self._set(
            splineSpecs=splineSpecs,
            inputCols=inputCols,
        )

    def _transform(self, dataset: DataFrame) -> DataFrame:
        specs = _decode_spline_specs(self.getOrDefault(self.splineSpecs))
        result = dataset

        for feature in self.getOrDefault(self.inputCols):
            if feature not in specs:
                raise ValueError(
                    f"No fitted spline specification found for '{feature}'."
                )

            spec = specs[feature]
            transformed_feature = f"__imputed_{feature}"

            if transformed_feature not in result.columns:
                raise ValueError(
                    f"Expected imputed spline feature "
                    f"'{transformed_feature}' not found in input DataFrame."
                )

            full_knots = build_full_knot_vector(spec)
            x = F.col(transformed_feature)

            for basis_index in range(_number_of_basis_functions(spec)):
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


def _get_gam_feature_groups(
    config: dict,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Determine strict additive GAM V1 feature treatment."""

    features = config["parameters"]["modelling"]["features"]

    numerical_features = list(features.get("numerical_features", []))
    categorical_features = list(features.get("categorical_features", []))
    engineered_features = list(features.get("engineered_features", []))

    gam_config = config["parameters"]["modelling"]["gam"]
    spline_features = list(gam_config["feature_transform"]["spline"])

    if len(spline_features) != len(set(spline_features)):
        raise ValueError("GAM spline feature configuration contains duplicates.")

    numerical_set = set(numerical_features)
    unknown = sorted(set(spline_features) - numerical_set)

    if unknown:
        raise ValueError(
            "GAM spline features must be configured numerical features. "
            f"Unknown features: {unknown}"
        )

    linear_numerical_features = [
        f"__imputed_{feature}"
        for feature in numerical_features
        if feature not in set(spline_features)
    ]

    return (
        spline_features,
        linear_numerical_features,
        categorical_features,
        engineered_features,
    )


def _get_gam_model_features(config: dict) -> list[str]:
    """Build the final additive GAM feature-column list."""

    (
        spline_features,
        linear_numerical_features,
        categorical_features,
        engineered_features,
    ) = _get_gam_feature_groups(config)

    spline_config = config["parameters"]["modelling"]["gam"]["spline"]
    degree = int(spline_config["degree"])
    num_knots = int(spline_config["num_knots"])

    # K quantile knots + d+1 repeated boundaries produce
    # K+d-1 basis functions.
    num_basis = num_knots + degree - 1

    spline_basis_features = [
        f"{feature}_spline_{basis_index}"
        for feature in spline_features
        for basis_index in range(num_basis)
    ]

    encoded_categorical_features = [
        f"__encoded_{feature}" for feature in categorical_features
    ]

    return (
        linear_numerical_features
        + spline_basis_features
        + encoded_categorical_features
        + engineered_features
    )


def _build_gam_preparation_stages(config: dict) -> list[Any]:
    """Build all preprocessing stages that belong inside the GAM Pipeline."""

    features = config["parameters"]["modelling"]["features"]

    numerical_features = list(features.get("numerical_features", []))
    categorical_features = list(features.get("categorical_features", []))

    stages: list[Any] = []

    # Keep the existing deterministic preprocessing semantics inside the
    # persisted model artifact.
    null_numerical_columns = (
        ["months_since_last_delinquency"]
        if "months_since_last_delinquency" in numerical_features
        else []
    )

    stages.append(
        GAMPreparationTransformer(
            numericalNullColumns=null_numerical_columns,
            categoricalColumns=categorical_features,
        )
    )

    if numerical_features:
        stages.append(
            Imputer(
                inputCols=numerical_features,
                outputCols=[f"__imputed_{feature}" for feature in numerical_features],
                strategy="median",
            )
        )

    indexed_columns = []

    for feature in categorical_features:
        indexed = f"__indexed_{feature}"

        stages.append(
            StringIndexer(
                inputCol=feature,
                outputCol=indexed,
                handleInvalid="keep",
            )
        )
        indexed_columns.append(indexed)

    if categorical_features:
        stages.append(
            OneHotEncoder(
                inputCols=indexed_columns,
                outputCols=[f"__encoded_{feature}" for feature in categorical_features],
                handleInvalid="keep",
            )
        )

    return stages


def build_gam_pipeline(config: dict) -> Pipeline:
    """Build the complete strictly additive GAM V1 Spark Pipeline."""

    spline_features, _, _, _ = _get_gam_feature_groups(config)
    model_features = _get_gam_model_features(config)

    spline_config = config["parameters"]["modelling"]["gam"]["spline"]

    stages: list[Any] = _build_gam_preparation_stages(config)

    stages.append(
        GAMSplineEstimator(
            inputCols=spline_features,
            degree=int(spline_config["degree"]),
            numKnots=int(spline_config["num_knots"]),
        )
    )

    stages.append(
        VectorAssembler(
            inputCols=model_features,
            outputCol="features",
            handleInvalid="keep",
        )
    )

    stages.append(
        LogisticRegression(
            featuresCol="features",
            labelCol="label",
        )
    )

    return Pipeline(stages=stages)


def train_gam_spark(
    training_df: DataFrame,
    config: dict,
) -> Any:
    """Fit and return the complete serializable GAM PipelineModel."""

    return build_gam_pipeline(config).fit(training_df)
