"""Spark GAM V1 modelling utilities.

GAM V1 is a strictly additive Generalized Additive Model:

    logit(PD) = beta_0 + sum_j f_j(X_j)

Selected numerical features are represented by cubic B-spline bases.
Remaining numerical features enter linearly. Categorical variables use
Spark StringIndexer/OneHotEncoder.

The complete model is one Spark PipelineModel. Preprocessing, fitted
spline knots, feature assembly, and logistic-regression coefficients are
therefore persisted with the PipelineModel.

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
) -> dict[str, GAMSplineSpec]:
    """
    Fit spline specifications using observed training values only.

    Knot estimation is performed on the original feature columns.
    Null/NaN values are excluded from knot estimation.

    Missing values are handled separately by the GAM preparation/imputation
    stage during transformation. This prevents median imputation from
    artificially creating repeated quantile knots.
    """

    if not features:
        raise ValueError(
            "GAM requires at least one spline feature."
        )

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
    # Median imputation can create an artificial mass at one value and
    # therefore produce repeated quantile knots.
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

    # One Spark action for all spline features.
    row = df.select(
        *aggregation_expressions,
    ).first()

    if row is None:
        raise ValueError(
            "Unable to fit GAM spline specifications: "
            "training DataFrame is empty."
        )

    specs: dict[str, GAMSplineSpec] = {}

    for feature in features:
        valid_count = row[f"__valid_count_{feature}"]

        if valid_count is None or valid_count == 0:
            raise ValueError(
                f"Spline feature '{feature}' has no valid observed "
                "training values."
            )

        quantiles = row[f"__quantiles_{feature}"]

        if quantiles is None:
            raise ValueError(
                f"Failed to compute spline quantiles for '{feature}'."
            )

        quantiles = [
            float(value)
            for value in quantiles
            if value is not None
        ]

        if len(quantiles) != num_knots:
            raise ValueError(
                f"Failed to compute {num_knots} spline quantiles "
                f"for '{feature}'. Got {len(quantiles)}."
            )

        if not np.isfinite(quantiles).all():
            raise ValueError(
                f"Non-finite spline knot values generated "
                f"for '{feature}': {quantiles}"
            )

        lower_bound = quantiles[0]
        upper_bound = quantiles[-1]

        if lower_bound >= upper_bound:
            raise ValueError(
                f"Spline feature '{feature}' has insufficient variation "
                f"in the observed training data: "
                f"lower_bound={lower_bound}, "
                f"upper_bound={upper_bound}, "
                f"valid_count={valid_count}, "
                f"quantiles={quantiles}."
            )

        internal_knots = quantiles[1:-1]

        all_knots = [
            lower_bound,
            *internal_knots,
            upper_bound,
        ]

        if any(
            right <= left
            for left, right in zip(
                all_knots,
                all_knots[1:],
            )
        ):
            raise ValueError(
                f"Spline feature '{feature}' has repeated quantile "
                f"knots in observed training data. "
                f"quantiles={quantiles}. "
                f"Reduce num_knots or choose a feature with greater "
                f"variation."
            )

        specs[feature] = GAMSplineSpec(
            feature=feature,
            degree=degree,
            knots=internal_knots,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    return specs



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

        specs = _fit_gam_spline_specs_spark(
            df=dataset,
            features=features,
            degree=degree,
            num_knots=num_knots,
        )

        return GAMSplineModel(
            splineSpecs=_encode_spline_specs(specs),
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

    @keyword_only
    def __init__(
        self,
        splineSpecs: str | None = None,
        inputCols: list[str] | None = None,
    ) -> None:

        super().__init__()

        self._setDefault(
            splineSpecs="{}",
            inputCols=[],
        )

        self.setParams(
            splineSpecs=splineSpecs or "{}",
            inputCols=inputCols or [],
        )

    @keyword_only
    def setParams(
        self,
        splineSpecs: str | None = None,
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

        features = list(
            self.getOrDefault(
                self.inputCols
            )
        )

        if not features:
            raise ValueError(
                "Fitted GAM spline model contains no spline features."
            )

        result = dataset

        for feature in features:

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
    Determine strict additive GAM V1 feature treatment.

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
# PIPELINE
# ======================================================================


def build_gam_pipeline(
    config: dict,
) -> Pipeline:
    """Build the complete strictly additive GAM V1 pipeline."""

    spline_features, _, _, _ = (
        _get_gam_feature_groups(
            config
        )
    )

    model_features = (
        _get_gam_model_features(
            config
        )
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

    stages: list[Any] = (
        _build_gam_preparation_stages(
            config
        )
    )

    # --------------------------------------------------------------
    # Fit spline knots on training data.
    # --------------------------------------------------------------

    stages.append(
        GAMSplineEstimator(
            inputCols=spline_features,
            degree=int(
                spline_config["degree"]
            ),
            numKnots=int(
                spline_config["num_knots"]
            ),
        )
    )

    # --------------------------------------------------------------
    # Assemble final additive feature vector.
    # --------------------------------------------------------------

    stages.append(
        VectorAssembler(
            inputCols=model_features,
            outputCol="features",
            handleInvalid="keep",
        )
    )

    # --------------------------------------------------------------
    # Logistic regression.
    # --------------------------------------------------------------

    stages.append(
        LogisticRegression(
            featuresCol="features",
            labelCol="label",
        )
    )

    return Pipeline(
        stages=stages
    )


# ======================================================================
# TRAINING ENTRY POINT
# ======================================================================


def train_gam_spark(
    training_df: DataFrame,
    config: dict,
) -> Any:
    """
    Fit and return the complete serializable GAM PipelineModel.
    """

    if "label" not in training_df.columns:
        raise ValueError(
            "GAM training DataFrame must contain "
            "the Spark ML target column 'label'."
        )

    pipeline = build_gam_pipeline(
        config
    )

    return pipeline.fit(
        training_df
    )