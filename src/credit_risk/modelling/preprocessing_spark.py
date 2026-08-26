"""Spark modelling feature preprocessing utilities."""

from __future__ import annotations

from typing import Any

from pyspark import keyword_only
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import (
    Imputer,
    OneHotEncoder,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.param.shared import Param, Params, TypeConverters
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class CategoricalNullHandler(
    Transformer,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """
    Spark ML Transformer that replaces null categorical values with
    the configured 'Unknown' value.

    This transformer contains no learned state and is therefore fully
    reproducible when persisted as part of a Spark PipelineModel.
    """

    columns = Param(
        Params._dummy(),
        "columns",
        "Categorical columns to clean.",
        typeConverter=TypeConverters.toList,
    )

    unknown_value = Param(
        Params._dummy(),
        "unknown_value",
        "Replacement value for null categorical values.",
        typeConverter=TypeConverters.toString,
    )

    @keyword_only
    def __init__(
        self,
        *,
        columns: list[str] | None = None,
        unknown_value: str = "Unknown",
    ) -> None:
        super().__init__()

        self._setDefault(
            unknown_value="Unknown",
        )

        kwargs = self._input_kwargs

        self._set(
            **kwargs,
        )

    def _transform(
        self,
        dataset: DataFrame,
    ) -> DataFrame:
        columns = self.getOrDefault(
            self.columns,
        )

        unknown_value = self.getOrDefault(
            self.unknown_value,
        )

        result = dataset

        for column in columns:
            result = result.withColumn(
                column,
                F.coalesce(
                    F.col(column).cast("string"),
                    F.lit(unknown_value),
                ),
            )

        return result


def split_features_target_spark(
    df: DataFrame,
    config: dict,
) -> tuple[DataFrame, DataFrame]:
    """Separate baseline predictors from the modelling target."""

    model_features = (
        config["parameters"]["modelling"]["features"]["numerical_features"]
        + config["parameters"]["modelling"]["features"]["categorical_features"]
        + config["parameters"]["modelling"]["features"]["engineered_features"]
    )

    target = config["parameters"]["target"]["name"]
    approach = config["parameters"]["modelling_approach"]

    # ------------------------------------------------------------------
    # Determine the natural modelling grain.
    # ------------------------------------------------------------------

    if approach == "origination":

        grain_columns = [
            "loan_id",
        ]

    elif approach == "behavioral":

        grain_columns = [
            "loan_id",
            "calculated_loan_age",
        ]

    else:
        raise ValueError(f"Unsupported modelling approach: {approach}")

    # ------------------------------------------------------------------
    # Validate required columns.
    # ------------------------------------------------------------------

    required_columns = grain_columns + model_features + [target]

    missing_columns = sorted(set(required_columns) - set(df.columns))

    if missing_columns:
        raise ValueError("Missing modelling columns: " + ", ".join(missing_columns))

    # ------------------------------------------------------------------
    # Validate target.
    # ------------------------------------------------------------------

    if df.filter(F.col(target).isNull()).limit(1).count() > 0:
        raise ValueError(f"Target column contains missing values: {target}")

    # ------------------------------------------------------------------
    # X contains the natural grain columns so that the transformed
    # feature population can later be safely joined to y.
    #
    # If a grain column is also a modelling feature, keep it only once.
    # ------------------------------------------------------------------

    X_columns = list(dict.fromkeys(grain_columns + model_features))

    X = df.select(*X_columns)

    # ------------------------------------------------------------------
    # y contains the same natural grain columns + target.
    # ------------------------------------------------------------------

    y_columns = list(dict.fromkeys(grain_columns + [target]))

    y = df.select(*y_columns)

    return X, y


def build_preprocessor_spark(
    config: dict,
) -> Pipeline:
    """
    Build the complete, self-contained Spark preprocessing pipeline.

    Numerical features:
        median imputation.

    Categorical features:
        null -> "Unknown"
        -> StringIndexer
        -> OneHotEncoder

    Engineered features:
        passthrough.

    Final output:
        features

    The returned Pipeline is unfitted.
    """

    features_config = config["parameters"]["modelling"]["features"]

    numerical_features = features_config["numerical_features"]

    categorical_features = features_config["categorical_features"]

    engineered_features = features_config["engineered_features"]

    stages: list[Any] = []

    # --------------------------------------------------------------
    # Categorical null handling
    #
    # This is deliberately a Pipeline stage so it becomes part of
    # the persisted PipelineModel.
    # --------------------------------------------------------------

    if categorical_features:

        stages.append(
            CategoricalNullHandler(
                columns=categorical_features,
                unknown_value="Unknown",
            )
        )

    # --------------------------------------------------------------
    # Numerical median imputation
    # --------------------------------------------------------------

    numerical_output_columns = []

    if numerical_features:

        numerical_output_columns = [
            f"__imputed_{column}" for column in numerical_features
        ]

        stages.append(
            Imputer(
                inputCols=numerical_features,
                outputCols=numerical_output_columns,
                strategy="median",
            )
        )

    # --------------------------------------------------------------
    # Categorical indexing
    #
    # handleInvalid="keep" is the Spark equivalent of allowing
    # unseen categories during validation/OOT/scoring.
    # --------------------------------------------------------------

    indexed_columns = []

    for column in categorical_features:

        indexed_column = f"__indexed_{column}"

        stages.append(
            StringIndexer(
                inputCol=column,
                outputCol=indexed_column,
                handleInvalid="keep",
            )
        )

        indexed_columns.append(
            indexed_column,
        )

    # --------------------------------------------------------------
    # One-hot encoding
    # --------------------------------------------------------------

    encoded_columns = []

    if categorical_features:

        encoded_columns = [f"__encoded_{column}" for column in categorical_features]

        stages.append(
            OneHotEncoder(
                inputCols=indexed_columns,
                outputCols=encoded_columns,
                handleInvalid="keep",
            )
        )

    # --------------------------------------------------------------
    # Assemble final feature vector
    # --------------------------------------------------------------

    assembler_inputs = numerical_output_columns + encoded_columns + engineered_features

    if not assembler_inputs:
        raise ValueError(
            "No modelling features configured. "
            "At least one numerical, categorical, or engineered "
            "feature is required."
        )

    stages.append(
        VectorAssembler(
            inputCols=assembler_inputs,
            outputCol="features",
            handleInvalid="keep",
        )
    )

    return Pipeline(
        stages=stages,
    )


def fit_preprocessor_spark(
    X_train: DataFrame,
    config: dict,
) -> PipelineModel:
    """
    Fit the complete Spark preprocessing pipeline on training data.

    Returns the fitted PipelineModel that should be persisted.
    """

    preprocessor = build_preprocessor_spark(
        config,
    )

    return preprocessor.fit(
        X_train,
    )


def transform_with_preprocessor_spark(
    X: DataFrame,
    preprocessor_model: PipelineModel,
) -> DataFrame:
    """
    Transform data using a previously fitted PipelineModel.

    No preprocessing configuration or helper transformation is
    required at inference time.
    """

    return preprocessor_model.transform(
        X,
    )
