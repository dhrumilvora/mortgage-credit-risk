"""Spark modelling feature preprocessing utilities."""

from __future__ import annotations

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Imputer,
    OneHotEncoder,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def split_features_target_spark(
    df: DataFrame,
    config: dict,
) -> tuple[DataFrame, DataFrame]:
    """Separate modelling predictors and target."""

    model_features = (
        config["parameters"]["modelling"]["features"]["numerical_features"]
        + config["parameters"]["modelling"]["features"]["categorical_features"]
        + config["parameters"]["modelling"]["features"]["engineered_features"]
    )

    target = config["parameters"]["target"]["name"]
    approach = config["parameters"]["modelling_approach"]

    # --------------------------------------------------------------
    # Determine natural modelling grain.
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # Validate required columns.
    # --------------------------------------------------------------

    required_columns = grain_columns + model_features + [target]

    missing_columns = sorted(set(required_columns) - set(df.columns))

    if missing_columns:
        raise ValueError("Missing modelling columns: " + ", ".join(missing_columns))

    # --------------------------------------------------------------
    # Validate target.
    # --------------------------------------------------------------

    if df.filter(F.col(target).isNull()).limit(1).count() > 0:
        raise ValueError(f"Target column contains missing values: {target}")

    # --------------------------------------------------------------
    # Select X.
    #
    # A grain column may also be a modelling feature.
    # dict.fromkeys() preserves order while removing duplicates.
    # --------------------------------------------------------------

    X_columns = list(dict.fromkeys(grain_columns + model_features))

    X = df.select(*X_columns)

    # --------------------------------------------------------------
    # Select y.
    # --------------------------------------------------------------

    y_columns = list(dict.fromkeys(grain_columns + [target]))

    y = df.select(*y_columns)

    return X, y


def prepare_features_spark(
    df: DataFrame,
    config: dict,
) -> DataFrame:
    """
    Apply deterministic feature preparation before Spark ML.

    Categorical null values are replaced with 'Unknown', matching
    the existing Pandas preprocessing behavior.

    This operation contains no learned state and is therefore not
    part of the persisted PipelineModel.
    """

    categorical_features = config["parameters"]["modelling"]["features"][
        "categorical_features"
    ]
    df = df.withColumn(
        "months_since_last_delinquency",
        F.coalesce(F.col("months_since_last_delinquency"), F.lit(-1)),
    )
    result = df

    for column in categorical_features:

        result = result.withColumn(
            column,
            F.coalesce(
                F.col(column).cast("string"),
                F.lit("Unknown"),
            ),
        )

    return result


def build_preprocessor_spark(
    config: dict,
    assemble_features: bool = True,
) -> Pipeline:
    """
    Build the Spark preprocessing pipeline.

    Numerical features:
        median imputation.

    Categorical features:
        StringIndexer
        OneHotEncoder

    Engineered features:
        passthrough.

    Final output:
        features

    All stages are native Spark ML stages so that the fitted
    PipelineModel can be persisted and reloaded reliably.
    """

    features_config = config["parameters"]["modelling"]["features"]

    numerical_features = features_config["numerical_features"]

    categorical_features = features_config["categorical_features"]

    engineered_features = features_config["engineered_features"]

    stages = []

    # --------------------------------------------------------------
    # Numerical median imputation.
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
    # Categorical indexing.
    #
    # handleInvalid="keep" ensures unseen categories in validation
    # and OOT do not cause transformation failures.
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

        indexed_columns.append(indexed_column)

    # --------------------------------------------------------------
    # One-hot encoding.
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
    # Assemble final feature vector.
    # --------------------------------------------------------------

    assembler_inputs = numerical_output_columns + encoded_columns + engineered_features

    # --------------------------------------------------------------
    # Assemble final feature vector.
    #
    # GAM requires access to the individually transformed columns
    # before final vector assembly so that spline basis expansion
    # can be applied to numerical features.
    # --------------------------------------------------------------

    if assemble_features:

        assembler_inputs = (
            numerical_output_columns + encoded_columns + engineered_features
        )

        if not assembler_inputs:
            raise ValueError(
                "No modelling features configured. "
                "At least one numerical, categorical, or "
                "engineered feature is required."
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
    assemble_features: bool = True,
) -> PipelineModel:
    """
    Prepare and fit the Spark preprocessing pipeline.

    Returns the fitted PipelineModel that should be persisted.
    """

    X_train_prepared = prepare_features_spark(
        X_train,
        config,
    )

    preprocessor = build_preprocessor_spark(
        config,
        assemble_features=assemble_features,
    )

    return preprocessor.fit(
        X_train_prepared,
    )


def transform_with_preprocessor_spark(
    X: DataFrame,
    preprocessor_model: PipelineModel,
    config: dict,
) -> DataFrame:
    """
    Prepare and transform data using a previously fitted
    Spark preprocessing PipelineModel.
    """

    X_prepared = prepare_features_spark(
        X,
        config,
    )

    return preprocessor_model.transform(
        X_prepared,
    )
