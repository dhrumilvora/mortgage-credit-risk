from __future__ import annotations
from pyspark.sql import DataFrame, functions as F


def _prepare_training_data_spark(
    X_train: DataFrame,
    y_train: DataFrame,
    config: dict,
) -> DataFrame:
    """Join transformed features and target into a Spark ML dataset."""

    approach = config["parameters"]["modelling_approach"]
    target = config["parameters"]["target"]["name"]

    # --------------------------------------------------------------
    # Determine the natural modelling grain.
    # --------------------------------------------------------------

    if approach == "origination":
        join_keys = ["loan_id"]

    elif approach == "behavioral":
        join_keys = [
            "loan_id",
            "calculated_loan_age",
        ]

    else:
        raise ValueError(f"Unsupported modelling approach: {approach}")

    # --------------------------------------------------------------
    # Validate join columns.
    # --------------------------------------------------------------

    missing_x = sorted(set(join_keys) - set(X_train.columns))

    if missing_x:
        raise ValueError(
            "Missing training feature join columns: " + ", ".join(missing_x)
        )

    missing_y = sorted(set(join_keys + [target]) - set(y_train.columns))

    if missing_y:
        raise ValueError("Missing training target columns: " + ", ".join(missing_y))

    # --------------------------------------------------------------
    # Validate uniqueness of the target grain.
    #
    # There should be exactly one target per modelling observation.
    # --------------------------------------------------------------

    duplicate_targets = (
        y_train.groupBy(*join_keys).count().filter(F.col("count") > 1).limit(1).count()
    )

    if duplicate_targets:
        raise ValueError(
            "Duplicate target observations detected for "
            "the modelling grain: " + ", ".join(join_keys)
        )

    # --------------------------------------------------------------
    # Join transformed features to target.
    # --------------------------------------------------------------

    training_df = X_train.join(
        y_train.select(
            *join_keys,
            F.col(target).alias("label"),
        ),
        on=join_keys,
        how="inner",
    ).select(
        "features",
        "label",
    )

    # --------------------------------------------------------------
    # Validate that the join did not lose observations.
    # --------------------------------------------------------------

    X_count = X_train.count()
    training_count = training_df.count()

    if X_count != training_count:
        raise ValueError(
            "Feature/target join changed the training population: "
            f"features={X_count:,}, "
            f"joined={training_count:,}."
        )

    return training_df


def train_model_spark(X_train: DataFrame, y_train: DataFrame, config: dict):
    algorithm = config["parameters"]["modelling"]["algorithm"]
    target = config["parameters"]["target"]["name"]
    if X_train.limit(1).count() == 0:
        raise ValueError("Training feature DataFrame is empty.")

    if y_train.limit(1).count() == 0:
        raise ValueError("Training target DataFrame is empty.")
    X_train_count = X_train.count()
    y_train_count = y_train.count()

    if X_train_count != y_train_count:
        raise ValueError(
            "Training features and target contain different "
            "numbers of rows: "
            f"X_train={X_train_count:,}, "
            f"y_train={y_train_count:,}."
        )

    # --------------------------------------------------------------
    # Validate target contains at least two classes
    #
    # Equivalent to:
    #
    #     y_train.nunique() < 2
    # --------------------------------------------------------------

    target_class_count = y_train.select(target).distinct().count()

    if target_class_count < 2:
        raise ValueError("Training target must contain at least two classes.")
    training_df = _prepare_training_data_spark(
        X_train,
        y_train,
        config,
    )
    if algorithm == "logistic_regression":
        from credit_risk.modelling.models.spark.logistic_regression import (
            train_logistic_regression_spark,
        )

        return train_logistic_regression_spark(training_df, config)
    elif algorithm == "random_forest":
        from credit_risk.modelling.models.spark.random_forest import (
            train_random_forest_spark,
        )

        return train_random_forest_spark(training_df, config)
    elif algorithm == "xgboost":
        from credit_risk.modelling.models.spark.xgboost import train_xgboost_spark

        return train_xgboost_spark(training_df, config)

    raise ValueError(f"Unsupported modelling algorithm: {algorithm}")
