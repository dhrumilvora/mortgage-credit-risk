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
    algorithm = config["parameters"]["modelling"]["algorithm"]

    # --------------------------------------------------------------
    # Determine natural modelling grain
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
    # Validate columns locally.
    # No Spark job required.
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
    # Prepare target
    # --------------------------------------------------------------

    target_df = y_train.select(
        *join_keys,
        F.col(target).alias("label"),
    )

    # --------------------------------------------------------------
    # Join
    # --------------------------------------------------------------

    joined_df = X_train.join(
        target_df,
        on=join_keys,
        how="inner",
    )

    # --------------------------------------------------------------
    # GAM requires the individually transformed feature columns.
    #
    # The GAM spline transformation happens after preprocessing,
    # before final VectorAssembler.
    # --------------------------------------------------------------

    if algorithm == "gam":
    
        features = config["parameters"]["modelling"]["features"]
    
        feature_columns = (
            features.get("numerical_features", [])
            + features.get("categorical_features", [])
            + features.get("engineered_features", [])
        )
    
        return joined_df.select(
            *feature_columns,
            "label",
        )
    # --------------------------------------------------------------
    # Existing models receive the assembled feature vector.
    # --------------------------------------------------------------

    return joined_df.select(
        "features",
        "label",
    )


def train_model_spark(
    X_train: DataFrame,
    y_train: DataFrame,
    X_val: DataFrame,
    y_val: DataFrame,
    config: dict,
):
    algorithm = config["parameters"]["modelling"]["algorithm"]
    target = config["parameters"]["target"]["name"]

    # --------------------------------------------------------------
    # Basic validation using metadata/schema only
    # --------------------------------------------------------------

    if not X_train.columns:
        raise ValueError("Training feature DataFrame has no columns.")

    if not y_train.columns:
        raise ValueError("Training target DataFrame has no columns.")

    # --------------------------------------------------------------
    # Validate target column locally
    # --------------------------------------------------------------

    if target not in y_train.columns:
        raise ValueError(f"Target column '{target}' not found in y_train.")

    # --------------------------------------------------------------
    # Prepare training data
    # --------------------------------------------------------------

    training_df = _prepare_training_data_spark(
        X_train,
        y_train,
        config,
    )
    val_df = _prepare_training_data_spark(X_val, y_val, config)
    # --------------------------------------------------------------
    # Materialize ONCE.
    #
    # This gives us one actual validation count and prevents
    # repeated recomputation when the model subsequently consumes
    # the same dataframe.
    # --------------------------------------------------------------

    training_count = training_df.count()

    if training_count == 0:
        raise ValueError("Training feature/target join produced zero rows.")

    # --------------------------------------------------------------
    # Validate target classes.
    #
    # This is a small aggregation compared with the previous
    # full-data duplicate check.
    # --------------------------------------------------------------

    target_classes = training_df.select("label").distinct().limit(2).collect()

    if len(target_classes) < 2:

        raise ValueError("Training target must contain at least two classes.")

    # --------------------------------------------------------------
    # Train model
    # --------------------------------------------------------------

    if algorithm == "logistic_regression":

        from credit_risk.modelling.models.spark.logistic_regression import (
            train_logistic_regression_spark,
        )

        return train_logistic_regression_spark(
            training_df,
            config,
        )

    elif algorithm == "random_forest":

        from credit_risk.modelling.models.spark.random_forest import (
            train_random_forest_spark,
        )

        return train_random_forest_spark(
            training_df,
            config,
        )

    elif algorithm == "xgboost":

        from credit_risk.modelling.models.spark.xgboost import (
            train_xgboost_spark,
        )

        train_xgb = training_df.withColumn("is_validation", F.lit(False)).unionByName(
            val_df.withColumn("is_validation", F.lit(True))
        )

        del training_df
        del val_df
        return train_xgboost_spark(
            train_xgb,
            config,
        )
    elif algorithm == "gam":

        from credit_risk.modelling.models.spark.gam import train_gam_spark

        return train_gam_spark(
            training_df,
            config,
        )
        

    raise ValueError(f"Unsupported modelling algorithm: {algorithm}")
