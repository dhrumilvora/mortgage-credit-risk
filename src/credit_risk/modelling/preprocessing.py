from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


def split_features_target(
    df: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate baseline predictors from the modelling target."""
    MODEL_FEATURES = (
        config["parameters"]["modelling"]["features"]["numerical_features"]
        + config["parameters"]["modelling"]["features"]["categorical_features"]
        + config["parameters"]["modelling"]["features"]["engineered_features"]
    )
    target = config["parameters"]["target"]["name"]

    missing_features = sorted(set(MODEL_FEATURES) - set(df.columns))

    if missing_features:
        raise ValueError(
            "Missing baseline modelling features: " + ", ".join(missing_features)
        )

    if target not in df.columns:
        raise ValueError(f"Target column not found in modelling dataset: {target}")

    X = df.loc[:, MODEL_FEATURES].copy()
    y = df[target].copy()

    return X, y


def build_preprocessor(config: dict) -> ColumnTransformer:

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    missing_values=pd.NA,
                    strategy="constant",
                    fill_value="Unknown",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                config["parameters"]["modelling"]["features"]["numerical_features"],
            ),
            (
                "categorical",
                categorical_pipeline,
                config["parameters"]["modelling"]["features"]["categorical_features"],
            ),
            (
                "engineered",
                "passthrough",
                config["parameters"]["modelling"]["features"]["engineered_features"],
            ),
        ],
        remainder="drop",
    )

    return preprocessor
