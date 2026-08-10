"""Tests for modelling preprocessing utilities."""

import numpy as np
import pandas as pd
import pytest

from credit_risk.features.eligibility_origination import (
    CATEGORICAL_BASELINE_FEATURES,
    ENGINEERED_BASELINE_FEATURES,
    MODEL_FEATURES,
    NUMERICAL_BASELINE_FEATURES,
)
from credit_risk.modelling.preprocessing import (
    build_preprocessor,
    split_features_target,
)


@pytest.fixture
def config() -> dict:
    """Minimal configuration required by modelling preprocessing."""

    return {
        "parameters": {
            "target": {
                "name": "ever_90dpd_24m",
            },
        },
    }


@pytest.fixture
def modelling_df() -> pd.DataFrame:
    """Synthetic modelling population containing all approved features."""

    data = {}

    for feature in NUMERICAL_BASELINE_FEATURES:
        data[feature] = [10.0, 20.0, 30.0, 40.0]

    for feature in CATEGORICAL_BASELINE_FEATURES:
        data[feature] = ["A", "B", "A", "B"]

    for feature in ENGINEERED_BASELINE_FEATURES:
        data[feature] = [0, 1, 0, 1]

    data["loan_id"] = ["L1", "L2", "L3", "L4"]
    data["vintage"] = [2015, 2015, 2015, 2015]
    data["ever_90dpd_24m"] = [0, 0, 1, 0]

    return pd.DataFrame(data)


def test_split_features_target(
    modelling_df: pd.DataFrame,
    config: dict,
) -> None:
    """Only approved modelling features should enter X."""

    X, y = split_features_target(
        modelling_df,
        config,
    )

    assert X.columns.tolist() == MODEL_FEATURES
    assert y.tolist() == [0, 0, 1, 0]


def test_metadata_does_not_enter_predictors(
    modelling_df: pd.DataFrame,
    config: dict,
) -> None:
    """Identifiers, metadata, and target must not enter X."""

    X, _ = split_features_target(
        modelling_df,
        config,
    )

    assert "loan_id" not in X.columns
    assert "vintage" not in X.columns
    assert "ever_90dpd_24m" not in X.columns


def test_unexpected_column_does_not_enter_predictors(
    modelling_df: pd.DataFrame,
    config: dict,
) -> None:
    """Unexpected columns must not silently become predictors."""

    modelling_df["future_information"] = [
        100,
        200,
        300,
        400,
    ]

    X, _ = split_features_target(
        modelling_df,
        config,
    )

    assert "future_information" not in X.columns
    assert X.columns.tolist() == MODEL_FEATURES


def test_missing_model_feature_raises(
    modelling_df: pd.DataFrame,
    config: dict,
) -> None:
    """All approved modelling features must be present."""

    modelling_df = modelling_df.drop(
        columns=MODEL_FEATURES[0],
    )

    with pytest.raises(ValueError):
        split_features_target(
            modelling_df,
            config,
        )


def test_missing_target_raises(
    modelling_df: pd.DataFrame,
    config: dict,
) -> None:
    """Configured target must be present."""

    modelling_df = modelling_df.drop(
        columns="ever_90dpd_24m",
    )

    with pytest.raises(
        ValueError,
        match="Target column not found",
    ):
        split_features_target(
            modelling_df,
            config,
        )


def test_model_feature_classification_is_complete() -> None:
    """Every model feature must have a preprocessing treatment."""

    classified = (
        NUMERICAL_BASELINE_FEATURES
        + CATEGORICAL_BASELINE_FEATURES
        + ENGINEERED_BASELINE_FEATURES
    )

    assert set(classified) == set(MODEL_FEATURES)
    assert len(classified) == len(MODEL_FEATURES)
    assert len(classified) == len(set(classified))


def test_numerical_missing_values_use_training_median(
    modelling_df: pd.DataFrame,
) -> None:
    """Numerical missing values should use the training median."""

    X_train = modelling_df[MODEL_FEATURES].copy()

    X_train["original_dti"] = [
        10.0,
        20.0,
        30.0,
        np.nan,
    ]

    preprocessor = build_preprocessor()

    preprocessor.fit(X_train)

    numerical_imputer = preprocessor.named_transformers_["numerical"].named_steps[
        "imputer"
    ]

    dti_position = NUMERICAL_BASELINE_FEATURES.index("original_dti")

    dti_median = numerical_imputer.statistics_[dti_position]

    assert dti_median == 20.0


def test_validation_data_does_not_influence_imputation() -> None:
    """Validation data must not influence fitted imputation statistics."""

    train_data = {}
    validation_data = {}

    for feature in NUMERICAL_BASELINE_FEATURES:
        train_data[feature] = [
            10.0,
            20.0,
            30.0,
        ]

        validation_data[feature] = [
            10000.0,
            np.nan,
        ]

    for feature in CATEGORICAL_BASELINE_FEATURES:
        train_data[feature] = [
            "A",
            "A",
            "B",
        ]

        validation_data[feature] = [
            "A",
            "B",
        ]

    for feature in ENGINEERED_BASELINE_FEATURES:
        train_data[feature] = [
            0,
            0,
            0,
        ]

        validation_data[feature] = [
            0,
            1,
        ]

    X_train = pd.DataFrame(train_data)
    X_validation = pd.DataFrame(validation_data)

    preprocessor = build_preprocessor()

    preprocessor.fit(X_train)

    numerical_imputer = preprocessor.named_transformers_["numerical"].named_steps[
        "imputer"
    ]

    dti_position = NUMERICAL_BASELINE_FEATURES.index("original_dti")

    # Training median:
    # median(10, 20, 30) = 20.
    assert numerical_imputer.statistics_[dti_position] == 20.0

    transformed_validation = preprocessor.transform(X_validation)

    assert transformed_validation.shape[0] == 2

    # Validation transformation must not modify fitted statistics.
    assert numerical_imputer.statistics_[dti_position] == 20.0


def test_categorical_missing_values_are_handled(
    modelling_df: pd.DataFrame,
) -> None:
    """Missing categorical values should be handled successfully."""

    X_train = modelling_df[MODEL_FEATURES].copy()

    categorical_feature = CATEGORICAL_BASELINE_FEATURES[0]

    X_train.loc[
        0,
        categorical_feature,
    ] = np.nan

    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(X_train)

    assert transformed.shape[0] == len(X_train)


def test_categorical_missing_value_becomes_unknown(
    modelling_df: pd.DataFrame,
) -> None:
    """Missing categorical values should become an Unknown category."""

    X_train = modelling_df[MODEL_FEATURES].copy()

    categorical_feature = CATEGORICAL_BASELINE_FEATURES[0]

    X_train.loc[
        0,
        categorical_feature,
    ] = np.nan

    preprocessor = build_preprocessor()

    preprocessor.fit(X_train)

    categorical_transformer = preprocessor.named_transformers_["categorical"]

    imputer = categorical_transformer.named_steps["imputer"]

    imputed = imputer.transform(X_train[CATEGORICAL_BASELINE_FEATURES])

    feature_position = CATEGORICAL_BASELINE_FEATURES.index(categorical_feature)

    assert imputed[0, feature_position] == "Unknown"


def test_unseen_validation_category_does_not_fail(
    modelling_df: pd.DataFrame,
) -> None:
    """Categories unseen during training should transform safely."""

    X_train = modelling_df[MODEL_FEATURES].copy()

    X_validation = modelling_df[MODEL_FEATURES].iloc[:1].copy()

    categorical_feature = CATEGORICAL_BASELINE_FEATURES[0]

    X_validation[categorical_feature] = "UNSEEN_CATEGORY"

    preprocessor = build_preprocessor()

    preprocessor.fit(X_train)

    transformed = preprocessor.transform(X_validation)

    assert transformed.shape[0] == 1


def test_engineered_dti_indicator_is_preserved(
    modelling_df: pd.DataFrame,
) -> None:
    """DTI missingness indicator should pass through unchanged."""

    X_train = modelling_df[MODEL_FEATURES].copy()

    X_train["original_dti_missing"] = [
        0,
        1,
        0,
        1,
    ]

    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(X_train)

    feature_names = list(preprocessor.get_feature_names_out())

    indicator_position = feature_names.index("engineered__original_dti_missing")

    indicator_values = transformed[
        :,
        indicator_position,
    ]

    if hasattr(indicator_values, "toarray"):
        indicator_values = indicator_values.toarray().ravel()
    else:
        indicator_values = np.asarray(indicator_values).ravel()

    assert indicator_values.tolist() == [
        0,
        1,
        0,
        1,
    ]


def test_transformed_data_contains_no_missing_values(
    modelling_df: pd.DataFrame,
) -> None:
    """Handled missing values should not survive transformation."""

    X_train = modelling_df[MODEL_FEATURES].copy()

    X_train.loc[
        0,
        "original_dti",
    ] = np.nan

    categorical_feature = CATEGORICAL_BASELINE_FEATURES[0]

    X_train.loc[
        1,
        categorical_feature,
    ] = np.nan

    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(X_train)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    assert not pd.isna(transformed).any()


def test_unexpected_features_are_dropped(
    modelling_df: pd.DataFrame,
) -> None:
    """Columns outside the preprocessing contract must be dropped."""

    X_train = modelling_df[MODEL_FEATURES].copy()

    X_train["future_performance_information"] = [
        100,
        200,
        300,
        400,
    ]

    preprocessor = build_preprocessor()

    preprocessor.fit(X_train)

    feature_names = preprocessor.get_feature_names_out()

    assert not any(
        "future_performance_information" in feature for feature in feature_names
    )
