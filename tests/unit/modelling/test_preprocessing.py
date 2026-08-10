import numpy as np
import pandas as pd
import pytest

from credit_risk.modelling.preprocessing import build_preprocessor, split_features_target


@pytest.fixture
def config() -> dict:
    return {
        "parameters": {
            "target": {"name": "ever_90dpd_24m"},
            "modelling": {
                "features": {
                    "numerical_features": ["original_dti"],
                    "categorical_features": ["occupancy_status"],
                    "engineered_features": ["original_dti_missing"],
                }
            },
        }
    }


@pytest.fixture
def modelling_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loan_id": ["L1", "L2", "L3", "L4"],
            "vintage": [2015] * 4,
            "original_dti": [10.0, 20.0, 30.0, np.nan],
            "occupancy_status": ["P", "I", "P", None],
            "original_dti_missing": [0, 0, 0, 1],
            "ever_90dpd_24m": [0, 0, 1, 0],
        }
    )


def test_split_features_target_uses_only_configured_features(modelling_df, config) -> None:
    X, y = split_features_target(modelling_df, config)
    assert X.columns.tolist() == ["original_dti", "occupancy_status", "original_dti_missing"]
    assert "loan_id" not in X
    assert "vintage" not in X
    assert y.tolist() == [0, 0, 1, 0]


def test_split_features_target_rejects_missing_configured_feature(modelling_df, config) -> None:
    with pytest.raises(ValueError, match="original_dti"):
        split_features_target(modelling_df.drop(columns="original_dti"), config)


def test_preprocessor_uses_training_median_and_handles_unknown_categories(modelling_df, config) -> None:
    X_train, _ = split_features_target(modelling_df.iloc[:3], config)
    X_validation, _ = split_features_target(modelling_df.iloc[3:], config)
    X_validation.loc[:, "occupancy_status"] = "S"

    preprocessor = build_preprocessor(config)
    preprocessor.fit(X_train)
    transformed = preprocessor.transform(X_validation)

    imputer = preprocessor.named_transformers_["numerical"].named_steps["imputer"]
    assert imputer.statistics_.tolist() == [20.0]
    assert transformed.shape[0] == 1


def test_preprocessor_outputs_no_missing_values(modelling_df, config) -> None:
    X, _ = split_features_target(modelling_df, config)
    transformed = build_preprocessor(config).fit_transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    assert not pd.isna(transformed).any()
