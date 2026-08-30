import pytest

from credit_risk.features.eligibility_origination import validate_baseline_features
from credit_risk.features.eligibility_performance import (
    BASELINE_FEATURES as PERFORMANCE_BASELINE_FEATURES,
    CHALLENGER_FEATURES,
    IDENTIFIER_FIELDS,
    STATE_FIELDS,
    TERMINATION_FIELDS,
    TIME_FIELDS,
    validate_features as validate_performance_features,
)


@pytest.fixture
def config() -> dict:
    return {
        "parameters": {
            "data": {
                "id_col": "loan_id",
                "preprocess": {
                    "features": {
                        "numerical_features": ["credit_score", "original_dti"],
                        "categorical_features": ["occupancy_status"],
                    }
                },
            }
        }
    }


def test_validation_accepts_configured_features(config: dict) -> None:
    validate_baseline_features(
        ["loan_id", "credit_score", "original_dti", "occupancy_status"], config
    )


def test_validation_reports_missing_configured_feature(config: dict) -> None:
    with pytest.raises(ValueError, match="original_dti"):
        validate_baseline_features(["loan_id", "credit_score", "occupancy_status"], config)


def test_validation_reports_missing_configured_identifier(config: dict) -> None:
    with pytest.raises(ValueError, match="loan_id"):
        validate_baseline_features(["credit_score", "original_dti", "occupancy_status"], config)


def test_performance_validation_requires_challenger_fields() -> None:
    required_columns = (
        IDENTIFIER_FIELDS
        + TIME_FIELDS
        + PERFORMANCE_BASELINE_FEATURES
        + STATE_FIELDS
        + TERMINATION_FIELDS
        + CHALLENGER_FEATURES
    )

    validate_performance_features(required_columns)

    with pytest.raises(ValueError, match="borrower_assistance_plan"):
        validate_performance_features(
            [column for column in required_columns if column != "borrower_assistance_plan"]
        )
