import pytest

from credit_risk.features.eligibility_origination import (
    BASELINE_FEATURES,
    IDENTIFIER_FEATURE,
    validate_baseline_features,
)


def test_baseline_features_are_valid():
    """Validation should pass when all required fields exist."""

    columns = IDENTIFIER_FEATURE + BASELINE_FEATURES

    validate_baseline_features(columns)


def test_missing_baseline_feature_raises_error():
    """Validation should fail when a baseline feature is missing."""

    columns = IDENTIFIER_FEATURE + BASELINE_FEATURES
    columns = [column for column in columns if column != "original_dti"]

    with pytest.raises(
        ValueError,
        match="Missing required baseline features: original_dti",
    ):
        validate_baseline_features(columns)


def test_missing_identifier_raises_error():
    """Validation should fail when loan_id is missing."""

    columns = BASELINE_FEATURES.copy()

    with pytest.raises(
        ValueError,
        match="Missing required baseline features: loan_id",
    ):
        validate_baseline_features(columns)
