import pytest

from credit_risk.features.eligibility import (
    BASELINE_FEATURES,
    validate_baseline_features,
)


def test_baseline_features_are_valid():
    validate_baseline_features(BASELINE_FEATURES)


def test_missing_baseline_feature_raises():
    columns = [feature for feature in BASELINE_FEATURES if feature != "credit_score"]

    with pytest.raises(
        ValueError,
        match="credit_score",
    ):
        validate_baseline_features(columns)
