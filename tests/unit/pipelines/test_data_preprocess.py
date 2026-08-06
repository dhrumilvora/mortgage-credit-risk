import pandas as pd

from credit_risk.features.eligibility import (
    BASELINE_FEATURES,
    IDENTIFIER_FEATURE,
)

from credit_risk.pipelines.data_preprocess import build_origination


def test_build_origination():
    """Test the complete origination preprocessing flow."""

    required_fields = IDENTIFIER_FEATURE + BASELINE_FEATURES

    data = {column: [1, 1] for column in required_fields}

    # Identifiers
    data["loan_id"] = ["loan_1", "loan_2"]

    # Sentinel + normal value
    data["original_dti"] = [999, 35]

    df = pd.DataFrame(data)

    result = build_origination(df)

    # Identifier survives preprocessing
    assert "loan_id" in result.columns
    assert result["loan_id"].tolist() == [
        "loan_1",
        "loan_2",
    ]

    # Sentinel converted to missing
    assert pd.isna(result.loc[0, "original_dti"])
    assert result.loc[1, "original_dti"] == 35

    # Missingness indicator
    assert result["original_dti_missing"].tolist() == [
        1,
        0,
    ]

    # Final schema
    expected_columns = IDENTIFIER_FEATURE + BASELINE_FEATURES + ["original_dti_missing"]

    assert result.columns.tolist() == expected_columns
