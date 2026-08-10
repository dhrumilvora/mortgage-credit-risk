import pandas as pd

from credit_risk.pipelines.data_preprocess import build_origination


def test_build_origination_applies_configured_selection_and_sentinels() -> None:
    config = {
        "parameters": {
            "data": {
                "id_col": "loan_id",
                "preprocess": {
                    "features": {
                        "numerical_features": ["original_dti", "original_ltv"],
                        "categorical_features": ["first_time_homebuyer_flag"],
                    }
                },
            }
        }
    }
    source = pd.DataFrame(
        {
            "loan_id": ["L1", "L2"],
            "original_dti": [999, 35],
            "original_ltv": [80, 999],
            "first_time_homebuyer_flag": ["9", "Y"],
            "future_column": [1, 2],
        }
    )

    result = build_origination(source, config)

    assert result.columns.tolist() == [
        "loan_id", "original_dti", "original_ltv", "first_time_homebuyer_flag", "original_dti_missing"
    ]
    assert pd.isna(result.loc[0, "original_dti"])
    assert pd.isna(result.loc[1, "original_ltv"])
    assert pd.isna(result.loc[0, "first_time_homebuyer_flag"])
    assert result["original_dti_missing"].tolist() == [1, 0]
