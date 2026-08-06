import pandas as pd

from credit_risk.features.origination import (
    normalize_sentinel_values,
    add_missing_indicators,
)


def test_numeric_sentinels_are_converted_to_missing():
    df = pd.DataFrame(
        {
            "original_dti": [35, 999],
            "original_ltv": [80, 999],
            "original_cltv": [85, 999],
        }
    )

    result = normalize_sentinel_values(df)
    print(result)
    assert result["original_dti"].isna().tolist() == [False, True]
    assert result["original_ltv"].isna().tolist() == [False, True]
    assert result["original_cltv"].isna().tolist() == [False, True]


def test_first_time_homebuyer_unknown_is_converted_to_missing():
    df = pd.DataFrame(
        {
            "first_time_homebuyer_flag": ["Y", "N", "9"],
        }
    )

    result = normalize_sentinel_values(df)

    assert result["first_time_homebuyer_flag"].isna().tolist() == [
        False,
        False,
        True,
    ]


def test_normalization_does_not_modify_input():
    df = pd.DataFrame(
        {
            "original_dti": [35, 999],
        }
    )

    normalize_sentinel_values(df)

    assert df["original_dti"].tolist() == [35, 999]


def test_dti_missing_indicator():
    df = pd.DataFrame(
        {
            "original_dti": [35, pd.NA, 42],
        }
    )

    result = add_missing_indicators(df)

    assert result["original_dti_missing"].tolist() == [0, 1, 0]


def test_missing_indicator_does_not_modify_input():
    df = pd.DataFrame(
        {
            "original_dti": [35, pd.NA],
        }
    )

    add_missing_indicators(df)

    assert "original_dti_missing" not in df.columns
