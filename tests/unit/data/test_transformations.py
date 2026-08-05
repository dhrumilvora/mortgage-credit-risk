import pandas as pd

from credit_risk.data.transformers import (
    to_month_period,
    to_nullable_integer,
)


def test_to_month_period():

    series = pd.Series(["201503", "201504", None])

    result = to_month_period(series)

    assert result.iloc[0] == pd.Period("2015-03", freq="M")
    assert result.iloc[1] == pd.Period("2015-04", freq="M")
    assert pd.isna(result.iloc[2])


def test_to_nullable_integer():

    series = pd.Series(["100", "200", None])

    result = to_nullable_integer(series)

    assert str(result.dtype) == "Int64"
    assert result.iloc[0] == 100
    assert pd.isna(result.iloc[2])


from credit_risk.data.transformers import transform_performance


def test_delinquency_status_remains_string():

    df = pd.DataFrame(
        {
            "current_loan_delinquency_status": [
                "00",
                "01",
                "03",
                "RA",
                "XX",
            ]
        }
    )


from credit_risk.data.transformers import to_string


def test_encoded_values_are_preserved():

    series = pd.Series(["00", "01", "03", "RA", "XX"])

    result = to_string(series)

    assert result.tolist() == [
        "00",
        "01",
        "03",
        "RA",
        "XX",
    ]
