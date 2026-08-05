import pandas as pd

from credit_risk.data.validation import (
    ValidationResult,
    validate_unique_loan_id,
    validate_unique_loan_period,
)


def test_duplicate_origination_loan_is_invalid():

    df = pd.DataFrame(
        {
            "loan_id": [
                "LOAN1",
                "LOAN1",
            ]
        }
    )

    result = ValidationResult()

    validate_unique_loan_id(df, result)

    assert not result.is_valid
    assert len(result.errors) == 1


def test_unique_origination_loans_are_valid():

    df = pd.DataFrame(
        {
            "loan_id": [
                "LOAN1",
                "LOAN2",
            ]
        }
    )

    result = ValidationResult()

    validate_unique_loan_id(df, result)

    assert result.is_valid


def test_duplicate_loan_period_is_invalid():

    df = pd.DataFrame(
        {
            "loan_id": ["LOAN1", "LOAN1"],
            "period": [201501, 201501],
        }
    )

    result = ValidationResult()

    validate_unique_loan_period(df, result)

    assert not result.is_valid


def test_same_loan_different_period_is_valid():

    df = pd.DataFrame(
        {
            "loan_id": ["LOAN1", "LOAN1"],
            "period": [201501, 201502],
        }
    )

    result = ValidationResult()

    validate_unique_loan_period(df, result)

    assert result.is_valid
