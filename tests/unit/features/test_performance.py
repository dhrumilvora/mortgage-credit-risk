import pandas as pd
import pytest

from credit_risk.features.performance import (
    build_performance,
    validate_performance_columns,
    validate_performance_grain,
)


def make_performance_df():
    return pd.DataFrame(
        {
            "loan_id": ["B", "A", "A"],
            "monthly_reporting_period": [
                "201502",
                "201502",
                "201501",
            ],
            "loan_age": [1, 1, 0],
            "current_loan_delinquency_status": [
                "00",
                "01",
                "00",
            ],
            "zero_balance_code": [
                pd.NA,
                pd.NA,
                pd.NA,
            ],
        }
    )


def test_build_performance_preserves_rows():
    perf = make_performance_df()

    result = build_performance(perf)

    assert len(result) == len(perf)


def test_build_performance_normalizes_types():
    perf = make_performance_df()

    result = build_performance(perf)

    assert str(result["loan_id"].dtype) == "string"
    assert str(result["loan_age"].dtype) == "Int64"
    assert pd.api.types.is_datetime64_any_dtype(result["monthly_reporting_period"])


def test_build_performance_sorts_history():
    perf = make_performance_df()

    result = build_performance(perf)

    loan_a = result.loc[result["loan_id"] == "A"]

    assert loan_a["loan_age"].tolist() == [0, 1]


def test_missing_required_column_raises():
    perf = make_performance_df().drop(columns="loan_age")

    with pytest.raises(
        ValueError,
        match="loan_age",
    ):
        validate_performance_columns(perf)


def test_duplicate_loan_month_raises():
    perf = make_performance_df()

    duplicate = perf.iloc[[0]].copy()

    perf = pd.concat(
        [perf, duplicate],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        build_performance(perf)


def test_negative_loan_age_raises():
    perf = make_performance_df()

    perf.loc[0, "loan_age"] = -1

    with pytest.raises(
        ValueError,
        match="negative loan_age",
    ):
        build_performance(perf)


def test_build_performance_does_not_modify_input():
    perf = make_performance_df()

    original = perf.copy(deep=True)

    build_performance(perf)

    pd.testing.assert_frame_equal(
        perf,
        original,
    )
