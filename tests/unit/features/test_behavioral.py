"""Tests for pandas behavioral feature construction."""

import pandas as pd
import pytest

from credit_risk.features.behavioral import (
    add_behavioral_challenger_features,
    add_behavioral_history_features,
    add_calculated_loan_age,
    add_loan_trajectory_features,
    add_prior_serious_delinquency_flag,
    build_behavioral_features,
    build_behavioral_risk_set,
)


@pytest.fixture
def config() -> dict:
    return {
        "parameters": {
            "behavioral": {"observation_ages": [3], "lookback_windows_months": [2]},
            "target": {"serious_delinquency_threshold": 90},
        }
    }


@pytest.fixture
def master() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loan_id": ["loan-1", "loan-1", "loan-1", "loan-2", "loan-2"],
            "period": pd.PeriodIndex(
                ["2020-01", "2020-02", "2020-03", "2020-01", "2020-03"],
                freq="M",
            ),
            "ddlpi": pd.PeriodIndex(
                ["2019-12", "2020-01", "2020-01", "2019-12", "2020-01"],
                freq="M",
            ),
            "calculated_loan_age": [1, 2, 3, 1, 3],
            "current_loan_delinquency_status": ["0", "30", "60", "0", "RA"],
            "zero_balance_code": [pd.NA] * 5,
            "original_upb": [100_000.0] * 5,
            "current_actual_upb": [90_000.0, 80_000.0, 70_000.0, 95_000.0, 85_000.0],
            "original_interest_rate": [4.0] * 5,
            "current_interest_rate": [4.0, 4.0, 4.5, 4.0, 4.0],
            "current_non_interest_bearing_upb": [0.0, 8_000.0, 7_000.0, 0.0, 0.0],
            "current_interest_bearing_upb": [90_000.0, 72_000.0, 63_000.0, 95_000.0, 85_000.0],
            "modification_flag": ["N", "Y", "N", "N", "N"],
            "payment_deferral_flag": ["N", "N", "Y", "N", "N"],
            "borrower_assistance_plan": ["N", "Y", "N", "N", "N"],
            "delinquency_due_to_disaster": ["N", "N", "Y", "N", "N"],
            "interest_rate_step_indicator": ["N", "Y", "Y", "N", "N"],
        }
    )


def test_add_prior_serious_delinquency_flag_handles_numeric_ra_and_invalid_values(config: dict) -> None:
    source = pd.DataFrame({"current_loan_delinquency_status": ["0", "90", "RA", "bad"]})

    result = add_prior_serious_delinquency_flag(source, config)

    assert result["is_serious_delinquency"].tolist() == [False, True, True, False]
    assert "is_serious_delinquency" not in source


def test_add_calculated_loan_age_uses_month_difference() -> None:
    source = pd.DataFrame(
        {
            "period": pd.PeriodIndex(["2020-01", "2021-02"], freq="M"),
            "first_payment_date": pd.PeriodIndex(["2020-01", "2020-02"], freq="M"),
        }
    )

    result = add_calculated_loan_age(source)

    assert result["calculated_loan_age"].tolist() == [1, 13]
    assert "calculated_loan_age" not in source


def test_add_behavioral_history_features_is_sorted_and_leakage_safe() -> None:
    source = pd.DataFrame(
        {
            "loan_id": ["loan-1", "loan-1", "loan-1"],
            "calculated_loan_age": [3, 1, 2],
            "current_loan_delinquency_status": ["60", "0", "30"],
        }
    )

    result = add_behavioral_history_features(source)

    assert result["calculated_loan_age"].tolist() == [1, 2, 3]
    assert result["max_dpd_to_date"].tolist() == [0, 30, 60]
    assert result["ever_30dpd_to_date"].tolist() == [0, 1, 1]
    assert result["ever_60dpd_to_date"].tolist() == [0, 0, 1]
    assert result["delinquency_months_to_date"].tolist() == [0, 1, 2]
    assert pd.isna(result.loc[0, "months_since_last_delinquency"])
    assert result["months_since_last_delinquency"].iloc[1:].tolist() == [0, 0]


def test_add_loan_trajectory_features_handles_zero_original_upb() -> None:
    source = pd.DataFrame(
        {
            "original_upb": [100.0, 0.0],
            "current_actual_upb": [75.0, 10.0],
            "original_interest_rate": [4.0, 3.5],
            "current_interest_rate": [4.5, 3.0],
        }
    )

    result = add_loan_trajectory_features(source)

    assert result["upb_change_from_origination"].tolist() == [-25.0, 10.0]
    assert result.loc[0, "upb_pct_change_from_origination"] == -0.25
    assert pd.isna(result.loc[1, "upb_pct_change_from_origination"])
    assert result["rate_change_from_origination"].tolist() == [0.5, -0.5]


def test_add_behavioral_challenger_features_uses_age_based_lookbacks(master: pd.DataFrame) -> None:
    result = add_behavioral_challenger_features(add_behavioral_history_features(master), [2])
    loan_one = result.loc[result["loan_id"].eq("loan-1")].reset_index(drop=True)

    assert loan_one["current_modification_flag"].tolist() == [0, 1, 0]
    assert loan_one["ever_modified"].tolist() == [0, 1, 1]
    assert loan_one["payment_deferral_count_2m"].tolist() == [0, 0, 1]
    assert loan_one["rate_step_count_2m"].tolist() == [0, 1, 2]
    assert loan_one["max_dpd_2m"].tolist() == [0.0, 30.0, 60.0]
    assert loan_one.loc[2, "non_interest_bearing_upb_pct"] == 0.1
    assert loan_one.loc[2, "interest_bearing_upb_pct"] == 0.9
    assert loan_one["months_since_ddlpi"].tolist() == [1, 1, 2]


def test_build_behavioral_risk_set_excludes_terminated_and_seriously_delinquent_loans(master: pd.DataFrame, config: dict) -> None:
    source = master.copy()
    source.loc[source["loan_id"].eq("loan-1") & source["calculated_loan_age"].eq(3), "zero_balance_code"] = "01"

    result = build_behavioral_risk_set(source, 3, config)

    assert result.empty
    assert "is_serious_delinquency" not in result.columns


def test_build_behavioral_risk_set_rejects_duplicate_observations(master: pd.DataFrame, config: dict) -> None:
    duplicated = pd.concat([master, master.iloc[[2]]], ignore_index=True)

    with pytest.raises(ValueError, match="Multiple performance rows"):
        build_behavioral_risk_set(duplicated, 3, config)


def test_build_behavioral_features_matches_spark_challenger_output(master: pd.DataFrame, config: dict) -> None:
    result = build_behavioral_features(master, config)

    assert result["loan_id"].tolist() == ["loan-1"]
    row = result.iloc[0]
    assert row["observation_age"] == 3
    assert row["ever_modified"] == 1
    assert row["ever_payment_deferred"] == 1
    assert row["ever_borrower_assistance"] == 1
    assert row["ever_disaster_delinquency"] == 1
    assert row["modification_count_2m"] == 1
    assert row["payment_deferral_count_2m"] == 1
    assert row["rate_step_count_2m"] == 2
    assert row["max_dpd_2m"] == 60
    assert row["months_since_ddlpi"] == 2
    assert "current_modification_flag" not in result.columns


@pytest.mark.parametrize(
    ("function", "args", "match"),
    [
        (add_prior_serious_delinquency_flag, (pd.DataFrame(), {}), "serious delinquency"),
        (add_calculated_loan_age, (pd.DataFrame(),), "calculated loan age"),
        (add_behavioral_history_features, (pd.DataFrame(),), "behavioral history"),
        (add_loan_trajectory_features, (pd.DataFrame(),), "loan trajectory"),
    ],
)
def test_behavioral_helpers_validate_required_columns(function, args, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        function(*args)


def test_build_behavioral_features_validates_configuration_and_input(master: pd.DataFrame, config: dict) -> None:
    empty_ages = {"parameters": {"behavioral": {"observation_ages": []}}}
    invalid_windows = {
        "parameters": {
            "behavioral": {"observation_ages": [3], "lookback_windows_months": [0]},
        }
    }

    with pytest.raises(ValueError, match="observation_ages cannot be empty"):
        build_behavioral_features(master, empty_ages)
    with pytest.raises(ValueError, match="positive values"):
        build_behavioral_features(master, invalid_windows)
    with pytest.raises(ValueError, match="behavioral feature construction"):
        build_behavioral_features(master.drop(columns="ddlpi"), config)
