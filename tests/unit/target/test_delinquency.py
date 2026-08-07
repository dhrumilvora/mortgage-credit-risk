"""Unit tests for serious-delinquency target construction."""

from __future__ import annotations

import pandas as pd

from credit_risk.target.delinquency import (
    add_serious_delinquency_flag,
    build_24m_serious_delinquency_target,
    build_cohort_eligibility,
    build_outcome_observability,
)

TARGET_CONFIG = {
    "name": "ever_90dpd_24m",
    "serious_delinquency_threshold": 3,
    "max_eligible_start_age": 1,
    "horizon_months": 24,
    "voluntary_payoff_zbc": "01",
}


def test_numeric_serious_delinquency():
    perf = pd.DataFrame(
        {
            "current_loan_delinquency_status": [
                "0",
                "1",
                "2",
                "3",
                "4",
            ]
        }
    )

    result = add_serious_delinquency_flag(
        perf,
        TARGET_CONFIG,
    )

    assert result["is_serious_delinquency"].tolist() == [
        False,
        False,
        False,
        True,
        True,
    ]


def test_reo_acquisition_is_serious_delinquency():
    perf = pd.DataFrame(
        {
            "current_loan_delinquency_status": [
                "RA",
            ]
        }
    )

    result = add_serious_delinquency_flag(
        perf,
        TARGET_CONFIG,
    )

    assert bool(result.loc[0, "is_serious_delinquency"])


def test_unknown_status_is_not_serious_delinquency():
    perf = pd.DataFrame(
        {
            "current_loan_delinquency_status": [
                "XX",
            ]
        }
    )

    result = add_serious_delinquency_flag(
        perf,
        TARGET_CONFIG,
    )

    assert not bool(result.loc[0, "is_serious_delinquency"])


def test_cohort_eligibility_by_first_loan_age():
    perf = pd.DataFrame(
        {
            "loan_id": [
                "A",
                "A",
                "B",
                "B",
            ],
            "loan_age": [
                0,
                1,
                2,
                3,
            ],
        }
    )

    result = build_cohort_eligibility(
        perf,
        TARGET_CONFIG,
    ).set_index("loan_id")

    assert bool(result.loc["A", "is_start_eligible"])
    assert not bool(result.loc["B", "is_start_eligible"])


def test_start_age_one_is_eligible_but_two_is_not():
    perf = pd.DataFrame(
        {
            "loan_id": [
                "A",
                "B",
            ],
            "loan_age": [
                1,
                2,
            ],
        }
    )

    result = build_cohort_eligibility(
        perf,
        TARGET_CONFIG,
    ).set_index("loan_id")

    assert bool(result.loc["A", "is_start_eligible"])
    assert not bool(result.loc["B", "is_start_eligible"])


def test_completed_horizon_is_observable():
    perf = pd.DataFrame(
        {
            "loan_id": [
                "A",
                "A",
            ],
            "loan_age": [
                0,
                24,
            ],
            "current_loan_delinquency_status": [
                "0",
                "0",
            ],
            "zero_balance_code": [
                pd.NA,
                pd.NA,
            ],
        }
    )

    result = build_outcome_observability(
        perf,
        TARGET_CONFIG,
    ).set_index("loan_id")

    assert bool(result.loc["A", "completed_horizon"])
    assert bool(result.loc["A", "is_outcome_observable"])


def test_voluntary_early_payoff_is_observable():
    perf = pd.DataFrame(
        {
            "loan_id": [
                "A",
                "A",
            ],
            "loan_age": [
                0,
                12,
            ],
            "current_loan_delinquency_status": [
                "0",
                "0",
            ],
            "zero_balance_code": [
                pd.NA,
                "01",
            ],
        }
    )

    result = build_outcome_observability(
        perf,
        TARGET_CONFIG,
    ).set_index("loan_id")

    assert bool(result.loc["A", "voluntary_early_payoff"])
    assert bool(result.loc["A", "is_outcome_observable"])


def test_special_early_termination_is_not_observable():
    perf = pd.DataFrame(
        {
            "loan_id": [
                "A",
                "A",
            ],
            "loan_age": [
                0,
                12,
            ],
            "current_loan_delinquency_status": [
                "0",
                "0",
            ],
            "zero_balance_code": [
                pd.NA,
                "03",
            ],
        }
    )

    result = build_outcome_observability(
        perf,
        TARGET_CONFIG,
    ).set_index("loan_id")

    assert not bool(result.loc["A", "completed_horizon"])
    assert not bool(result.loc["A", "voluntary_early_payoff"])
    assert not bool(result.loc["A", "is_outcome_observable"])


def test_serious_delinquency_makes_early_outcome_observable():
    perf = pd.DataFrame(
        {
            "loan_id": [
                "A",
                "A",
            ],
            "loan_age": [
                0,
                10,
            ],
            "current_loan_delinquency_status": [
                "0",
                "3",
            ],
            "zero_balance_code": [
                pd.NA,
                pd.NA,
            ],
        }
    )

    result = build_outcome_observability(
        perf,
        TARGET_CONFIG,
    ).set_index("loan_id")

    assert bool(result.loc["A", "ever_serious_delinquency"])
    assert bool(result.loc["A", "is_outcome_observable"])


def test_final_target_construction():
    perf = pd.DataFrame(
        {
            # A: completes horizon, no event -> target 0
            # B: serious delinquency -> target 1
            # C: starts too late -> excluded
            "loan_id": [
                "A",
                "A",
                "B",
                "B",
                "C",
                "C",
            ],
            "loan_age": [
                0,
                24,
                0,
                10,
                2,
                24,
            ],
            "current_loan_delinquency_status": [
                "0",
                "0",
                "0",
                "3",
                "0",
                "0",
            ],
            "zero_balance_code": [
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
            ],
        }
    )

    result = (
        build_24m_serious_delinquency_target(
            perf,
            TARGET_CONFIG,
        )
        .sort_values("loan_id")
        .reset_index(drop=True)
    )

    expected = pd.DataFrame(
        {
            "loan_id": [
                "A",
                "B",
            ],
            "ever_90dpd_24m": pd.Series(
                [0, 1],
                dtype="int8",
            ),
        }
    )

    pd.testing.assert_frame_equal(
        result,
        expected,
    )


def test_missing_zero_balance_code_does_not_break_observability():
    perf = pd.DataFrame(
        {
            "loan_id": [
                "A",
                "A",
            ],
            "loan_age": [
                0,
                24,
            ],
            "current_loan_delinquency_status": [
                "0",
                "0",
            ],
            "zero_balance_code": [
                pd.NA,
                pd.NA,
            ],
        }
    )

    result = build_outcome_observability(
        perf,
        TARGET_CONFIG,
    ).set_index("loan_id")

    assert not bool(result.loc["A", "voluntary_early_payoff"])

    assert bool(result.loc["A", "completed_horizon"])

    assert bool(result.loc["A", "is_outcome_observable"])


def test_missing_zbc_early_exit_is_not_observable():
    perf = pd.DataFrame(
        {
            "loan_id": [
                "A",
                "A",
            ],
            "loan_age": [
                0,
                12,
            ],
            "current_loan_delinquency_status": [
                "0",
                "0",
            ],
            "zero_balance_code": [
                pd.NA,
                pd.NA,
            ],
        }
    )

    result = build_outcome_observability(
        perf,
        TARGET_CONFIG,
    ).set_index("loan_id")

    assert not bool(result.loc["A", "completed_horizon"])

    assert not bool(result.loc["A", "voluntary_early_payoff"])

    assert not bool(result.loc["A", "is_outcome_observable"])
