import pandas as pd

from credit_risk.target.delinquency import (
    add_serious_delinquency_flag,
    build_cohort_eligibility,
    build_outcome_observability,
    build_24m_serious_delinquency_target,
)


def test_numeric_serious_delinquency():
    df = pd.DataFrame(
        {
            "current_loan_delinquency_status": [
                "00",
                "01",
                "02",
                "03",
                "04",
                "12",
            ]
        }
    )

    result = add_serious_delinquency_flag(df)

    assert result["is_serious_delinquency"].tolist() == [
        False,
        False,
        False,
        True,
        True,
        True,
    ]


def test_reo_acquisition_is_serious_delinquency():
    df = pd.DataFrame(
        {
            "current_loan_delinquency_status": [
                "00",
                "RA",
            ]
        }
    )

    result = add_serious_delinquency_flag(df)

    assert result["is_serious_delinquency"].tolist() == [
        False,
        True,
    ]


def test_unknown_status_is_not_serious_delinquency():
    df = pd.DataFrame(
        {
            "current_loan_delinquency_status": [
                "XX",
                None,
            ]
        }
    )

    result = add_serious_delinquency_flag(df)

    assert result["is_serious_delinquency"].tolist() == [
        False,
        False,
    ]


def test_cohort_eligibility_by_first_loan_age():
    df = pd.DataFrame(
        {
            "loan_id": [
                "A",
                "A",
                "A",
                "B",
                "B",
                "C",
                "C",
                "D",
                "D",
            ],
            "loan_age": [
                0,
                1,
                2,
                1,
                2,
                2,
                3,
                25,
                26,
            ],
        }
    )

    result = build_cohort_eligibility(df)

    result = result.set_index("loan_id")

    assert result.loc["A", "first_loan_age"] == 0
    assert result.loc["A", "is_start_eligible"]

    assert result.loc["B", "first_loan_age"] == 1
    assert result.loc["B", "is_start_eligible"]

    assert result.loc["C", "first_loan_age"] == 2
    assert not result.loc["C", "is_start_eligible"]

    assert result.loc["D", "first_loan_age"] == 25
    assert not result.loc["D", "is_start_eligible"]


def test_start_age_one_is_eligible_but_two_is_not():
    df = pd.DataFrame(
        {
            "loan_id": ["A", "B"],
            "loan_age": [1, 2],
        }
    )

    result = build_cohort_eligibility(df).set_index("loan_id")

    assert result.loc["A", "is_start_eligible"]
    assert not result.loc["B", "is_start_eligible"]


def test_completed_horizon_is_observable():
    df = pd.DataFrame(
        {
            "loan_id": ["A", "A", "A"],
            "loan_age": [0, 12, 24],
            "current_loan_delinquency_status": ["00", "00", "00"],
            "zero_balance_code": [None, None, None],
        }
    )

    result = build_outcome_observability(df).set_index("loan_id")

    assert result.loc["A", "is_outcome_observable"]


def test_voluntary_early_payoff_is_observable():
    df = pd.DataFrame(
        {
            "loan_id": ["A", "A"],
            "loan_age": [0, 15],
            "current_loan_delinquency_status": ["00", "00"],
            "zero_balance_code": [None, "01"],
        }
    )

    result = build_outcome_observability(df).set_index("loan_id")

    assert result.loc["A", "voluntary_early_payoff"]
    assert result.loc["A", "is_outcome_observable"]


def test_special_early_termination_is_not_observable():
    df = pd.DataFrame(
        {
            "loan_id": ["A", "A"],
            "loan_age": [0, 15],
            "current_loan_delinquency_status": ["00", "00"],
            "zero_balance_code": [None, "96"],
        }
    )

    result = build_outcome_observability(df).set_index("loan_id")

    assert not result.loc["A", "is_outcome_observable"]


def test_serious_delinquency_makes_early_outcome_observable():
    df = pd.DataFrame(
        {
            "loan_id": ["A", "A", "A"],
            "loan_age": [0, 10, 15],
            "current_loan_delinquency_status": ["00", "03", "06"],
            "zero_balance_code": [None, None, "96"],
        }
    )

    result = build_outcome_observability(df).set_index("loan_id")

    assert result.loc["A", "ever_serious_delinquency"]
    assert result.loc["A", "is_outcome_observable"]


def test_final_target_construction():
    df = pd.DataFrame(
        {
            "loan_id": [
                # A: healthy through 24m
                "A",
                "A",
                "A",
                # B: serious delinquency
                "B",
                "B",
                "B",
                # C: voluntary payoff
                "C",
                "C",
                # D: unexplained/special early exit
                "D",
                "D",
                # E: starts too late
                "E",
                "E",
            ],
            "loan_age": [
                0,
                12,
                24,
                0,
                10,
                20,
                0,
                15,
                0,
                15,
                2,
                24,
            ],
            "current_loan_delinquency_status": [
                "00",
                "00",
                "00",
                "00",
                "03",
                "04",
                "00",
                "00",
                "00",
                "00",
                "00",
                "00",
            ],
            "zero_balance_code": [
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                "01",
                None,
                "96",
                None,
                None,
            ],
        }
    )

    result = build_24m_serious_delinquency_target(df).set_index("loan_id")

    # A: observed through month 24, no event
    assert result.loc["A", "ever_90dpd_24m"] == 0

    # B: observed serious delinquency
    assert result.loc["B", "ever_90dpd_24m"] == 1

    # C: voluntary payoff before month 24
    assert result.loc["C", "ever_90dpd_24m"] == 0

    # D: special early termination -> excluded
    assert "D" not in result.index

    # E: first observed at age 2 -> excluded
    assert "E" not in result.index


def test_missing_zero_balance_code_does_not_break_observability():
    df = pd.DataFrame(
        {
            "loan_id": ["A", "A", "A"],
            "loan_age": [0, 12, 24],
            "current_loan_delinquency_status": ["00", "00", "00"],
            "zero_balance_code": pd.Series(
                [pd.NA, pd.NA, pd.NA],
                dtype="string",
            ),
        }
    )

    result = build_outcome_observability(df).set_index("loan_id")

    assert result.loc["A", "completed_horizon"]
    assert not result.loc["A", "voluntary_early_payoff"]
    assert result.loc["A", "is_outcome_observable"]


def test_missing_zbc_early_exit_is_not_observable():
    df = pd.DataFrame(
        {
            "loan_id": ["A", "A"],
            "loan_age": [0, 15],
            "current_loan_delinquency_status": ["00", "00"],
            "zero_balance_code": pd.Series(
                [pd.NA, pd.NA],
                dtype="string",
            ),
        }
    )

    result = build_outcome_observability(df).set_index("loan_id")

    assert not result.loc["A", "voluntary_early_payoff"]
    assert not result.loc["A", "is_outcome_observable"]
