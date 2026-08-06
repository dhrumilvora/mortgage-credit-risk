from __future__ import annotations
import pandas as pd

SERIOUS_DELINQUENCY_THRESHOLD = 3
MAX_ELIGIBLE_START_AGE = 1
TARGET_HORIZON_MONTHS = 24
VOLUNTARY_PAYOFF_ZBC = "01"


def add_serious_delinquency_flag(perf: pd.DataFrame) -> pd.DataFrame:
    result = perf.copy()
    delq_numeric = pd.to_numeric(
        result["current_loan_delinquency_status"], errors="coerce"
    )
    result["is_serious_delinquency"] = delq_numeric.ge(
        SERIOUS_DELINQUENCY_THRESHOLD
    ) | result["current_loan_delinquency_status"].eq("RA")
    return result


def build_cohort_eligibility(perf: pd.DataFrame) -> pd.DataFrame:
    eligibility = perf.groupby("loan_id", as_index=False).agg(
        first_loan_age=("loan_age", "min")
    )
    eligibility["is_start_eligible"] = (
        eligibility["first_loan_age"] <= MAX_ELIGIBLE_START_AGE
    )
    return eligibility


def build_outcome_observability(perf: pd.DataFrame) -> pd.DataFrame:
    perf = add_serious_delinquency_flag(perf)
    within_horizon = perf.loc[perf["loan_age"].between(0, TARGET_HORIZON_MONTHS)].copy()
    summary = (
        within_horizon.sort_values(["loan_id", "loan_age"])
        .groupby("loan_id")
        .agg(
            last_loan_age=("loan_age", "max"),
            ever_serious_delinquency=("is_serious_delinquency", "max"),
            final_zero_balance_code=("zero_balance_code", "last"),
        )
    ).reset_index()
    summary["completed_horizon"] = summary["last_loan_age"] >= TARGET_HORIZON_MONTHS

    summary["voluntary_early_payoff"] = summary["last_loan_age"].lt(
        TARGET_HORIZON_MONTHS
    ) & summary["final_zero_balance_code"].eq(VOLUNTARY_PAYOFF_ZBC).fillna(False)

    summary["is_outcome_observable"] = (
        summary["ever_serious_delinquency"].fillna(False)
        | summary["completed_horizon"].fillna(False)
        | summary["voluntary_early_payoff"].fillna(False)
    )

    return summary


def build_24m_serious_delinquency_target(perf: pd.DataFrame) -> pd.DataFrame:
    eligibility = build_cohort_eligibility(perf)
    observability = build_outcome_observability(perf)
    target = eligibility.merge(
        observability, on=["loan_id"], how="left", validate="one_to_one"
    )
    target["is_target_eligible"] = target["is_start_eligible"] & target[
        "is_outcome_observable"
    ].fillna(False)

    target["ever_90dpd_24m"] = (
        target["ever_serious_delinquency"].fillna(False).astype("int8")
    )
    return target.loc[
        target["is_target_eligible"],
        [
            "loan_id",
            "ever_90dpd_24m",
        ],
    ].reset_index(drop=True)
