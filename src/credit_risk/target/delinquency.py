from __future__ import annotations
import pandas as pd


def add_serious_delinquency_flag(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    result = df.copy()
    delq_numeric = pd.to_numeric(
        result["current_loan_delinquency_status"], errors="coerce"
    )
    result["is_serious_delinquency"] = delq_numeric.ge(
        config["parameters"]["target"]["serious_delinquency_threshold"]
    ) | result["current_loan_delinquency_status"].eq("RA")
    return result


def build_cohort_eligibility(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    eligibility = df.groupby("loan_id", as_index=False).agg(
        first_loan_age=("loan_age", "min")
    )
    eligibility["is_start_eligible"] = (
        eligibility["first_loan_age"]
        <= config["parameters"]["target"]["max_eligible_start_age"]
    )
    return eligibility


def build_outcome_observability(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = add_serious_delinquency_flag(df, config)
    within_horizon = df.loc[
        df["loan_age"].between(0, config["parameters"]["target"]["horizon_months"])
    ].copy()
    summary = (
        within_horizon.sort_values(["loan_id", "loan_age"])
        .groupby("loan_id")
        .agg(
            last_loan_age=("loan_age", "max"),
            ever_serious_delinquency=("is_serious_delinquency", "max"),
            final_zero_balance_code=("zero_balance_code", "last"),
        )
    ).reset_index()
    summary["completed_horizon"] = (
        summary["last_loan_age"] >= config["parameters"]["target"]["horizon_months"]
    )

    summary["voluntary_early_payoff"] = summary["last_loan_age"].lt(
        config["parameters"]["target"]["horizon_months"]
    ) & summary["final_zero_balance_code"].eq(
        config["parameters"]["target"]["voluntary_payoffs_zbc"]
    ).fillna(
        False
    )

    summary["is_outcome_observable"] = (
        summary["ever_serious_delinquency"].fillna(False)
        | summary["completed_horizon"].fillna(False)
        | summary["voluntary_early_payoff"].fillna(False)
    )

    return summary


def build_24m_serious_delinquency_target(
    df: pd.DataFrame, config: dict
) -> pd.DataFrame:
    eligibility = build_cohort_eligibility(df, config)
    observability = build_outcome_observability(df, config)
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
