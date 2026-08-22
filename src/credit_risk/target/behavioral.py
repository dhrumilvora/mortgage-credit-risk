"""Behavioral target construction for mortgage credit-risk modelling."""

from __future__ import annotations

import pandas as pd


def add_serious_delinquency_flag(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Add a row-level serious-delinquency indicator.

    A serious delinquency is defined as either:
    - numeric delinquency status >= configured threshold; or
    - REO acquisition status ("RA").
    """
    result = df.copy()

    delinquency_numeric = pd.to_numeric(
        result["current_loan_delinquency_status"],
        errors="coerce",
    )

    result["is_serious_delinquency"] = delinquency_numeric.ge(
        config["parameters"]["target"]["serious_delinquency_threshold"]
    ) | result["current_loan_delinquency_status"].eq("RA")

    return result


def _build_observation_population(
    df: pd.DataFrame,
    observation_age: int,
) -> pd.DataFrame:
    """
    Build the population observable at a specific loan age.

    Output grain:
        one row per loan
    """
    return (
        df.loc[
            df["calculated_loan_age"].eq(observation_age),
            [
                "loan_id",
                "calculated_loan_age",
            ],
        ]
        .drop_duplicates(
            subset=["loan_id"],
        )
        .copy()
    )


def _build_future_window(
    df: pd.DataFrame,
    observation_age: int,
    prediction_horizon: int,
) -> pd.DataFrame:
    """
    Select performance observations in the forward prediction window.

    Window:
        observation_age + 1
        through
        observation_age + prediction_horizon
    """
    future_start = observation_age + 1
    future_end = observation_age + prediction_horizon

    return df.loc[
        df["calculated_loan_age"].between(
            future_start,
            future_end,
        )
    ].copy()


def _summarize_future_outcome(
    future_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize future performance to one row per loan.

    The summary contains:
    - last observed loan age in the future window;
    - whether serious delinquency ever occurred;
    - final zero-balance code observed in the future window.
    """
    if future_df.empty:
        return pd.DataFrame(
            columns=[
                "loan_id",
                "last_future_loan_age",
                "ever_serious_delinquency",
                "final_zero_balance_code",
            ]
        )

    return (
        future_df.sort_values(
            [
                "loan_id",
                "calculated_loan_age",
            ],
        )
        .groupby(
            "loan_id",
            as_index=False,
        )
        .agg(
            last_future_loan_age=(
                "calculated_loan_age",
                "max",
            ),
            ever_serious_delinquency=(
                "is_serious_delinquency",
                "max",
            ),
            final_zero_balance_code=(
                "zero_balance_code",
                "last",
            ),
        )
    )


def _add_observability_flags(
    future_summary: pd.DataFrame,
    observation_age: int,
    prediction_horizon: int,
    voluntary_payoff_zbc,
) -> pd.DataFrame:
    """
    Add future-horizon completion and observability flags.

    A future outcome is considered observable when:
    - the complete prediction horizon is observed;
    - a serious delinquency is observed; or
    - the loan voluntarily pays off before the end of the horizon.
    """
    result = future_summary.copy()

    future_end = observation_age + prediction_horizon

    result["completed_horizon"] = result["last_future_loan_age"] >= future_end

    result["voluntary_early_payoff"] = (
        result["last_future_loan_age"].lt(future_end)
        & result["final_zero_balance_code"].eq(
            voluntary_payoff_zbc,
        )
    ).fillna(False)

    result["is_outcome_observable"] = (
        result["ever_serious_delinquency"].fillna(False)
        | result["completed_horizon"].fillna(False)
        | result["voluntary_early_payoff"].fillna(False)
    )

    return result


def _build_observation_target(
    df: pd.DataFrame,
    observation_age: int,
    prediction_horizon: int,
    voluntary_payoff_zbc,
) -> pd.DataFrame:
    """
    Build the behavioral target for one observation age.

    Output grain:
        loan_id x observation_age
    """
    observation_df = _build_observation_population(
        df=df,
        observation_age=observation_age,
    )

    if observation_df.empty:
        return pd.DataFrame(
            columns=[
                "loan_id",
                "observation_age",
                "future_90dpd_12m",
            ]
        )

    future_df = _build_future_window(
        df=df,
        observation_age=observation_age,
        prediction_horizon=prediction_horizon,
    )

    if future_df.empty:
        return pd.DataFrame(
            columns=[
                "loan_id",
                "observation_age",
                "future_90dpd_12m",
            ]
        )

    future_summary = _summarize_future_outcome(
        future_df=future_df,
    )

    future_summary = _add_observability_flags(
        future_summary=future_summary,
        observation_age=observation_age,
        prediction_horizon=prediction_horizon,
        voluntary_payoff_zbc=voluntary_payoff_zbc,
    )

    result = observation_df.merge(
        future_summary,
        on="loan_id",
        how="left",
        validate="one_to_one",
    )

    result["is_target_eligible"] = result["is_outcome_observable"].fillna(False)

    result["future_90dpd_12m"] = (
        result["ever_serious_delinquency"].fillna(False).astype("int8")
    )

    result["observation_age"] = observation_age

    return result.loc[
        result["is_target_eligible"],
        [
            "loan_id",
            "observation_age",
            "future_90dpd_12m",
        ],
    ].reset_index(drop=True)


def build_behavioral_target(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Build point-in-time behavioral targets.

    Each eligible row represents a loan at a configured observation age.
    The target indicates whether serious delinquency occurs during the
    subsequent prediction horizon.

    Current target definition:

        future_90dpd_12m = 1

    when serious delinquency occurs between:

        observation_age + 1

    and:

        observation_age + prediction_horizon

    Output grain:
        loan_id x observation_age
    """
    behavioral_config = config["parameters"]["behavioral"]

    observation_ages = behavioral_config["observation_ages"]

    prediction_horizon = behavioral_config["prediction_horizon_months"]

    voluntary_payoff_zbc = config["parameters"]["target"]["voluntary_payoffs_zbc"]

    df = add_serious_delinquency_flag(
        df=df,
        config=config,
    )

    results = []

    for observation_age in observation_ages:
        observation_target = _build_observation_target(
            df=df,
            observation_age=observation_age,
            prediction_horizon=prediction_horizon,
            voluntary_payoff_zbc=voluntary_payoff_zbc,
        )

        if observation_target.empty:
            continue

        results.append(observation_target)

    if not results:
        return pd.DataFrame(
            columns=[
                "loan_id",
                "observation_age",
                "future_90dpd_12m",
            ]
        )

    return pd.concat(
        results,
        ignore_index=True,
    ).reset_index(drop=True)
