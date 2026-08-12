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

    delq_numeric = pd.to_numeric(
        result["current_loan_delinquency_status"],
        errors="coerce",
    )

    result["is_serious_delinquency"] = delq_numeric.ge(
        config["parameters"]["target"]["serious_delinquency_threshold"]
    ) | result["current_loan_delinquency_status"].eq("RA")

    return result


def build_behavioral_target(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Build point-in-time behavioral targets.

    Each eligible row represents a loan at a configured observation age.
    The target indicates whether serious delinquency occurs during the
    subsequent prediction horizon.

    Output grain:
        loan_id x observation_age
    """

    behavioral_config = config["parameters"]["behavioral"]

    observation_ages = behavioral_config["observation_ages"]
    prediction_horizon = behavioral_config["prediction_horizon_months"]

    df = add_serious_delinquency_flag(
        df,
        config,
    )

    results = []

    for observation_age in observation_ages:

        # --------------------------------------------------------------
        # Information available at the observation point
        # --------------------------------------------------------------

        observation_df = df.loc[
            df["loan_age"].eq(observation_age),
            [
                "loan_id",
                "loan_age",
            ],
        ].drop_duplicates(
            subset=["loan_id"],
        )

        if observation_df.empty:
            continue

        # --------------------------------------------------------------
        # Future performance window
        #
        # Target window:
        # observation_age + 1
        # through
        # observation_age + prediction_horizon
        # --------------------------------------------------------------

        future_start = observation_age + 1
        future_end = observation_age + prediction_horizon

        future_df = df.loc[
            df["loan_age"].between(
                future_start,
                future_end,
            )
        ].copy()

        if future_df.empty:
            continue

        # --------------------------------------------------------------
        # Summarise future outcome by loan
        # --------------------------------------------------------------

        future_summary = (
            future_df.sort_values(
                ["loan_id", "loan_age"],
            )
            .groupby("loan_id")
            .agg(
                last_future_loan_age=("loan_age", "max"),
                ever_serious_delinquency=(
                    "is_serious_delinquency",
                    "max",
                ),
                final_zero_balance_code=(
                    "zero_balance_code",
                    "last",
                ),
            )
            .reset_index()
        )

        # --------------------------------------------------------------
        # Determine whether the complete future window is observable
        # --------------------------------------------------------------

        future_summary["completed_horizon"] = (
            future_summary["last_future_loan_age"] >= future_end
        )

        # --------------------------------------------------------------
        # Voluntary early payoff
        #
        # A voluntary payoff before the end of the horizon makes the
        # remaining future period unobservable, but the loan has a known
        # non-delinquent termination.
        # --------------------------------------------------------------

        future_summary["voluntary_early_payoff"] = (
            future_summary["last_future_loan_age"].lt(future_end)
            & future_summary["final_zero_balance_code"].eq(
                config["parameters"]["target"]["voluntary_payoffs_zbc"]
            )
        ).fillna(False)

        # --------------------------------------------------------------
        # Outcome observability
        # --------------------------------------------------------------

        future_summary["is_outcome_observable"] = (
            future_summary["ever_serious_delinquency"].fillna(False)
            | future_summary["completed_horizon"].fillna(False)
            | future_summary["voluntary_early_payoff"].fillna(False)
        )

        # --------------------------------------------------------------
        # Construct target
        # --------------------------------------------------------------

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

        results.append(
            result.loc[
                result["is_target_eligible"],
                [
                    "loan_id",
                    "observation_age",
                    "future_90dpd_12m",
                ],
            ]
        )

    if not results:
        return pd.DataFrame(
            columns=[
                "loan_id",
                "observation_age",
                "future_90dpd_12m",
            ]
        )

    target = pd.concat(
        results,
        ignore_index=True,
    )

    return target.reset_index(drop=True)
