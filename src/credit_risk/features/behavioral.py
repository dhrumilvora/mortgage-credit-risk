"""Point-in-time behavioral feature construction for mortgage credit-risk modelling."""

from __future__ import annotations

import pandas as pd


def add_prior_serious_delinquency_flag(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Add a row-level serious-delinquency indicator.

    A serious delinquency is defined as either:
    - numeric delinquency status >= configured threshold; or
    - REO acquisition status ("RA").

    This flag is used only to determine whether a loan has already
    experienced serious delinquency by a behavioral observation point.
    """
    result = df.copy()

    delinquency_numeric = pd.to_numeric(
        result["current_loan_delinquency_status"],
        errors="coerce",
    )

    serious_delinquency_threshold = config["parameters"]["target"][
        "serious_delinquency_threshold"
    ]

    result["is_serious_delinquency"] = delinquency_numeric.ge(
        serious_delinquency_threshold,
    ) | result["current_loan_delinquency_status"].eq("RA")

    return result


def build_behavioral_risk_set(
    master: pd.DataFrame,
    observation_age: int,
    config: dict,
) -> pd.DataFrame:
    """
    Build the eligible point-in-time loan population for one observation age.

    A loan is eligible when:

    - an exact performance observation exists at the observation age;
    - the loan has not experienced serious delinquency at or before
      the observation age;
    - the loan has not already terminated at the observation age.

    Output grain:
        loan_id x observation_age
    """
    if observation_age < 0:
        raise ValueError(
            f"observation_age must be non-negative, got {observation_age}."
        )

    required_columns = {
        "loan_id",
        "calculated_loan_age",
        "current_loan_delinquency_status",
        "zero_balance_code",
    }

    missing_columns = sorted(required_columns - set(master.columns))

    if missing_columns:
        raise ValueError(
            "Missing columns required to build behavioral risk set: "
            + ", ".join(missing_columns)
        )

    working = add_prior_serious_delinquency_flag(
        master,
        config,
    )

    # ------------------------------------------------------------------
    # Determine whether each loan has already experienced serious
    # delinquency at or before the observation age.
    # ------------------------------------------------------------------

    prior_history = working.loc[working["calculated_loan_age"].le(observation_age)]

    prior_serious_delinquency = (
        prior_history.groupby("loan_id")["is_serious_delinquency"].max().astype(bool)
    )

    # ------------------------------------------------------------------
    # Select the exact point-in-time observation.
    # ------------------------------------------------------------------

    observation = working.loc[working["calculated_loan_age"].eq(observation_age)].copy()

    if observation.empty:
        return pd.DataFrame(columns=working.columns.tolist())

    # ------------------------------------------------------------------
    # There must be exactly one performance row per loan at a given
    # observation age.
    # ------------------------------------------------------------------

    duplicate_mask = observation.duplicated(
        subset=["loan_id"],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())

        raise ValueError(
            "Multiple performance rows found for the same loan at "
            f"observation_age={observation_age}; "
            f"duplicate_rows={duplicate_count:,}."
        )

    # ------------------------------------------------------------------
    # Determine whether serious delinquency occurred by the
    # observation point.
    # ------------------------------------------------------------------

    observation["has_prior_serious_delinquency"] = (
        observation["loan_id"].map(prior_serious_delinquency).fillna(False).astype(bool)
    )

    # ------------------------------------------------------------------
    # A populated zero-balance code indicates that the loan has
    # terminated at the observation point.
    # ------------------------------------------------------------------

    observation["is_terminated_at_observation"] = observation[
        "zero_balance_code"
    ].notna()

    # ------------------------------------------------------------------
    # Build the risk set.
    # ------------------------------------------------------------------

    eligible = observation.loc[
        ~observation["has_prior_serious_delinquency"]
        & ~observation["is_terminated_at_observation"]
    ].copy()

    # ------------------------------------------------------------------
    # Explicitly identify the observation point.
    # ------------------------------------------------------------------

    eligible["observation_age"] = observation_age

    # ------------------------------------------------------------------
    # Remove internal helper fields.
    # ------------------------------------------------------------------

    eligible = eligible.drop(
        columns=[
            "is_serious_delinquency",
            "has_prior_serious_delinquency",
            "is_terminated_at_observation",
        ],
        errors="ignore",
    )

    # ------------------------------------------------------------------
    # Defensive grain validation.
    # ------------------------------------------------------------------

    duplicate_mask = eligible.duplicated(
        subset=[
            "loan_id",
            "observation_age",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        raise ValueError(
            "Behavioral risk set violates expected " "loan_id x observation_age grain."
        )

    return eligible.reset_index(
        drop=True,
    )


def build_behavioral_features(
    master: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Build the complete point-in-time behavioral feature population.

    Configured observation ages are extracted from the master loan-month
    dataset and concatenated into one dataset.

    Output grain:
        loan_id x observation_age

    This function does not construct a future target.

    Every returned feature must represent information available at the
    corresponding observation age.
    """
    behavioral_config = config["parameters"]["behavioral"]

    observation_ages = behavioral_config.get(
        "observation_ages",
        [],
    )

    if not observation_ages:
        raise ValueError("parameters.behavioral.observation_ages cannot be empty.")

    master = add_behavioral_history_features(
        master,
    )
    master = add_loan_trajectory_features(
        master,
    )
    results = []

    for observation_age in observation_ages:
        population = build_behavioral_risk_set(
            master=master,
            observation_age=observation_age,
            config=config,
        )

        if population.empty:
            continue

        results.append(population)

    if not results:
        return pd.DataFrame()

    behavioral_features = pd.concat(
        results,
        ignore_index=True,
    )

    # ------------------------------------------------------------------
    # Final grain validation.
    # ------------------------------------------------------------------

    duplicate_mask = behavioral_features.duplicated(
        subset=[
            "loan_id",
            "observation_age",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())

        raise ValueError(
            "Behavioral feature population contains duplicate "
            "loan_id x observation_age rows: "
            f"{duplicate_count:,}."
        )

    return behavioral_features.reset_index(
        drop=True,
    )


def add_calculated_loan_age(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate loan age from period and first payment date.

    Uses the convention:
        age = months(period - first_payment_date) + 1
    """
    result = df.copy()

    period_month = result["period"].astype(str)
    first_payment_month = result["first_payment_date"].astype(str)

    period_year = period_month.str[:4].astype(int)
    period_number = period_month.str[5:7].astype(int)

    first_payment_year = first_payment_month.str[:4].astype(int)
    first_payment_number = first_payment_month.str[5:7].astype(int)

    result["calculated_loan_age"] = (
        (period_year - first_payment_year) * 12
        + (period_number - first_payment_number)
        + 1
    )

    return result


def add_behavioral_history_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add leakage-safe historical behavioral features.

    All features are calculated using information available up to and
    including the current performance month.

    Required columns:
        loan_id
        calculated_loan_age
        current_loan_delinquency_status
    """
    result = df.copy()

    # ------------------------------------------------------------------
    # Numeric delinquency status
    #
    # Non-numeric values such as "RA" become NaN here.
    # ------------------------------------------------------------------

    result["current_dpd_numeric"] = pd.to_numeric(
        result["current_loan_delinquency_status"],
        errors="coerce",
    )

    # Sort chronologically within each loan before calculating history.
    result = result.sort_values(
        [
            "loan_id",
            "calculated_loan_age",
        ]
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Current delinquency indicators
    # ------------------------------------------------------------------

    result["current_dpd_30_plus"] = (
        result["current_dpd_numeric"].ge(30).fillna(False).astype("int8")
    )

    result["current_dpd_60_plus"] = (
        result["current_dpd_numeric"].ge(60).fillna(False).astype("int8")
    )

    result["current_delinquency_flag"] = (
        result["current_dpd_numeric"].gt(0).fillna(False).astype("int8")
    )

    # ------------------------------------------------------------------
    # Maximum delinquency observed to date
    #
    # min_count=1 preserves NaN when no numeric DPD exists yet.
    # ------------------------------------------------------------------

    result["max_dpd_to_date"] = result.groupby("loan_id")[
        "current_dpd_numeric"
    ].cummax()

    # ------------------------------------------------------------------
    # Whether 30/60 DPD has ever occurred up to the current month.
    # ------------------------------------------------------------------

    result["ever_30dpd_to_date"] = (
        result.groupby("loan_id")["current_dpd_30_plus"].cummax().astype("int8")
    )

    result["ever_60dpd_to_date"] = (
        result.groupby("loan_id")["current_dpd_60_plus"].cummax().astype("int8")
    )

    # ------------------------------------------------------------------
    # Number of months with any numeric delinquency observed to date.
    # ------------------------------------------------------------------

    result["delinquency_months_to_date"] = (
        result.groupby("loan_id")["current_delinquency_flag"].cumsum().astype("int16")
    )

    # ------------------------------------------------------------------
    # Most recent month with numeric delinquency.
    #
    # Store the corresponding loan age rather than calendar period so
    # that the resulting feature is directly comparable across loans.
    # ------------------------------------------------------------------

    delinquency_age = result["calculated_loan_age"].where(
        result["current_delinquency_flag"].eq(1)
    )

    last_delinquency_age = delinquency_age.groupby(result["loan_id"]).ffill()

    result["months_since_last_delinquency"] = (
        result["calculated_loan_age"] - last_delinquency_age
    )

    # Loans with no prior delinquency have no meaningful "months since"
    # value. Keep these as NaN rather than assigning an arbitrary zero.
    result.loc[
        last_delinquency_age.isna(),
        "months_since_last_delinquency",
    ] = pd.NA

    return result


def add_loan_trajectory_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add simple post-origination loan trajectory features.

    Features are calculated using only information available at the
    current observation point.

    Required columns:
        original_upb
        current_actual_upb
        original_interest_rate
        current_interest_rate
    """
    result = df.copy()

    # --------------------------------------------------------------
    # UPB change from origination
    # --------------------------------------------------------------

    result["upb_change_from_origination"] = (
        result["current_actual_upb"] - result["original_upb"]
    )

    # --------------------------------------------------------------
    # Percentage UPB change from origination
    #
    # Preserve missing values and avoid division by zero.
    # --------------------------------------------------------------

    result["upb_pct_change_from_origination"] = result[
        "upb_change_from_origination"
    ].div(result["original_upb"].replace(0, pd.NA))

    # --------------------------------------------------------------
    # Interest-rate change from origination
    # --------------------------------------------------------------

    result["rate_change_from_origination"] = (
        result["current_interest_rate"] - result["original_interest_rate"]
    )

    return result
