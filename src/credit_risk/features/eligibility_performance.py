IDENTIFIER_FIELDS = [
    "loan_id",
]

TIME_FIELDS = [
    "period",
]

BASELINE_FEATURES = [
    "current_actual_upb",
    "current_interest_rate",
    "loan_age",
    "remaining_months_to_legal_maturity",
    "estimated_ltv",
]

STATE_FIELDS = [
    "current_loan_delinquency_status",
    "ddlpi",
]

TERMINATION_FIELDS = [
    "zero_balance_code",
    "zero_balance_effective_date",
]

CHALLENGER_FEATURES = [
    "modification_flag",
    "current_non_interest_bearing_upb",
    "current_interest_bearing_upb",
    "interest_rate_step_indicator",
    "payment_deferral_flag",
    "delinquency_due_to_disaster",
    "borrower_assistance_plan",
    "mi_cancellation_indicator",
    "servicer_name",
]

CHALLENGER_FEATURES = [
    "modification_flag",
    "current_non_interest_bearing_upb",
    "current_interest_bearing_upb",
    "interest_rate_step_indicator",
    "payment_deferral_flag",
    "delinquency_due_to_disaster",
    "borrower_assistance_plan",
    "mi_cancellation_indicator",
    "servicer_name",
    "delinquent_accrued_interest",
]


def validate_features(columns) -> None:
    """Validate fields required for baseline performance preprocessing."""

    required = (
        IDENTIFIER_FIELDS
        + TIME_FIELDS
        + BASELINE_FEATURES
        + STATE_FIELDS
        + TERMINATION_FIELDS
        + CHALLENGER_FEATURES
    )

    missing = sorted(set(required) - set(columns))

    if missing:
        raise ValueError("Missing required performance fields: " + ", ".join(missing))
