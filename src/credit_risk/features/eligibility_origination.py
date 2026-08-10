"""Origination feature eligibility definitions."""

IDENTIFIER_FEATURE = [
    "loan_id",
]


NUMERICAL_BASELINE_FEATURES = [
    # Borrower attributes
    "credit_score",
    "original_dti",
    "number_of_borrowers",
    # Collateral / leverage
    "original_ltv",
    "original_cltv",
    "mi_percentage",
    # Loan structure
    "original_upb",
    "original_interest_rate",
    "original_loan_term",
]


CATEGORICAL_BASELINE_FEATURES = [
    # Borrower attributes
    "first_time_homebuyer_flag",
    # Collateral
    "property_type",
    "occupancy_status",
    # Loan structure
    "loan_purpose",
    "channel",
    "super_conforming_flag",
    # Program
    "harp_indicator",
    # Geography
    "property_state",
]


BASELINE_FEATURES = NUMERICAL_BASELINE_FEATURES + CATEGORICAL_BASELINE_FEATURES


CHALLENGER_FEATURES = [
    "postal_code",
    "msa",
    "seller_name",
    "special_eligibility_program",
]


NON_PREDICTIVE_FIELDS = [
    "pre_harp_loan_id",
]


CONSTANT_OR_UNAVAILABLE_FIELDS = [
    "vantage_score_4",
    "amortization_type",
    "prepayment_penalty_flag",
    "property_valuation_method",
    "interest_only_indicator",
]


TIME_FIELDS = [
    "first_payment_date",
]


REDUNDANT_FIELDS = [
    "maturity_date",
]

ENGINEERED_BASELINE_FEATURES = [
    "original_dti_missing",
]

MODEL_FEATURES = BASELINE_FEATURES + ENGINEERED_BASELINE_FEATURES


def validate_baseline_features(columns) -> None:
    """Validate that all required baseline features are available."""

    available = set(columns)

    required = set(BASELINE_FEATURES + IDENTIFIER_FEATURE)

    missing = sorted(required - available)

    if missing:
        raise ValueError("Missing required baseline features: " + ", ".join(missing))
