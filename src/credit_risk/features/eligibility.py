IDENTIFIER_FEATURE = ["loan_id"]
BASELINE_FEATURES = [
    # Borrower attributes
    "credit_score",
    "original_dti",
    "number_of_borrowers",
    "first_time_homebuyer_flag",
    # Collateral / leverage
    "original_ltv",
    "original_cltv",
    "mi_percentage",
    "property_type",
    "occupancy_status",
    # Loan structure
    "original_upb",
    "original_interest_rate",
    "original_loan_term",
    "loan_purpose",
    "channel",
    "super_conforming_flag",
    # Program
    "harp_indicator",
    # Geography
    "property_state",
]


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


def validate_baseline_features(columns) -> None:
    """Validate that all required baseline features are available."""

    available = set(columns)
    required_cols = BASELINE_FEATURES + IDENTIFIER_FEATURE
    required = set(required_cols)

    missing = sorted(required - available)

    if missing:
        raise ValueError("Missing required baseline features: " + ", ".join(missing))
