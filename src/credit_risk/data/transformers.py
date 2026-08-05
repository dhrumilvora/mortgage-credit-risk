import pandas as pd

ORIGINATION_INTEGER_COLUMNS = [
    "credit_score",
    "mi_percentage",
    "number_of_units",
    "original_cltv",
    "original_dti",
    "original_upb",
    "original_ltv",
    "original_loan_term",
    "number_of_borrowers",
    "vantage_score_4",
]


ORIGINATION_FLOAT_COLUMNS = [
    "original_interest_rate",
]


ORIGINATION_STRING_COLUMNS = [
    "first_time_homebuyer_flag",
    "msa",
    "occupancy_status",
    "channel",
    "prepayment_penalty_flag",
    "amortization_type",
    "property_state",
    "property_type",
    "postal_code",
    "loan_id",
    "loan_purpose",
    "seller_name",
    "super_conforming_flag",
    "pre_harp_loan_id",
    "special_eligibility_program",
    "harp_indicator",
    "property_valuation_method",
    "interest_only_indicator",
]


ORIGINATION_DATE_COLUMNS = [
    "first_payment_date",
    "maturity_date",
]


def to_nullable_integer(series: pd.Series) -> pd.Series:
    """Convert a series to pandas nullable integer dtype."""
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def to_nullable_float(series: pd.Series) -> pd.Series:
    """Convert a series to pandas nullable float dtype."""
    return pd.to_numeric(series, errors="coerce").astype("Float64")


def to_string(series: pd.Series) -> pd.Series:
    """Convert a series to pandas nullable string dtype."""
    return series.astype("string")


def to_month_period(series: pd.Series) -> pd.Series:
    """Convert Freddie YYYYMM values to monthly Period values."""

    values = series.astype("string")

    parsed = pd.to_datetime(
        values,
        format="%Y%m",
        errors="coerce",
    )

    return parsed.dt.to_period("M")


def transform_origination(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Transform raw Freddie origination data into canonical types."""

    result = df.copy()

    for column in ORIGINATION_INTEGER_COLUMNS:
        result[column] = to_nullable_integer(result[column])

    for column in ORIGINATION_FLOAT_COLUMNS:
        result[column] = to_nullable_float(result[column])

    for column in ORIGINATION_STRING_COLUMNS:
        result[column] = to_string(result[column])

    for column in ORIGINATION_DATE_COLUMNS:
        result[column] = to_month_period(result[column])

    return result


PERFORMANCE_INTEGER_COLUMNS = [
    "loan_age",
    "remaining_months_to_legal_maturity",
    "estimated_ltv",
]


PERFORMANCE_FLOAT_COLUMNS = [
    "current_actual_upb",
    "current_interest_rate",
    "current_non_interest_bearing_upb",
    "mi_recoveries",
    "non_mi_recoveries",
    "total_expenses",
    "legal_costs",
    "maintenance_preservation_costs",
    "taxes_and_insurance",
    "miscellaneous_expenses",
    "actual_loss",
    "cumulative_modification_costs",
    "zero_balance_removal_upb",
    "delinquent_accrued_interest",
    "current_period_modification_costs",
    "current_interest_bearing_upb",
    "bankruptcy_cramdown_costs",
]


PERFORMANCE_STRING_COLUMNS = [
    "loan_id",
    "current_loan_delinquency_status",
    "modification_flag",
    "zero_balance_code",
    "net_sales_proceeds",
    "interest_rate_step_indicator",
    "payment_deferral_flag",
    "delinquency_due_to_disaster",
    "borrower_assistance_plan",
    "mi_cancellation_indicator",
    "servicer_name",
]


PERFORMANCE_DATE_COLUMNS = [
    "period",
    "defect_settlement_date",
    "zero_balance_effective_date",
    "ddlpi",
]


def transform_performance(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Transform raw Freddie performance data into canonical types."""

    result = df.copy()

    for column in PERFORMANCE_INTEGER_COLUMNS:
        result[column] = to_nullable_integer(result[column])

    for column in PERFORMANCE_FLOAT_COLUMNS:
        result[column] = to_nullable_float(result[column])

    for column in PERFORMANCE_STRING_COLUMNS:
        result[column] = to_string(result[column])

    for column in PERFORMANCE_DATE_COLUMNS:
        result[column] = to_month_period(result[column])

    return result
