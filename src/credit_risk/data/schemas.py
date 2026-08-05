from dataclasses import dataclass

@dataclass(frozen = True)
class Field:
    position: int
    source_name: str
    name: str
    data_type: str
    max_length: int

PERFORMANCE_SCHEMA = (
    Field(1, "Loan Identifier", "loan_id", "string", 12),
    Field(2, "Period", "period", "date_yyyymm", 6),
    Field(3, "Current Actual UPB", "current_actual_upb", "numeric", 12),
    Field(
        4,
        "Current Loan Delinquency Status",
        "current_loan_delinquency_status",
        "string",
        3,
    ),
    Field(5, "Loan Age", "loan_age", "numeric", 3),
    Field(
        6,
        "Remaining Months to Legal Maturity",
        "remaining_months_to_legal_maturity",
        "numeric",
        3,
    ),
    Field(
        7,
        "Underwriting Defect and Major Servicing Defect Settlement Date",
        "defect_settlement_date",
        "date_yyyymm",
        6,
    ),
    Field(8, "Modification Flag", "modification_flag", "string", 1),
    Field(9, "Zero Balance Code", "zero_balance_code", "numeric", 2),
    Field(
        10,
        "Zero Balance Effective Date",
        "zero_balance_effective_date",
        "date_yyyymm",
        6,
    ),
    Field(11, "Current Interest Rate", "current_interest_rate", "numeric", 8),
    Field(
        12,
        "Current Non-Interest Bearing UPB",
        "current_non_interest_bearing_upb",
        "numeric",
        12,
    ),
    Field(
        13,
        "Due Date of Last Paid Installment (DDLPI)",
        "ddlpi",
        "date_yyyymm",
        6,
    ),
    Field(14, "MI Recoveries", "mi_recoveries", "numeric", 12),
    # Important: Freddie defines this as Alpha-Numeric, not numeric.
    Field(15, "Net Sales Proceeds", "net_sales_proceeds", "string", 14),
    Field(16, "Non MI Recoveries", "non_mi_recoveries", "numeric", 12),
    Field(17, "Total Expenses", "total_expenses", "numeric", 12),
    Field(18, "Legal Costs", "legal_costs", "numeric", 12),
    Field(
        19,
        "Maintenance and Preservation Costs",
        "maintenance_preservation_costs",
        "numeric",
        12,
    ),
    Field(20, "Taxes and Insurance", "taxes_and_insurance", "numeric", 12),
    Field(
        21,
        "Miscellaneous Expenses",
        "miscellaneous_expenses",
        "numeric",
        12,
    ),
    Field(22, "Actual Loss", "actual_loss", "numeric", 12),
    Field(
        23,
        "Cumulative Modification Costs",
        "cumulative_modification_costs",
        "numeric",
        12,
    ),
    Field(
        24,
        "Interest Rate Step Indicator",
        "interest_rate_step_indicator",
        "string",
        1,
    ),
    Field(
        25,
        "Payment Deferral Flag",
        "payment_deferral_flag",
        "string",
        1,
    ),
    Field(
        26,
        "Estimated Loan-to-Value (ELTV)",
        "estimated_ltv",
        "numeric",
        4,
    ),
    Field(
        27,
        "Zero Balance Removal UPB",
        "zero_balance_removal_upb",
        "numeric",
        12,
    ),
    Field(
        28,
        "Delinquent Accrued Interest",
        "delinquent_accrued_interest",
        "numeric",
        12,
    ),
    Field(
        29,
        "Delinquency Due to Disaster",
        "delinquency_due_to_disaster",
        "string",
        1,
    ),
    Field(
        30,
        "Borrower Assistance Plan",
        "borrower_assistance_plan",
        "string",
        1,
    ),
    Field(
        31,
        "Current Period Modification Costs",
        "current_period_modification_costs",
        "numeric",
        12,
    ),
    Field(
        32,
        "Current Interest Bearing UPB",
        "current_interest_bearing_upb",
        "numeric",
        12,
    ),
    Field(
        33,
        "Mortgage Insurance Cancellation Indicator",
        "mi_cancellation_indicator",
        "string",
        1,
    ),
    Field(34, "Servicer Name", "servicer_name", "string", 60),
    Field(
        35,
        "Bankruptcy Cramdown Costs",
        "bankruptcy_cramdown_costs",
        "numeric",
        12,
    ),
)


def get_column_names(schema: tuple[Field, ...]) -> list[str]:
    return [field.name for field in schema]
