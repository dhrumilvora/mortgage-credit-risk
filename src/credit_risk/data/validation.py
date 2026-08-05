from credit_risk.data.schemas import (
    ORIGINATION_SCHEMA,
    PERFORMANCE_SCHEMA,
    get_column_names,
)
from dataclasses import dataclass, field
import pandas as pd


class DataValidationError(Exception):
    """ "Raised when input data violates data requirement schemas"""


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def raise_if_invalid(self) -> None:
        if not self.is_valid:
            formatted = "\n".join(f"- {error}" for error in self.errors)
            raise DataValidationError(f"Data validation failed:\n{formatted}")


def validate_required_columns(
    df: pd.DataFrame, expected_columns: list[str], result: ValidationResult
) -> None:
    missing = sorted(set(expected_columns) - set(df.columns))
    unexpected = sorted(set(df.columns) - set(expected_columns))
    if missing:
        result.add_error(f"Missing required columns: {missing}")

    if unexpected:
        result.add_error(f"Unexpected Columns: {unexpected}")


def validate_non_empty(
    df: pd.DataFrame,
    result: ValidationResult,
) -> None:

    if df.empty:
        result.add_error("Dataset contains no rows.")


def validate_loan_id_not_null(
    df: pd.DataFrame,
    result: ValidationResult,
) -> None:

    null_count = df["loan_id"].isna().sum()

    if null_count:
        result.add_error(f"{null_count:,} rows contain a null loan_id.")


def validate_loan_id_not_blank(
    df: pd.DataFrame,
    result: ValidationResult,
) -> None:

    blank_count = df["loan_id"].astype("string").str.strip().eq("").sum()

    if blank_count:
        result.add_error(f"{blank_count:,} rows contain a blank loan_id.")


def validate_unique_loan_id(
    df: pd.DataFrame,
    result: ValidationResult,
) -> None:

    duplicate_count = df["loan_id"].duplicated().sum()

    if duplicate_count:
        result.add_error(f"{duplicate_count:,} duplicate loan_id rows found.")


def validate_unique_loan_period(
    df: pd.DataFrame,
    result: ValidationResult,
) -> None:

    duplicate_count = df.duplicated(subset=["loan_id", "period"]).sum()

    if duplicate_count:
        result.add_error(
            f"{duplicate_count:,} duplicate " f"(loan_id, period) records found."
        )


def validate_origination(
    df: pd.DataFrame,
) -> ValidationResult:

    result = ValidationResult()

    validate_non_empty(df, result)

    validate_required_columns(
        df,
        get_column_names(ORIGINATION_SCHEMA),
        result,
    )

    if "loan_id" in df.columns:
        validate_loan_id_not_null(df, result)
        validate_loan_id_not_blank(df, result)
        validate_unique_loan_id(df, result)

    return result


def validate_performance(
    df: pd.DataFrame,
) -> ValidationResult:

    result = ValidationResult()

    validate_non_empty(df, result)

    validate_required_columns(
        df,
        get_column_names(PERFORMANCE_SCHEMA),
        result,
    )

    if "loan_id" in df.columns:
        validate_loan_id_not_null(df, result)
        validate_loan_id_not_blank(df, result)

    if {"loan_id", "period"}.issubset(df.columns):
        validate_unique_loan_period(df, result)

    return result


def validate_referential_integrity(
    origination: pd.DataFrame,
    performance: pd.DataFrame,
) -> ValidationResult:

    result = ValidationResult()

    orig_ids = set(origination["loan_id"].dropna().unique())

    perf_ids = set(performance["loan_id"].dropna().unique())

    orphan_ids = perf_ids - orig_ids

    if orphan_ids:
        result.add_error(
            f"{len(orphan_ids):,} performance loan IDs "
            f"do not exist in origination data."
        )

    return result
