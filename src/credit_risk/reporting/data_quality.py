"""Data-quality reporting for the mortgage credit-risk pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.utils.config import create_path

# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------


def load_reporting_data(
    config: dict,
) -> dict[str, pd.DataFrame]:
    """Load persisted pipeline datasets required for reporting."""

    parameters = config["parameters"]
    catalog = config["catalog"]

    base_path = catalog["base"]
    provider = parameters["data"]["data_provider"]
    vintage = parameters["reporting"]["vintage"]
    modelling_approach = parameters["modelling_approach"]
    origination_path = create_path(
        base_path,
        catalog,
        "origination_path",
        provider,
        vintage,
    )

    performance_path = create_path(
        base_path,
        catalog,
        "performance_path",
        provider,
        vintage,
    )

    model_input_path = create_path(
        base_path,
        catalog,
        "model_input_path",
        modelling_approach,
        provider,
        vintage,
    )

    return {
        "origination": pd.read_parquet(origination_path),
        "performance": pd.read_parquet(performance_path),
        "model_input": pd.read_parquet(model_input_path),
    }


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------


def build_dataset_summary(
    data: dict[str, pd.DataFrame],
    id_col: str,
) -> pd.DataFrame:
    """Build high-level structural statistics for pipeline datasets."""

    rows = []

    for dataset_name, df in data.items():
        rows.append(
            {
                "dataset": dataset_name,
                "rows": len(df),
                "columns": df.shape[1],
                "unique_loans": df[id_col].nunique(),
                "missing_loan_ids": int(df[id_col].isna().sum()),
                "duplicate_rows": int(df.duplicated().sum()),
            }
        )

    return pd.DataFrame(rows)


def build_target_summary(
    model_input: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    """Build high-level target statistics."""

    target = model_input[target_col]

    total = len(model_input)
    events = int(target.eq(1).sum())
    non_events = int(target.eq(0).sum())

    return pd.DataFrame(
        [
            {"metric": "Total Loans", "value": total},
            {"metric": "Events", "value": events},
            {"metric": "Non-Events", "value": non_events},
            {
                "metric": "Event Rate",
                "value": events / total if total else np.nan,
            },
            {
                "metric": "Missing Target",
                "value": int(target.isna().sum()),
            },
        ]
    )


# ---------------------------------------------------------------------
# Dataset reconciliation
# ---------------------------------------------------------------------


def build_dataset_reconciliation(
    data: dict[str, pd.DataFrame],
    id_col: str,
) -> pd.DataFrame:
    """Reconcile loan populations across pipeline stages."""

    origination = data["origination"]
    performance = data["performance"]
    model_input = data["model_input"]

    orig_loans = set(origination[id_col].dropna())
    perf_loans = set(performance[id_col].dropna())
    model_loans = set(model_input[id_col].dropna())

    rows = [
        {
            "metric": "Origination Loans",
            "value": len(orig_loans),
        },
        {
            "metric": "Performance Loans",
            "value": len(perf_loans),
        },
        {
            "metric": "Model Input Loans",
            "value": len(model_loans),
        },
        {
            "metric": "Origination Without Performance",
            "value": len(orig_loans - perf_loans),
        },
        {
            "metric": "Performance Without Origination",
            "value": len(perf_loans - orig_loans),
        },
        {
            "metric": "Model Loans Not In Origination",
            "value": len(model_loans - orig_loans),
        },
        {
            "metric": "Model Loans Not In Performance",
            "value": len(model_loans - perf_loans),
        },
        {
            "metric": "Population Retained",
            "value": (len(model_loans) / len(orig_loans) if orig_loans else np.nan),
        },
    ]

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Grain checks
# ---------------------------------------------------------------------


def build_grain_checks(
    data: dict[str, pd.DataFrame],
    id_col: str,
    time_col: str,
) -> pd.DataFrame:
    """Validate expected grain for each pipeline dataset."""

    origination = data["origination"]
    performance = data["performance"]
    model_input = data["model_input"]

    checks = []

    orig_duplicates = int(origination.duplicated(subset=[id_col]).sum())

    checks.append(
        {
            "dataset": "origination",
            "expected_grain": id_col,
            "duplicate_records": orig_duplicates,
            "status": "PASS" if orig_duplicates == 0 else "FAIL",
        }
    )

    perf_duplicates = int(performance.duplicated(subset=[id_col, time_col]).sum())

    checks.append(
        {
            "dataset": "performance",
            "expected_grain": f"{id_col} + {time_col}",
            "duplicate_records": perf_duplicates,
            "status": "PASS" if perf_duplicates == 0 else "FAIL",
        }
    )

    model_duplicates = int(model_input.duplicated(subset=[id_col]).sum())

    checks.append(
        {
            "dataset": "model_input",
            "expected_grain": id_col,
            "duplicate_records": model_duplicates,
            "status": "PASS" if model_duplicates == 0 else "FAIL",
        }
    )

    return pd.DataFrame(checks)


# ---------------------------------------------------------------------
# Cohort funnel
# ---------------------------------------------------------------------


def build_cohort_funnel(
    performance: pd.DataFrame,
    model_input: pd.DataFrame,
    target_config: dict,
    id_col: str,
) -> pd.DataFrame:
    """Reconstruct the target-eligibility funnel for reporting."""

    horizon = target_config["horizon_months"]
    max_start_age = target_config["max_eligible_start_age"]
    voluntary_payoff = target_config["voluntary_payoffs_zbc"]
    threshold = target_config["serious_delinquency_threshold"]

    perf = performance.copy()

    delinquency_numeric = pd.to_numeric(
        perf["current_loan_delinquency_status"],
        errors="coerce",
    )

    perf["_serious"] = delinquency_numeric.ge(threshold) | perf[
        "current_loan_delinquency_status"
    ].eq("RA")

    first_age = perf.groupby(id_col)["loan_age"].min()

    start_eligible_ids = set(first_age[first_age <= max_start_age].index)

    within_horizon = perf[perf["loan_age"].between(0, horizon)].copy()

    observable = (
        within_horizon.sort_values([id_col, "loan_age"])
        .groupby(id_col)
        .agg(
            last_loan_age=("loan_age", "max"),
            ever_serious=("_serious", "max"),
            final_zero_balance_code=("zero_balance_code", "last"),
        )
    )

    observable["completed_horizon"] = observable["last_loan_age"] >= horizon

    observable["voluntary_early_payoff"] = observable["last_loan_age"].lt(
        horizon
    ) & observable["final_zero_balance_code"].eq(voluntary_payoff).fillna(False)

    observable["is_observable"] = (
        observable["ever_serious"].fillna(False)
        | observable["completed_horizon"].fillna(False)
        | observable["voluntary_early_payoff"].fillna(False)
    )

    observable_ids = set(observable.index[observable["is_observable"]])

    eligible_ids = start_eligible_ids & observable_ids

    rows = [
        {
            "stage": "Performance Population",
            "loans": perf[id_col].nunique(),
        },
        {
            "stage": "Start Eligible",
            "loans": len(start_eligible_ids),
        },
        {
            "stage": "Outcome Observable",
            "loans": len(observable_ids),
        },
        {
            "stage": "Target Eligible",
            "loans": len(eligible_ids),
        },
        {
            "stage": "Final Model Input",
            "loans": model_input[id_col].nunique(),
        },
    ]

    result = pd.DataFrame(rows)

    result["retention_from_previous"] = result["loans"].div(result["loans"].shift(1))

    return result


# ---------------------------------------------------------------------
# Missingness
# ---------------------------------------------------------------------


def build_missingness_report(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Build feature-level missingness statistics."""

    rows = []

    for column in df.columns:
        rows.append(
            {
                "feature": column,
                "dtype": str(df[column].dtype),
                "missing_count": int(df[column].isna().sum()),
                "missing_pct": df[column].isna().mean(),
                "unique_values": df[column].nunique(dropna=True),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["missing_pct", "feature"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# Numeric profile
# ---------------------------------------------------------------------


def build_numeric_profile(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Build descriptive statistics for numerical features."""

    numeric = df.select_dtypes(include=np.number)

    if numeric.empty:
        return pd.DataFrame()

    profile = numeric.describe(
        percentiles=[
            0.01,
            0.05,
            0.50,
            0.95,
            0.99,
        ]
    ).T

    profile = profile.rename(
        columns={
            "1%": "p01",
            "5%": "p05",
            "50%": "median",
            "95%": "p95",
            "99%": "p99",
        }
    )

    profile["missing_count"] = numeric.isna().sum()
    profile["missing_pct"] = numeric.isna().mean()

    return profile.reset_index(names="feature")


# ---------------------------------------------------------------------
# Categorical profile
# ---------------------------------------------------------------------


def build_categorical_profile(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Build descriptive statistics for categorical features."""

    categorical = df.select_dtypes(include=["object", "string", "category"])

    rows = []

    for column in categorical.columns:
        series = df[column]

        counts = series.value_counts(
            dropna=False,
        )

        if counts.empty:
            top_value = None
            top_count = 0
            top_share = np.nan
        else:
            top_value = counts.index[0]
            top_count = int(counts.iloc[0])
            top_share = top_count / len(series)

        rows.append(
            {
                "feature": column,
                "unique_values": series.nunique(dropna=True),
                "missing_count": int(series.isna().sum()),
                "missing_pct": series.isna().mean(),
                "top_value": top_value,
                "top_count": top_count,
                "top_share": top_share,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Value checks
# ---------------------------------------------------------------------


def build_value_checks(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Check numerical features for invalid values."""

    numeric = df.select_dtypes(include=np.number)

    rows = []

    for column in numeric.columns:
        series = numeric[column]

        inf_count = int(np.isinf(series.to_numpy()).sum())

        rows.append(
            {
                "feature": column,
                "infinite_count": inf_count,
                "negative_count": int(series.lt(0).sum()),
                "zero_count": int(series.eq(0).sum()),
                "minimum": series.min(),
                "maximum": series.max(),
                "status": "PASS" if inf_count == 0 else "FAIL",
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Sentinel checks
# ---------------------------------------------------------------------


def build_sentinel_checks(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Check whether known Freddie Mac sentinel values remain after
    preprocessing.
    """

    sentinel_rules = {
        "credit_score": [9999],
        "original_dti": [999],
        "original_ltv": [999],
        "original_cltv": [999],
    }

    rows = []

    for feature, sentinels in sentinel_rules.items():
        if feature not in df.columns:
            continue

        count = int(df[feature].isin(sentinels).sum())

        rows.append(
            {
                "feature": feature,
                "sentinel_values": str(sentinels),
                "remaining_count": count,
                "status": "PASS" if count == 0 else "FAIL",
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Constant / near-constant features
# ---------------------------------------------------------------------


def build_constant_feature_checks(
    df: pd.DataFrame,
    exclude: set[str] | None = None,
    near_constant_threshold: float = 0.99,
) -> pd.DataFrame:
    """Identify constant and near-constant features."""

    exclude = exclude or set()

    rows = []

    for column in df.columns:
        if column in exclude:
            continue

        series = df[column]

        counts = series.value_counts(
            normalize=True,
            dropna=False,
        )

        unique_count = series.nunique(dropna=False)

        dominant_share = counts.iloc[0] if not counts.empty else np.nan

        if unique_count <= 1:
            classification = "CONSTANT"
            status = "WARN"

        elif dominant_share >= near_constant_threshold:
            classification = "NEAR_CONSTANT"
            status = "WARN"

        else:
            classification = "OK"
            status = "PASS"

        rows.append(
            {
                "feature": column,
                "unique_values": unique_count,
                "dominant_share": dominant_share,
                "classification": classification,
                "status": status,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Leakage checks
# ---------------------------------------------------------------------


def build_leakage_checks(
    df: pd.DataFrame,
    target_col: str,
    id_col: str,
) -> pd.DataFrame:
    """Detect fields that should not enter an origination-time model."""

    performance_fields = {
        "period",
        "current_actual_upb",
        "current_interest_rate",
        "loan_age",
        "remaining_months_to_legal_maturity",
        "estimated_ltv",
        "current_loan_delinquency_status",
        "ddlpi",
        "zero_balance_code",
        "zero_balance_effective_date",
        "modification_flag",
        "current_non_interest_bearing_upb",
        "current_interest_bearing_upb",
        "interest_rate_step_indicator",
        "payment_deferral_flag",
        "delinquency_due_to_disaster",
        "borrower_assistance_plan",
        "mi_cancellation_indicator",
        "servicer_name",
    }

    rows = []

    for column in sorted(performance_fields):
        present = column in df.columns

        rows.append(
            {
                "feature": column,
                "check": "Performance leakage",
                "present": present,
                "status": "FAIL" if present else "PASS",
            }
        )

    rows.append(
        {
            "feature": id_col,
            "check": "Identifier excluded from model features",
            "present": id_col in df.columns,
            "status": "INFO",
        }
    )

    rows.append(
        {
            "feature": target_col,
            "check": "Target excluded from model features",
            "present": target_col in df.columns,
            "status": "INFO",
        }
    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Missing-indicator consistency
# ---------------------------------------------------------------------


def build_indicator_checks(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Check consistency between missing values and missing indicators."""

    rows = []

    indicator_suffix = "_missing"

    indicator_columns = [
        column for column in df.columns if column.endswith(indicator_suffix)
    ]

    for indicator in indicator_columns:
        feature = indicator.removesuffix(indicator_suffix)

        if feature not in df.columns:
            rows.append(
                {
                    "feature": feature,
                    "indicator": indicator,
                    "mismatch_count": np.nan,
                    "status": "WARN",
                }
            )
            continue

        expected = df[feature].isna().astype(int)

        actual = pd.to_numeric(
            df[indicator],
            errors="coerce",
        )

        mismatches = int(actual.ne(expected).sum())

        rows.append(
            {
                "feature": feature,
                "indicator": indicator,
                "mismatch_count": mismatches,
                "status": "PASS" if mismatches == 0 else "FAIL",
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Target vs feature profile
# ---------------------------------------------------------------------


def build_target_feature_profile(
    df: pd.DataFrame,
    target_col: str,
    id_col: str,
) -> pd.DataFrame:
    """Compare numerical feature distributions by target class."""

    numeric_columns = [
        column
        for column in df.select_dtypes(include=np.number).columns
        if column not in {target_col, id_col}
    ]

    rows = []

    for feature in numeric_columns:
        grouped = df.groupby(target_col)[feature].agg(
            count="count",
            mean="mean",
            median="median",
            min="min",
            max="max",
        )

        for target_value, values in grouped.iterrows():
            rows.append(
                {
                    "feature": feature,
                    "target": target_value,
                    "count": values["count"],
                    "mean": values["mean"],
                    "median": values["median"],
                    "min": values["min"],
                    "max": values["max"],
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Issue consolidation
# ---------------------------------------------------------------------


def build_issue_summary(
    reports: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Consolidate WARN and FAIL results from QC tables."""

    issues = []

    for report_name, report in reports.items():
        if report.empty or "status" not in report.columns:
            continue

        flagged = report[report["status"].isin(["WARN", "FAIL"])].copy()

        if flagged.empty:
            continue

        flagged.insert(
            0,
            "report",
            report_name,
        )

        issues.append(flagged)

    if not issues:
        return pd.DataFrame(
            columns=[
                "report",
                "status",
            ]
        )

    return pd.concat(
        issues,
        ignore_index=True,
        sort=False,
    )


# ---------------------------------------------------------------------
# Complete report
# ---------------------------------------------------------------------


def build_data_quality_report(
    data: pd.DataFrame,
    config: dict,
) -> dict[str, pd.DataFrame]:
    """
    Build the complete pre-modelling data-quality report.

    The reporting layer reads persisted pipeline outputs and does not
    modify any underlying datasets.
    """

    parameters = config["parameters"]

    id_col = parameters["data"]["id_col"]
    time_col = parameters["data"]["time"]

    target_config = parameters["target"]
    target_col = target_config["name"]

    model_input = data["model_input"]

    reports = {
        "00_Dataset_Summary": build_dataset_summary(
            data,
            id_col,
        ),
        "01_Reconciliation": build_dataset_reconciliation(
            data,
            id_col,
        ),
        "02_Grain_Checks": build_grain_checks(
            data,
            id_col,
            time_col,
        ),
        "03_Cohort_Funnel": build_cohort_funnel(
            data["performance"],
            model_input,
            target_config,
            id_col,
        ),
        "04_Target": build_target_summary(
            model_input,
            target_col,
        ),
        "05_Missingness": build_missingness_report(
            model_input,
        ),
        "06_Numeric_Profile": build_numeric_profile(
            model_input,
        ),
        "07_Categorical": build_categorical_profile(
            model_input,
        ),
        "08_Value_Checks": build_value_checks(
            model_input,
        ),
        "09_Sentinel_Checks": build_sentinel_checks(
            model_input,
        ),
        "10_Constant_Features": build_constant_feature_checks(
            model_input,
            exclude={id_col, target_col},
        ),
        "11_Leakage_Checks": build_leakage_checks(
            model_input,
            target_col,
            id_col,
        ),
        "12_Indicator_Checks": build_indicator_checks(
            model_input,
        ),
        "13_Target_Features": build_target_feature_profile(
            model_input,
            target_col,
            id_col,
        ),
    }

    reports["14_Issues"] = build_issue_summary(reports)

    return reports
