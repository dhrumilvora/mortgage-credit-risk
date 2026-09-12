from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from credit_risk.utils.config import create_path

PROJECT_ROOT = Path(__file__).resolve().parent


@st.cache_data
def load_config():
    """
    Load the existing project configuration.

    This deliberately uses the same configuration files as P1.
    """

    parameters_dir = PROJECT_ROOT / "config" / "parameters"

    catalog_path = PROJECT_ROOT / "config" / "catalog" / "base.yml"

    with (parameters_dir / "base.yml").open(
        "r",
        encoding="utf-8",
    ) as file:
        base = yaml.safe_load(file)

    approach = base["parameters"]["modelling_approach"]

    with (parameters_dir / f"{approach}.yml").open(
        "r",
        encoding="utf-8",
    ) as file:
        approach_config = yaml.safe_load(file)

    with catalog_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        catalog = yaml.safe_load(file)

    parameters = base

    def deep_merge(
        left,
        right,
    ):
        result = dict(left)

        for key, value in right.items():

            if (
                key in result
                and isinstance(
                    result[key],
                    dict,
                )
                and isinstance(
                    value,
                    dict,
                )
            ):
                result[key] = deep_merge(
                    result[key],
                    value,
                )
            else:
                result[key] = value

        return result

    parameters = deep_merge(
        parameters,
        approach_config,
    )

    return {
        **parameters,
        **catalog,
    }


@st.cache_data
def load_scores(
    config,
    vintage,
):
    approach = config["parameters"]["modelling_approach"]

    provider = config["parameters"]["data"]["data_provider"]

    path = create_path(
        config["catalog"]["base"],
        config["catalog"],
        "early_intervention_output",
        approach,
        provider,
        vintage,
        must_exist=True,
    )

    return pd.read_parquet(
        path,
    )


st.set_page_config(
    page_title="Mortgage Early Intervention",
    layout="wide",
)


config = load_config()

st.title("Mortgage Early Intervention System")

st.caption("Frozen behavioral P1 GAM • " "3 / 6 / 9 / 12 month intervention points")

data_config = config["parameters"]["data"]

vintages = data_config["all_vintages"]

pd_horizon = config["parameters"]["behavioral"]["prediction_horizon_months"]

pd_column = f"p1_pd_{pd_horizon}m"


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

st.sidebar.header("Controls")

vintage = st.sidebar.selectbox(
    "Vintage",
    vintages,
)

df = load_scores(
    config,
    vintage,
)

periods = sorted(df["period"].dropna().unique())

selected_period = st.sidebar.selectbox(
    "Reporting period",
    periods,
)

available_ages = sorted(
    df.loc[
        df["period"] == selected_period,
        "observation_age",
    ]
    .dropna()
    .unique()
)

selected_age = st.sidebar.selectbox(
    "Observation age",
    available_ages,
)

threshold_default = (
    config["parameters"]
    .get(
        "early_intervention",
        {},
    )
    .get(
        "threshold",
        config["parameters"]["evaluation"]["classification"]["threshold"],
    )
)

display_threshold = st.sidebar.slider(
    "Display threshold",
    min_value=0.0,
    max_value=1.0,
    value=float(threshold_default),
    step=0.005,
)


# ------------------------------------------------------------------
# Filter current operational population
# ------------------------------------------------------------------

current = df[
    (df["period"] == selected_period) & (df["observation_age"] == selected_age)
].copy()

current["display_flag"] = current[pd_column] >= display_threshold


# ------------------------------------------------------------------
# Portfolio summary
# ------------------------------------------------------------------

total_loans = len(current)

flagged = current[current["display_flag"]]

flagged_loans = len(flagged)

flagged_upb = flagged["current_actual_upb"].sum()

total_upb = current["current_actual_upb"].sum()

flagged_pct = flagged_loans / total_loans if total_loans > 0 else 0.0

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Loans at observation point",
    f"{total_loans:,}",
)

col2.metric(
    "Flagged loans",
    f"{flagged_loans:,}",
)

col3.metric(
    "Flagged population",
    f"{flagged_pct:.1%}",
)

col4.metric(
    "Flagged UPB",
    f"${flagged_upb:,.0f}",
)


# ------------------------------------------------------------------
# Intervention queue
# ------------------------------------------------------------------

st.subheader("Intervention Queue")

queue_columns = [
    "intervention_rank",
    "loan_id",
    "vintage",
    "period",
    "observation_age",
    pd_column,
    "current_actual_upb",
]

queue = flagged[queue_columns].sort_values("intervention_rank")

st.dataframe(
    queue,
    use_container_width=True,
    hide_index=True,
)


# ------------------------------------------------------------------
# Individual loan view
# ------------------------------------------------------------------

st.subheader("Loan Detail")

loan_ids = queue["loan_id"].tolist()

if loan_ids:

    selected_loan = st.selectbox(
        "Select loan",
        loan_ids,
    )

    loan = current[current["loan_id"] == selected_loan].iloc[0]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "3-Month PD",
        f"{loan[pd_column]:.2%}",
    )

    c2.metric(
        "UPB",
        f"${loan['current_actual_upb']:,.0f}",
    )

    c3.metric(
        "Loan age",
        int(loan["calculated_loan_age"]),
    )

    detail_columns = [
        "loan_id",
        "vintage",
        "period",
        "observation_age",
        "calculated_loan_age",
        pd_column,
        "current_actual_upb",
        "current_interest_rate",
        "estimated_ltv",
        "max_dpd_to_date",
        "delinquency_months_to_date",
        "months_since_last_delinquency",
        "dpd_30_count_3m",
        "dpd_60_count_3m",
        "max_dpd_3m",
        "dpd_trend_6m",
    ]

    available_detail_columns = [
        column for column in detail_columns if column in current.columns
    ]

    st.dataframe(
        loan[available_detail_columns].to_frame("value"),
        use_container_width=True,
    )

else:

    st.info(
        "No loans meet the selected intervention threshold "
        "for this period and observation age."
    )
