import pandas as pd

from credit_risk.reporting.data_quality import (
    build_dataset_summary,
    build_grain_checks,
    build_target_summary,
)


def test_target_summary():

    model_input = pd.DataFrame(
        {
            "loan_id": ["A", "B", "C", "D"],
            "ever_90dpd_24m": [0, 0, 0, 1],
        }
    )

    result = build_target_summary(
        model_input,
        target_col="ever_90dpd_24m",
    )

    values = result.set_index("metric")["value"]

    assert values["Total Loans"] == 4
    assert values["Events"] == 1
    assert values["Non-Events"] == 3
    assert values["Event Rate"] == 0.25
    assert values["Missing Target"] == 0


def test_grain_checks_detect_performance_duplicates():

    data = {
        "origination": pd.DataFrame(
            {
                "loan_id": ["A", "B"],
            }
        ),
        "performance": pd.DataFrame(
            {
                "loan_id": ["A", "A", "A"],
                "period": [1, 1, 2],
            }
        ),
        "model_input": pd.DataFrame(
            {
                "loan_id": ["A", "B"],
            }
        ),
    }

    result = build_grain_checks(
        data,
        id_col="loan_id",
        time_col="period",
    )

    performance = result.loc[result["dataset"] == "performance"].iloc[0]

    assert performance["duplicate_records"] == 1
    assert performance["status"] == "FAIL"
