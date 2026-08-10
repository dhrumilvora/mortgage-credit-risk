import pandas as pd
import pytest

from credit_risk.features.feature_engineering import apply_transformations


def test_credit_score_binning_uses_configured_labels() -> None:
    config = {
        "parameters": {
            "feature_engineering": {
                "transformations": {
                    "credit_score": {
                        "enabled": True,
                        "method": "bin",
                        "bins": [0, 650, 700, "+inf"],
                        "labels": ["low", "medium", "high"],
                    }
                }
            }
        }
    }
    result = apply_transformations(pd.DataFrame({"credit_score": [649, 650, 800]}), config)
    assert result["credit_score_bins"].astype(str).tolist() == ["low", "medium", "high"]


def test_unsupported_transformation_is_rejected() -> None:
    config = {"parameters": {"feature_engineering": {"transformations": {"credit_score": {"method": "scale"}}}}}
    with pytest.raises(ValueError, match="Unsupported transformation"):
        apply_transformations(pd.DataFrame({"credit_score": [700]}), config)
