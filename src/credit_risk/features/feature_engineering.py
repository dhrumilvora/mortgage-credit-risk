import pandas as pd
import numpy as np


def apply_binning(
    df: pd.DataFrame,
    feature: str,
    transformation_config: dict,
) -> pd.DataFrame:
    def _parse_bin(value):
        if isinstance(value, str):
            if value.lower() in {"inf", "+inf"}:
                return np.inf
            if value.lower() == "-inf":
                return -np.inf

        return value

    result = df.copy()

    if not transformation_config.get("enabled", False):
        return result

    if transformation_config.get("method") != "bin":
        raise ValueError(
            f"Unsupported transformation method for '{feature}': "
            f"{transformation_config.get('method')}"
        )

    if feature not in result.columns:
        raise ValueError(f"Feature '{feature}' not found in dataframe.")

    result[f"{feature}_bins"] = pd.cut(
        result[feature],
        bins=[_parse_bin(value) for value in transformation_config["bins"]],
        labels=transformation_config["labels"],
        right=False,
    )

    return result


def apply_transformations(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Apply configured feature transformations."""

    result = df.copy()

    transformations = (
        config["parameters"].get("feature_engineering", {}).get("transformations", {})
    )

    for feature, transformation_config in transformations.items():
        if transformation_config.get("method") == "bin":
            result = apply_binning(
                result,
                feature,
                transformation_config,
            )
        elif transformation_config.get("method") == "log":
            if (result[feature] <= 0).any():
                raise ValueError(
                    f"Cannot apply log transformation to '{feature}': "
                    "feature contains zero or negative values."
                )
            result[f"{feature}_log"] = np.log(result[feature])
        else:
            raise ValueError(
                f"Unsupported transformation method for '{feature}': "
                f"{transformation_config.get('method')}"
            )

    return result
