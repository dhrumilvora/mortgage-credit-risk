"""Origination feature eligibility definitions."""


def validate_baseline_features(columns, config) -> None:
    """Validate that all required baseline features are available."""
    parameters = config["parameters"]
    available = set(columns)
    features = filter(
        None,
        [parameters["data"]["id_col"]]
        + parameters["data"]["preprocess"]["features"]["numerical_features"]
        + parameters["data"]["preprocess"]["features"]["categorical_features"],
    )
    required = set(features)

    missing = sorted(required - available)

    if missing:
        raise ValueError("Missing required baseline features: " + ", ".join(missing))
