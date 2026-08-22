from pathlib import Path
import yaml


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML configuration: {path}") from exc

    if config is None:
        raise ValueError(f"Configuration file is empty: {path}")

    if not isinstance(config, dict):
        raise TypeError(f"Configuration must contain a YAML mapping: {path}")

    return config


def _deep_merge(
    base: dict,
    override: dict,
) -> dict:
    """
    Recursively merge override into base.

    Nested dictionaries are merged recursively. Non-dictionary values
    from override replace values from base.
    """
    result = base.copy()

    for key, value in override.items():

        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(
                result[key],
                value,
            )

        else:
            result[key] = value

    return result


def read_config(
    project_path: Path,
) -> dict:
    """
    Load common and approach-specific parameters together with catalog.

    The returned configuration has the same structure as before:

        config["parameters"]
        config["catalog"]

    The modelling approach is read from parameters/base.yml and determines
    which approach-specific parameter file is loaded.
    """

    parameters_dir = project_path / "config" / "parameters"

    base_parameter_path = parameters_dir / "base.yml"

    catalog_path = project_path / "config" / "catalog" / "base.yml"

    # --------------------------------------------------------------
    # Load common parameters first.
    # --------------------------------------------------------------

    base_parameters = _load_yaml(
        base_parameter_path,
    )

    # --------------------------------------------------------------
    # Resolve modelling approach.
    # --------------------------------------------------------------

    approach = base_parameters["parameters"].get(
        "modelling_approach",
    )

    if not approach:
        raise ValueError("Missing 'modelling_approach' in " f"{base_parameter_path}")

    # --------------------------------------------------------------
    # Load approach-specific parameters.
    #
    # Example:
    #   modelling_approach: origination
    #       -> parameters/origination.yml
    #
    #   modelling_approach: behavioral
    #       -> parameters/behavioral.yml
    # --------------------------------------------------------------

    approach_parameter_path = parameters_dir / f"{approach}.yml"

    if not approach_parameter_path.exists():
        raise FileNotFoundError(
            "Approach-specific parameter file does not exist: "
            f"{approach_parameter_path}"
        )

    approach_parameters = _load_yaml(
        approach_parameter_path,
    )

    # --------------------------------------------------------------
    # Merge base + approach-specific parameters.
    # --------------------------------------------------------------

    parameters = _deep_merge(
        base_parameters,
        approach_parameters,
    )

    # --------------------------------------------------------------
    # Catalog remains completely unchanged.
    # --------------------------------------------------------------

    catalog = _load_yaml(
        catalog_path,
    )

    # --------------------------------------------------------------
    # Return the exact same top-level config structure as before.
    # --------------------------------------------------------------

    config = {
        **parameters,
        **catalog,
    }

    return config


def create_path(
    base_path: str | Path,
    catalog: dict,
    key: str,
    *subfolders: str | int,
    must_exist: bool = True,
) -> Path:
    """
    Construct a path from the configured catalog entry.

    Parameters
    ----------
    base_path
        Root directory.
    catalog
        Catalog configuration.
    key
        Catalog key.
    *subfolders
        Optional hierarchy appended below the configured folder.
    must_exist
        If False, parent directories are created as required.
    """

    if key not in catalog:
        raise KeyError(f"Catalog key not found: {key}")

    file_config = catalog[key]

    path = Path(base_path) / file_config["folder_name"]

    for folder in subfolders:
        path /= str(folder)

    if not must_exist:
        path.mkdir(
            parents=True,
            exist_ok=True,
        )

    if file_config["file_type"] == "folder":
        file_path = path / file_config["file_name"]

        if not must_exist:
            file_path.mkdir(
                parents=True,
                exist_ok=True,
            )
    else:
        file_path = path / f"{file_config['file_name']}.{file_config['file_type']}"

    if must_exist and not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    return file_path
