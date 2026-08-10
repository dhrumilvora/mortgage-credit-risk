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


def read_config(project_path: Path) -> dict:
    parameter_path = project_path / "config" / "parameters" / "base.yml"
    catalog_path = project_path / "config" / "catalog" / "base.yml"

    parameters = _load_yaml(parameter_path)
    catalog = _load_yaml(catalog_path)

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
