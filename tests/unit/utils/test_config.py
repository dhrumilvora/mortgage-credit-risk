from credit_risk.utils.config import create_path


def test_create_path_scopes_model_input_by_provider_and_vintage(tmp_path):
    catalog = {
        "model_input_path": {
            "folder_name": "03_processed",
            "file_name": "model-input",
            "file_type": "parquet",
        }
    }

    path = create_path(
        tmp_path,
        catalog,
        "model_input_path",
        "freddie_mac",
        2015,
        must_exist=False,
    )

    assert (
        path
        == tmp_path / "03_processed" / "freddie_mac" / "2015" / "model-input.parquet"
    )
