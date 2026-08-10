"""Tests for modelling artifact persistence."""

from pathlib import Path

import pytest

from credit_risk.modelling.artifacts import (
    _load_joblib,
    _load_json,
    _load_yaml,
    _save_joblib,
    _save_json,
    _save_yaml,
)


def test_save_and_load_joblib(tmp_path: Path) -> None:
    """Joblib artifacts should round-trip correctly."""

    path = tmp_path / "artifact.joblib"

    artifact = {
        "model": "logistic_regression",
        "value": 42,
    }

    _save_joblib(
        artifact,
        path,
    )

    loaded = _load_joblib(path)

    assert loaded == artifact


def test_save_and_load_json(tmp_path: Path) -> None:
    """JSON artifacts should round-trip correctly."""

    path = tmp_path / "artifact.json"

    artifact = {
        "training_rows": 1000,
        "event_rate": 0.05,
        "model_type": "logistic_regression",
    }

    _save_json(
        artifact,
        path,
    )

    loaded = _load_json(path)

    assert loaded == artifact


def test_save_and_load_yaml(tmp_path: Path) -> None:
    """YAML artifacts should round-trip correctly."""

    path = tmp_path / "artifact.yaml"

    artifact = {
        "model_type": "logistic_regression",
        "random_state": 42,
        "max_iter": 1000,
    }

    _save_yaml(
        artifact,
        path,
    )

    loaded = _load_yaml(path)

    assert loaded == artifact
