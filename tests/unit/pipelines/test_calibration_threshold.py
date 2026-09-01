import numpy as np

from credit_risk.evaluations.calibration import fit_calibration_isotonic
from credit_risk.pipelines.evaluation import _map_threshold_to_calibrated_scale


def test_calibrated_threshold_is_mapped_from_raw_probability_scale() -> None:
    """A raw cutoff cannot be reused after probability calibration."""

    calibrator = fit_calibration_isotonic(
        y_true=np.array([0, 0, 1, 1]),
        y_proba=np.array([0.1, 0.2, 0.3, 0.4]),
    )
    config = {"parameters": {"evaluation": {"calibration": {"method": "isotonic"}}}}

    calibrated_threshold = _map_threshold_to_calibrated_scale(
        raw_threshold=0.3,
        calibration_model=calibrator,
        config=config,
    )

    assert calibrated_threshold == 1.0
    assert calibrated_threshold != 0.3
