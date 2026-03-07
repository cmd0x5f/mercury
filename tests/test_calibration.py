"""Tests for Platt scaling calibration."""

import pytest

from src.model.calibration import PlattCalibrator


@pytest.fixture
def calibrator_with_data():
    """Fit a calibrator on synthetic prediction data."""
    cal = PlattCalibrator()

    # Simulate 200 predictions with 3 buckets
    raw_probs = []
    actual_buckets = []
    for i in range(200):
        # Bucket "A" is common, "B" medium, "C" rare
        if i % 5 == 0:
            actual = "C"
        elif i % 3 == 0:
            actual = "B"
        else:
            actual = "A"

        # Raw model gives noisy probabilities
        raw_probs.append({"A": 0.6 + (i % 7) * 0.02,
                          "B": 0.25 - (i % 5) * 0.01,
                          "C": 0.15 - (i % 3) * 0.01})
        actual_buckets.append(actual)

    cal.fit(raw_probs, actual_buckets)
    return cal


class TestPlattCalibrator:
    def test_fit_creates_calibrators(self, calibrator_with_data):
        # Should have fitted at least some buckets (those with enough samples)
        assert len(calibrator_with_data.calibrators) > 0

    def test_calibrate_sums_to_one(self, calibrator_with_data):
        raw = {"A": 0.6, "B": 0.25, "C": 0.15}
        calibrated = calibrator_with_data.calibrate(raw)
        assert sum(calibrated.values()) == pytest.approx(1.0, abs=1e-6)

    def test_calibrate_all_positive(self, calibrator_with_data):
        raw = {"A": 0.6, "B": 0.25, "C": 0.15}
        calibrated = calibrator_with_data.calibrate(raw)
        for bucket, prob in calibrated.items():
            assert prob > 0, f"Bucket {bucket} has non-positive probability"

    def test_calibrate_preserves_buckets(self, calibrator_with_data):
        raw = {"A": 0.6, "B": 0.25, "C": 0.15}
        calibrated = calibrator_with_data.calibrate(raw)
        assert set(calibrated.keys()) == set(raw.keys())

    def test_uncalibrated_buckets_pass_through(self):
        """Buckets without enough data should keep raw probabilities."""
        cal = PlattCalibrator()
        # Fit with too few samples for bucket "rare"
        raw_probs = [{"common": 0.9, "rare": 0.1}] * 20
        actuals = ["common"] * 20  # "rare" never hits -> not enough data
        cal.fit(raw_probs, actuals)

        result = cal.calibrate({"common": 0.9, "rare": 0.1})
        # "rare" had 0 positive samples, so no calibrator fitted
        assert "rare" in result
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-6)

    def test_empty_calibrator_is_identity(self):
        """With no fitted calibrators, calibrate returns normalized input."""
        cal = PlattCalibrator()
        raw = {"A": 0.6, "B": 0.3, "C": 0.1}
        result = cal.calibrate(raw)
        assert result == pytest.approx(raw, abs=1e-6)
