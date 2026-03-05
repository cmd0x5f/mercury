"""Probability calibration via Platt scaling (Phase 3)."""

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """Calibrates raw model probabilities using Platt scaling.

    For each bucket, fits a logistic regression: P(hit) = sigmoid(a * raw_prob + b)
    This corrects systematic over/under-confidence.
    """

    def __init__(self):
        self.calibrators: dict[str, LogisticRegression] = {}

    def fit(self, raw_probs: list[dict[str, float]], actual_buckets: list[str]):
        """Fit calibration on historical predictions vs outcomes.

        Args:
            raw_probs: list of {bucket: probability} dicts from model
            actual_buckets: list of actual outcome bucket names
        """
        buckets = list(raw_probs[0].keys())
        for bucket in buckets:
            X = np.array([p[bucket] for p in raw_probs]).reshape(-1, 1)
            y = np.array([1 if ab == bucket else 0 for ab in actual_buckets])

            if y.sum() < 5 or (1 - y).sum() < 5:
                continue  # not enough data for this bucket

            lr = LogisticRegression()
            lr.fit(X, y)
            self.calibrators[bucket] = lr

    def calibrate(self, raw_probs: dict[str, float]) -> dict[str, float]:
        """Calibrate a single prediction's bucket probabilities."""
        calibrated = {}
        for bucket, prob in raw_probs.items():
            if bucket in self.calibrators:
                calibrated[bucket] = float(
                    self.calibrators[bucket].predict_proba(
                        np.array([[prob]])
                    )[0, 1]
                )
            else:
                calibrated[bucket] = prob

        # Re-normalize
        total = sum(calibrated.values())
        return {k: v / total for k, v in calibrated.items()}
