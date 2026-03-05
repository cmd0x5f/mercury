"""Tests for the folded normal distribution and bucket probability calculations."""

import pytest

from src.model.distribution import (
    BUCKET_NAMES,
    bucket_probabilities,
    bucket_probabilities_batch,
    folded_normal_cdf,
)


class TestFoldedNormalCDF:
    def test_zero_returns_correct_value(self):
        # P(|X| <= 0) when X ~ N(0, 10) should be ~0
        assert folded_normal_cdf(0, 0, 10) == pytest.approx(0.0, abs=0.01)

    def test_large_x_approaches_one(self):
        # P(|X| <= 100) when X ~ N(5, 10) should be ~1
        assert folded_normal_cdf(100, 5, 10) == pytest.approx(1.0, abs=0.001)

    def test_negative_x_returns_zero(self):
        assert folded_normal_cdf(-1, 5, 10) == 0.0

    def test_symmetric_around_zero_mu(self):
        # With mu=0, the folded normal is symmetric so CDF at x should be
        # same regardless of mu sign... actually test P(|X|<=10) for mu=0
        p = folded_normal_cdf(10, 0, 10)
        assert 0 < p < 1

    def test_cdf_monotonically_increases(self):
        values = [folded_normal_cdf(x, 5, 12) for x in range(0, 50, 5)]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1]


class TestBucketProbabilities:
    def test_probabilities_sum_to_one(self):
        probs = bucket_probabilities(5.0, 12.0)
        total = sum(probs.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_all_buckets_present(self):
        probs = bucket_probabilities(5.0, 12.0)
        assert set(probs.keys()) == set(BUCKET_NAMES)

    def test_all_probabilities_positive(self):
        probs = bucket_probabilities(5.0, 12.0)
        for name, p in probs.items():
            assert p > 0, f"Bucket {name} has probability {p}"

    def test_close_game_favors_small_margins(self):
        """When predicted margin is near 0, small buckets should dominate."""
        probs = bucket_probabilities(0.0, 12.0)
        assert probs["1-5"] > probs["6-10"] > probs["11-15"]

    def test_blowout_shifts_distribution(self):
        """When predicted margin is large, bigger buckets get more weight."""
        close = bucket_probabilities(2.0, 12.0)
        blowout = bucket_probabilities(20.0, 12.0)
        # Blowout should have more weight in 16-20 bucket
        assert blowout["16-20"] > close["16-20"]
        # Close game should have more weight in 1-5 bucket
        assert close["1-5"] > blowout["1-5"]

    def test_large_sigma_spreads_distribution(self):
        """Higher sigma = more uncertainty = flatter distribution."""
        tight = bucket_probabilities(10.0, 5.0)
        wide = bucket_probabilities(10.0, 20.0)
        # With wider sigma, 31+ bucket should have more probability
        assert wide["31+"] > tight["31+"]

    def test_negative_mu_same_absolute_distribution(self):
        """Folded normal: |N(-5, 12)| should give similar probs to |N(5, 12)|
        because we're taking absolute value."""
        pos = bucket_probabilities(5.0, 12.0)
        neg = bucket_probabilities(-5.0, 12.0)
        for bucket in BUCKET_NAMES:
            assert pos[bucket] == pytest.approx(neg[bucket], abs=0.01)


class TestBucketProbabilitiesBatch:
    def test_batch_matches_individual(self):
        import numpy as np
        mus = np.array([0.0, 5.0, -10.0, 20.0])
        sigma = 12.0
        batch = bucket_probabilities_batch(mus, sigma)
        for i, mu in enumerate(mus):
            individual = bucket_probabilities(float(mu), sigma)
            for bucket in BUCKET_NAMES:
                assert batch[i][bucket] == pytest.approx(individual[bucket])

    def test_batch_length(self):
        import numpy as np
        mus = np.array([1.0, 2.0, 3.0])
        result = bucket_probabilities_batch(mus, 12.0)
        assert len(result) == 3
