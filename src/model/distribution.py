"""Folded normal distribution for winning margin bucket probabilities."""

import numpy as np
from scipy.stats import norm

# Standard "Any Team Winning Margin" buckets
BUCKETS = [
    ("1-5", 1, 5),
    ("6-10", 6, 10),
    ("11-15", 11, 15),
    ("16-20", 16, 20),
    ("21-25", 21, 25),
    ("26-30", 26, 30),
    ("31+", 31, None),
]

BUCKET_NAMES = [b[0] for b in BUCKETS]


def folded_normal_cdf(x: float, mu: float, sigma: float) -> float:
    """CDF of the folded normal distribution |N(mu, sigma)| at point x.

    P(|X| <= x) = Φ((x - mu)/σ) - Φ((-x - mu)/σ)  for x >= 0
    """
    if x < 0:
        return 0.0
    return norm.cdf((x - mu) / sigma) - norm.cdf((-x - mu) / sigma)


def bucket_probabilities(mu: float, sigma: float) -> dict[str, float]:
    """Compute probability for each winning margin bucket.

    Args:
        mu: predicted signed margin (home - away)
        sigma: standard deviation of prediction residuals

    Returns:
        dict mapping bucket name -> probability
    """
    probs = {}
    for name, low, high in BUCKETS:
        if high is not None:
            # P(low <= |X| <= high) = P(|X| <= high+0.5) - P(|X| <= low-0.5)
            # Using 0.5 continuity correction since scores are integers
            p = folded_normal_cdf(high + 0.5, mu, sigma) - folded_normal_cdf(low - 0.5, mu, sigma)
        else:
            # 31+: P(|X| >= 30.5) = 1 - P(|X| <= 30.5)
            p = 1.0 - folded_normal_cdf(low - 0.5, mu, sigma)
        probs[name] = max(p, 1e-6)  # floor to avoid log(0)

    # Normalize to sum to 1 (exclude exact 0 margin which is impossible in basketball)
    total = sum(probs.values())
    return {k: v / total for k, v in probs.items()}


def bucket_probabilities_batch(
    mus: np.ndarray, sigma: float
) -> list[dict[str, float]]:
    """Compute bucket probabilities for multiple games."""
    return [bucket_probabilities(float(mu), sigma) for mu in mus]
