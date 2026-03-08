"""Tests for the margin model — training and prediction."""

import numpy as np
import pandas as pd
import pytest

from src.model.distribution import BUCKET_NAMES
from src.model.margin_model import MarginModel


@pytest.fixture
def synthetic_games():
    """Generate synthetic NBA-like game data for testing.

    Creates 500 games with realistic-ish scores where the home team
    has a slight advantage.
    """
    rng = np.random.RandomState(42)
    n = 500
    teams = ["A", "B", "C", "D", "E", "F"]

    games = []
    for i in range(n):
        h, a = rng.choice(teams, size=2, replace=False)
        # Home team gets ~3 point advantage on average
        base = 105
        home_score = int(base + rng.normal(3, 12))
        away_score = int(base + rng.normal(0, 12))
        games.append({
            "date": f"2024-{1 + i // 200:02d}-{1 + (i % 28):02d}",
            "home_team": h,
            "away_team": a,
            "home_score": max(70, home_score),
            "away_score": max(70, away_score),
        })

    return pd.DataFrame(games)


class TestMarginModel:
    def test_train_sets_model_and_sigma(self, synthetic_games):
        model = MarginModel()
        model.train(synthetic_games)
        # With per-league models, check that at least one league model exists
        assert len(model.league_models) > 0 or model.fallback is not None
        assert model.sigma > 0
        assert model.sigma < 30  # reasonable range

    def test_predict_margin_returns_array(self, synthetic_games):
        from src.features.builder import build_features
        model = MarginModel()
        model.train(synthetic_games)

        featured = build_features(synthetic_games)
        preds = model.predict_margin(featured)
        assert len(preds) == len(featured)
        # Predictions should be in a reasonable range
        assert all(-50 < p < 50 for p in preds)

    def test_predict_buckets_valid_distributions(self, synthetic_games):
        from src.features.builder import build_features
        model = MarginModel()
        model.train(synthetic_games)

        featured = build_features(synthetic_games)
        buckets = model.predict_buckets(featured.head(10))
        assert len(buckets) == 10

        for probs in buckets:
            assert set(probs.keys()) == set(BUCKET_NAMES)
            total = sum(probs.values())
            assert total == pytest.approx(1.0, abs=1e-4)
            assert all(p > 0 for p in probs.values())

    def test_predict_single(self, synthetic_games):
        from src.features.builder import FEATURE_COLS, build_features
        model = MarginModel()
        model.train(synthetic_games)

        featured = build_features(synthetic_games)
        row = featured.iloc[-1]
        features = {col: row[col] for col in FEATURE_COLS}

        probs = model.predict_single(features)
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)

    def test_save_and_load(self, synthetic_games, tmp_path):
        model = MarginModel()
        model.train(synthetic_games)
        original_sigma = model.sigma

        path = tmp_path / "model.pkl"
        model.save(path)

        loaded = MarginModel()
        loaded.load(path)
        assert loaded.sigma == pytest.approx(original_sigma)
        assert len(loaded.league_models) > 0 or loaded.fallback is not None

    def test_evaluate_returns_metrics(self, synthetic_games):
        model = MarginModel()
        results = model.evaluate(synthetic_games)

        assert "mae" in results
        assert "rmse" in results
        assert "sigma" in results
        assert "bucket_accuracy" in results
        assert results["mae"] > 0
        assert results["rmse"] >= results["mae"]  # RMSE >= MAE always

    @pytest.mark.parametrize("backend_name", ["xgboost", "ridge", "rf"])
    def test_train_with_different_backends(self, synthetic_games, backend_name):
        model = MarginModel(backend_name=backend_name)
        model.train(synthetic_games)
        assert len(model.league_models) > 0 or model.fallback is not None
        assert model.sigma > 0

    def test_save_load_roundtrip_ridge(self, synthetic_games, tmp_path):
        model = MarginModel(backend_name="ridge")
        model.train(synthetic_games)
        original_sigma = model.sigma

        path = tmp_path / "model_ridge.pkl"
        model.save(path)

        loaded = MarginModel()
        loaded.load(path)
        assert loaded.backend_name == "ridge"
        assert loaded.sigma == pytest.approx(original_sigma)
        assert len(loaded.league_models) > 0 or loaded.fallback is not None
