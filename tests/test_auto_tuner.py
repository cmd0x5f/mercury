"""Tests for auto-tune pipeline: per-league backend selection."""

import numpy as np
import pandas as pd
import pytest

from src.model.auto_tuner import LeagueConfig, auto_tune
from src.model.margin_model import MarginModel


def _make_fake_games(n_nba=600, n_euro=200, seed=42):
    """Generate fake game data for two leagues."""
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n_nba):
        rows.append({
            "game_id": f"nba_{i}",
            "date": f"2024-{1 + i % 12:02d}-{1 + i % 28:02d}",
            "home_team": f"Home{i % 5}",
            "away_team": f"Away{i % 5}",
            "home_score": int(100 + rng.randn() * 12),
            "away_score": int(98 + rng.randn() * 12),
            "league": "NBA",
        })
    for i in range(n_euro):
        rows.append({
            "game_id": f"euro_{i}",
            "date": f"2024-{1 + i % 12:02d}-{1 + i % 28:02d}",
            "home_team": f"EuroHome{i % 4}",
            "away_team": f"EuroAway{i % 4}",
            "home_score": int(80 + rng.randn() * 10),
            "away_score": int(78 + rng.randn() * 10),
            "league": "Euroleague",
        })
    return pd.DataFrame(rows)


class TestLeagueConfig:
    def test_defaults(self):
        cfg = LeagueConfig(backend_name="xgboost")
        assert cfg.backend_name == "xgboost"
        assert cfg.backend_params == {}
        assert cfg.cv_score == 0.0

    def test_with_params(self):
        cfg = LeagueConfig(
            backend_name="ridge",
            backend_params={"alpha": 5.0},
            cv_score=11.5,
            tuned_score=11.2,
        )
        assert cfg.backend_params["alpha"] == 5.0
        assert cfg.tuned_score < cfg.cv_score


class TestMarginModelPerLeagueConfigs:
    def test_league_configs_accepted(self):
        """MarginModel accepts league_configs parameter."""
        configs = {
            "NBA": {"backend_name": "xgboost", "backend_params": {}},
            "Euroleague": {"backend_name": "ridge", "backend_params": {"alpha": 1.0}},
        }
        model = MarginModel(league_configs=configs)
        assert model.league_configs == configs

    def test_per_league_backend_selection(self):
        """Different leagues use different backends when league_configs is set."""
        games = _make_fake_games(n_nba=600, n_euro=0)
        configs = {
            "NBA": {"backend_name": "ridge", "backend_params": {}},
        }
        model = MarginModel(league_configs=configs)
        model.train(games)
        # NBA model should use Ridge since we configured it
        assert model.league_models["NBA"].backend.name() == "ridge"

    def test_fallback_uses_default_when_no_config(self):
        """Without league_configs, all leagues use the default backend."""
        games = _make_fake_games(n_nba=600, n_euro=0)
        model = MarginModel(backend_name="xgboost")
        model.train(games)
        assert model.league_models["NBA"].backend.name() == "xgboost"

    def test_save_load_v3_roundtrip(self, tmp_path):
        """v3 format preserves per-league backend names."""
        games = _make_fake_games(n_nba=600, n_euro=0)
        configs = {
            "NBA": {"backend_name": "ridge", "backend_params": {}},
        }
        model = MarginModel(league_configs=configs)
        model.train(games)

        path = tmp_path / "model.pkl"
        model.save(path)

        loaded = MarginModel()
        loaded.load(path)
        assert loaded.league_models["NBA"].backend.name() == "ridge"

    def test_save_load_v3_with_mixed_backends(self, tmp_path):
        """v3 handles different backends for different leagues."""
        games = _make_fake_games(n_nba=600, n_euro=600)
        configs = {
            "NBA": {"backend_name": "xgboost", "backend_params": {}},
            "Euroleague": {"backend_name": "ridge", "backend_params": {}},
        }
        model = MarginModel(league_configs=configs)
        model.train(games)

        assert model.league_models["NBA"].backend.name() == "xgboost"
        assert model.league_models["Euroleague"].backend.name() == "ridge"

        path = tmp_path / "model.pkl"
        model.save(path)

        loaded = MarginModel()
        loaded.load(path)
        assert loaded.league_models["NBA"].backend.name() == "xgboost"
        assert loaded.league_models["Euroleague"].backend.name() == "ridge"

        # Predictions should work
        from src.features.builder import FEATURE_COLS
        dummy_features = pd.DataFrame(
            {col: [0.0] for col in FEATURE_COLS}
        )
        for league in ["NBA", "Euroleague"]:
            probs = loaded.predict_buckets(dummy_features, league_name=league)
            assert len(probs) == 1
            assert abs(sum(probs[0].values()) - 1.0) < 0.01


class TestAutoTuneNoTune:
    """Test auto-tune with tune_trials=0 (comparison only, fast)."""

    def test_returns_configs_for_eligible_leagues(self):
        games = _make_fake_games(n_nba=600, n_euro=200)
        configs = auto_tune(
            games,
            tune_trials=0,
            backends_to_try=["xgboost", "ridge"],
        )
        # NBA has 600 games (>= MIN_STANDALONE_GAMES), so it's eligible
        assert "NBA" in configs
        assert configs["NBA"].backend_name in ("xgboost", "ridge")
        assert configs["NBA"].cv_score > 0

    def test_fallback_for_small_leagues(self):
        games = _make_fake_games(n_nba=600, n_euro=200)
        configs = auto_tune(
            games,
            tune_trials=0,
            backends_to_try=["xgboost", "ridge"],
        )
        # Euroleague has 200 games (< MIN_STANDALONE_GAMES), goes to fallback
        assert "__fallback__" in configs

    def test_single_league(self):
        games = _make_fake_games(n_nba=600, n_euro=0)
        configs = auto_tune(
            games,
            tune_trials=0,
            backends_to_try=["ridge"],
        )
        assert configs["NBA"].backend_name == "ridge"
