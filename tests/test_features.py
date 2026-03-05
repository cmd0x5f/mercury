"""Tests for feature engineering modules."""

import pandas as pd
import pytest

from src.features.builder import FEATURE_COLS, build_features, get_feature_matrix
from src.features.context import compute_context_features
from src.features.form import compute_rolling_margins
from src.features.team_strength import compute_elo_ratings, expected_score


@pytest.fixture
def games():
    """Small set of games for testing features."""
    return pd.DataFrame([
        {"date": "2024-01-01", "home_team": "A", "away_team": "B",
         "home_score": 110, "away_score": 100},
        {"date": "2024-01-03", "home_team": "C", "away_team": "A",
         "home_score": 105, "away_score": 108},
        {"date": "2024-01-04", "home_team": "B", "away_team": "C",
         "home_score": 95, "away_score": 102},
        {"date": "2024-01-06", "home_team": "A", "away_team": "C",
         "home_score": 115, "away_score": 99},
        {"date": "2024-01-07", "home_team": "B", "away_team": "A",
         "home_score": 100, "away_score": 112},
    ])


class TestExpectedScore:
    def test_equal_ratings(self):
        assert expected_score(1500, 1500) == pytest.approx(0.5)

    def test_stronger_team(self):
        assert expected_score(1600, 1400) > 0.5

    def test_weaker_team(self):
        assert expected_score(1400, 1600) < 0.5

    def test_symmetry(self):
        p = expected_score(1600, 1400)
        assert p + expected_score(1400, 1600) == pytest.approx(1.0)


class TestEloRatings:
    def test_adds_elo_columns(self, games):
        result = compute_elo_ratings(games)
        assert "home_elo" in result.columns
        assert "away_elo" in result.columns
        assert "elo_diff" in result.columns

    def test_initial_ratings(self, games):
        result = compute_elo_ratings(games, start_rating=1500)
        # First game: both teams start at 1500
        assert result.iloc[0]["home_elo"] == 1500
        assert result.iloc[0]["away_elo"] == 1500

    def test_winner_elo_increases(self, games):
        result = compute_elo_ratings(games)
        # Team A wins game 1 (home 110-100), so A's rating should increase
        # In game 2, A plays away — check A's away_elo > 1500
        game2 = result.iloc[1]
        assert game2["away_elo"] > 1500  # A's rating went up after winning

    def test_elo_diff_includes_home_advantage(self, games):
        result = compute_elo_ratings(games, home_advantage=100)
        row = result.iloc[0]
        assert row["elo_diff"] == row["home_elo"] + 100 - row["away_elo"]


class TestRollingMargins:
    def test_adds_rolling_columns(self, games):
        result = compute_rolling_margins(games)
        assert "home_avg_margin" in result.columns
        assert "away_avg_margin" in result.columns
        assert "home_avg_scored" in result.columns
        assert "away_avg_scored" in result.columns

    def test_defaults_for_first_game(self, games):
        result = compute_rolling_margins(games)
        # First game for teams A and B — should get defaults
        assert result.iloc[0]["home_avg_margin"] == 10.0
        assert result.iloc[0]["away_avg_margin"] == 10.0

    def test_rolling_updates_after_games(self, games):
        result = compute_rolling_margins(games)
        # After game 1 (A vs B, margin 10), team A's next appearance
        # should reflect that 10-point margin
        game2 = result.iloc[1]  # C vs A
        assert game2["away_avg_margin"] == 10.0  # A's avg margin from game 1


class TestContextFeatures:
    def test_adds_context_columns(self, games):
        result = compute_context_features(games)
        for col in ["home_rest_days", "away_rest_days",
                     "home_game_num", "away_game_num",
                     "is_b2b_home", "is_b2b_away"]:
            assert col in result.columns

    def test_rest_days_default_for_opener(self, games):
        result = compute_context_features(games)
        assert result.iloc[0]["home_rest_days"] == 7  # season opener default

    def test_rest_days_calculated(self, games):
        result = compute_context_features(games)
        # Team A plays Jan 1, then Jan 3 = 2 rest days
        game2 = result.iloc[1]  # Jan 3, C vs A
        assert game2["away_rest_days"] == 2

    def test_back_to_back_detection(self, games):
        result = compute_context_features(games)
        # Team A plays Jan 6 and Jan 7 = 1 day rest = back to back
        game5 = result.iloc[4]  # Jan 7, B vs A
        assert game5["away_rest_days"] == 1
        assert game5["is_b2b_away"] == 1


class TestFeatureBuilder:
    def test_build_features_returns_all_columns(self, games):
        result = build_features(games)
        for col in FEATURE_COLS:
            assert col in result.columns, f"Missing column: {col}"
        assert "margin" in result.columns

    def test_get_feature_matrix_shapes(self, games):
        featured = build_features(games)
        X, y = get_feature_matrix(featured)
        assert X.shape == (len(games), len(FEATURE_COLS))
        assert len(y) == len(games)

    def test_margin_is_signed(self, games):
        featured = build_features(games)
        # Game 2: C(105) vs A(108), margin = 105 - 108 = -3
        assert featured.iloc[1]["margin"] == -3
