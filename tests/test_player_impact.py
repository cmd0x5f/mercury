"""Tests for player availability impact features."""

import pandas as pd
import pytest

from src.features.player_impact import (
    compute_player_impact_scores,
    compute_missing_impact,
    _calc_missing,
    _update_roster,
)


@pytest.fixture
def player_logs():
    """Simulate player game logs across multiple games."""
    rows = []
    # 3 players on team A, 2 on team B, across 10 games each
    games_a = [f"G{i:03d}" for i in range(1, 11)]
    games_b = [f"G{i:03d}" for i in range(1, 11)]

    for i, gid in enumerate(games_a):
        date = f"2024-01-{i+1:02d}"
        # Player 1: star (36 min, 25 pts, +8)
        rows.append({
            "player_id": 1, "player_name": "Star A",
            "team": "A", "game_id": gid, "date": date,
            "minutes": 36.0, "points": 25, "plus_minus": 8.0,
        })
        # Player 2: starter (28 min, 14 pts, +3)
        rows.append({
            "player_id": 2, "player_name": "Starter A",
            "team": "A", "game_id": gid, "date": date,
            "minutes": 28.0, "points": 14, "plus_minus": 3.0,
        })
        # Player 3: bench (12 min, 5 pts, +1)
        rows.append({
            "player_id": 3, "player_name": "Bench A",
            "team": "A", "game_id": gid, "date": date,
            "minutes": 12.0, "points": 5, "plus_minus": 1.0,
        })

    for i, gid in enumerate(games_b):
        date = f"2024-01-{i+1:02d}"
        rows.append({
            "player_id": 10, "player_name": "Star B",
            "team": "B", "game_id": gid, "date": date,
            "minutes": 34.0, "points": 22, "plus_minus": 5.0,
        })
        rows.append({
            "player_id": 11, "player_name": "Role B",
            "team": "B", "game_id": gid, "date": date,
            "minutes": 20.0, "points": 8, "plus_minus": 0.0,
        })

    return pd.DataFrame(rows)


@pytest.fixture
def team_games():
    """Games between team A and team B."""
    return pd.DataFrame([
        {"game_id": f"G{i:03d}", "date": f"2024-01-{i:02d}",
         "home_team": "A", "away_team": "B"}
        for i in range(1, 11)
    ])


class TestComputePlayerImpactScores:
    def test_returns_expected_columns(self, player_logs):
        result = compute_player_impact_scores(player_logs)
        for col in ["player_id", "player_name", "team", "game_id", "date", "impact_score"]:
            assert col in result.columns

    def test_no_impact_for_early_games(self, player_logs):
        """Players need MIN_PLAYER_GAMES (5) before getting a score."""
        result = compute_player_impact_scores(player_logs)
        early = result[(result["player_id"] == 1) & (result["date"] <= "2024-01-05")]
        assert early["impact_score"].isna().all()

    def test_impact_after_enough_games(self, player_logs):
        """After 5+ games, impact score should be computed."""
        result = compute_player_impact_scores(player_logs)
        later = result[(result["player_id"] == 1) & (result["date"] > "2024-01-05")]
        assert later["impact_score"].notna().all()

    def test_star_has_higher_impact_than_bench(self, player_logs):
        """Star player should have higher impact than bench player."""
        result = compute_player_impact_scores(player_logs)
        later = result[result["date"] > "2024-01-05"]
        star = later[later["player_id"] == 1]["impact_score"].mean()
        bench = later[later["player_id"] == 3]["impact_score"].mean()
        assert star > bench

    def test_impact_score_positive(self, player_logs):
        """Impact scores should be positive for normal stat lines."""
        result = compute_player_impact_scores(player_logs)
        valid = result["impact_score"].dropna()
        assert (valid > 0).all()


class TestComputeMissingImpact:
    def test_returns_expected_columns(self, team_games, player_logs):
        impacts = compute_player_impact_scores(player_logs)
        result = compute_missing_impact(team_games, impacts)
        assert "game_id" in result.columns
        assert "home_missing_impact" in result.columns
        assert "away_missing_impact" in result.columns

    def test_no_missing_when_all_play(self, team_games, player_logs):
        """When all roster players play, missing impact should be 0."""
        impacts = compute_player_impact_scores(player_logs)
        result = compute_missing_impact(team_games, impacts)
        # All players play every game, so after the warmup period, missing should be 0
        late_games = result[result["game_id"] >= "G006"]
        assert (late_games["home_missing_impact"] == 0.0).all()
        assert (late_games["away_missing_impact"] == 0.0).all()

    def test_missing_impact_when_player_absent(self, team_games, player_logs):
        """Remove a star player from a late game — missing impact should increase."""
        # Drop Star A (player_id=1) from game G008
        logs_modified = player_logs[
            ~((player_logs["player_id"] == 1) & (player_logs["game_id"] == "G008"))
        ].copy()
        impacts = compute_player_impact_scores(logs_modified)
        result = compute_missing_impact(team_games, impacts)
        g8 = result[result["game_id"] == "G008"].iloc[0]
        # Home team A is missing their star, so home_missing_impact > 0
        assert g8["home_missing_impact"] > 0

    def test_early_games_zero_missing(self, team_games, player_logs):
        """First few games should have 0 missing impact (not enough roster history)."""
        impacts = compute_player_impact_scores(player_logs)
        result = compute_missing_impact(team_games, impacts)
        early = result[result["game_id"] <= "G005"]
        assert (early["home_missing_impact"] == 0.0).all()


class TestCalcMissing:
    def test_returns_zero_when_not_enough_history(self):
        assert _calc_missing({}, {}, team_game_count=2) == 0.0

    def test_returns_zero_when_all_played(self):
        roster = {
            1: {"impact_score": 0.7, "games_ago": 1},
            2: {"impact_score": 0.3, "games_ago": 2},
        }
        played = {1: 0.7, 2: 0.3}
        assert _calc_missing(roster, played, team_game_count=10) == 0.0

    def test_sums_missing_player_impact(self):
        roster = {
            1: {"impact_score": 0.7, "games_ago": 1},
            2: {"impact_score": 0.3, "games_ago": 2},
        }
        played = {2: 0.3}  # Player 1 missing
        result = _calc_missing(roster, played, team_game_count=10)
        assert result == pytest.approx(0.7)

    def test_ignores_stale_roster_players(self):
        """Players with games_ago > 10 shouldn't count as missing."""
        roster = {
            1: {"impact_score": 0.7, "games_ago": 11},  # stale
        }
        played = {}
        assert _calc_missing(roster, played, team_game_count=10) == 0.0


class TestUpdateRoster:
    def test_adds_new_players(self):
        rosters = {}
        counts = {}
        _update_roster(rosters, "A", {1: 0.5, 2: 0.3}, counts, 20)
        assert 1 in rosters["A"]
        assert rosters["A"][1]["games_ago"] == 0

    def test_ages_players(self):
        rosters = {"A": {1: {"impact_score": 0.5, "games_ago": 0}}}
        counts = {"A": 1}
        _update_roster(rosters, "A", {}, counts, 20)
        assert rosters["A"][1]["games_ago"] == 1

    def test_removes_stale_players(self):
        rosters = {"A": {1: {"impact_score": 0.5, "games_ago": 20}}}
        counts = {"A": 10}
        _update_roster(rosters, "A", {}, counts, 20)
        # games_ago becomes 21, which is > roster_window (20), so removed
        assert 1 not in rosters["A"]
