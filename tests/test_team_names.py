"""Tests for team name mapping."""

from src.data.team_names import NBA_TO_SPORTSPLUS, SPORTSPLUS_TO_NBA, normalize_team


class TestTeamNames:
    def test_all_30_nba_teams_mapped(self):
        assert len(SPORTSPLUS_TO_NBA) == 30

    def test_reverse_mapping_complete(self):
        assert len(NBA_TO_SPORTSPLUS) == 30

    def test_normalize_known_team(self):
        assert normalize_team("Los Angeles Lakers") == "LAL"
        assert normalize_team("Golden State Warriors") == "GSW"
        assert normalize_team("Denver Nuggets") == "DEN"

    def test_normalize_non_nba_returns_none(self):
        assert normalize_team("Fenerbahce") is None
        assert normalize_team("Real Madrid Baloncesto") is None
        assert normalize_team("Iowa Hawkeyes") is None

    def test_roundtrip(self):
        for full_name, abbr in SPORTSPLUS_TO_NBA.items():
            assert NBA_TO_SPORTSPLUS[abbr] == full_name
