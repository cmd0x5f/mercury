"""Tests for league and team name matching."""

from pathlib import Path

import pytest

from src.data.league_matcher import LeagueMatcher, TeamMatcher, load_league_config


CONFIG_PATH = Path(__file__).parents[1] / "config" / "league_mappings.yaml"


@pytest.fixture
def matcher():
    return LeagueMatcher(config_path=CONFIG_PATH)


@pytest.fixture
def team_matcher():
    tm = TeamMatcher()
    tm.register_teams("NBA", ["Los Angeles Lakers", "Boston Celtics", "Golden State Warriors"])
    tm.register_teams("Euroleague", ["Barcelona", "Real Madrid", "Olympiacos", "Fenerbahce"])
    return tm


# --- LeagueMatcher ---


class TestLeagueMatcher:
    def test_exact_match_canonical(self, matcher):
        assert matcher.match_league("NBA") == "NBA"
        assert matcher.match_league("Euroleague") == "Euroleague"
        assert matcher.match_league("Spain ACB") == "Spain ACB"

    def test_exact_match_case_insensitive(self, matcher):
        assert matcher.match_league("nba") == "NBA"
        assert matcher.match_league("euroleague") == "Euroleague"
        assert matcher.match_league("spain acb") == "Spain ACB"

    def test_match_by_alias(self, matcher):
        assert matcher.match_league("EuroLeague") == "Euroleague"
        assert matcher.match_league("Liga Endesa") == "Spain ACB"
        assert matcher.match_league("ACB") == "Spain ACB"
        assert matcher.match_league("VTB") == "VTB United League"
        assert matcher.match_league("BSL") == "Turkish Super Lig"
        assert matcher.match_league("BBL") == "German BBL"

    def test_alias_case_insensitive(self, matcher):
        assert matcher.match_league("liga endesa") == "Spain ACB"
        assert matcher.match_league("vtb") == "VTB United League"

    def test_fuzzy_match(self, matcher):
        # Close enough names should fuzzy match
        assert matcher.match_league("Euro League") == "Euroleague"
        assert matcher.match_league("Spanish ACB") == "Spain ACB"

    def test_no_match_returns_none(self, matcher):
        assert matcher.match_league("Martian Basketball League") is None
        assert matcher.match_league("") is None

    def test_whitespace_handling(self, matcher):
        assert matcher.match_league("  NBA  ") == "NBA"
        assert matcher.match_league(" Euroleague ") == "Euroleague"

    def test_get_flashscore_url(self, matcher):
        url = matcher.get_flashscore_url("Euroleague")
        assert url is not None
        assert "flashscore.com" in url
        assert "euroleague" in url

    def test_get_flashscore_url_unknown(self, matcher):
        assert matcher.get_flashscore_url("Unknown League") is None

    def test_get_source_nba(self, matcher):
        assert matcher.get_source("NBA") == "nba_api"

    def test_get_source_default_is_flashscore(self, matcher):
        # Leagues without explicit source default to flashscore
        assert matcher.get_source("Euroleague") == "flashscore"

    def test_get_all_flashscore_urls_excludes_nba_api(self, matcher):
        urls = matcher.get_all_flashscore_urls()
        # NBA uses nba_api, so it should NOT be in the flashscore URLs
        assert "NBA" not in urls
        # But other leagues should be present
        assert "Euroleague" in urls
        assert "Spain ACB" in urls

    def test_get_all_flashscore_urls_has_valid_urls(self, matcher):
        urls = matcher.get_all_flashscore_urls()
        for name, url in urls.items():
            assert url.startswith("https://www.flashscore.com/"), (
                f"{name} has invalid URL: {url}"
            )
            assert url.endswith("/results/"), (
                f"{name} URL doesn't end with /results/: {url}"
            )


class TestLoadLeagueConfig:
    def test_loads_yaml(self):
        config = load_league_config(CONFIG_PATH)
        assert "leagues" in config
        assert "NBA" in config["leagues"]

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_league_config(Path("/nonexistent/path.yaml"))


# --- TeamMatcher ---


class TestTeamMatcher:
    def test_exact_match(self, team_matcher):
        assert team_matcher.match_team("Barcelona", "Euroleague") == "Barcelona"
        assert team_matcher.match_team("Real Madrid", "Euroleague") == "Real Madrid"

    def test_exact_match_case_insensitive(self, team_matcher):
        assert team_matcher.match_team("barcelona", "Euroleague") == "Barcelona"
        assert team_matcher.match_team("REAL MADRID", "Euroleague") == "Real Madrid"

    def test_fuzzy_match(self, team_matcher):
        assert team_matcher.match_team("FC Barcelona", "Euroleague") == "Barcelona"
        assert team_matcher.match_team("Fenerbahce Istanbul", "Euroleague") == "Fenerbahce"

    def test_no_match_below_threshold(self, team_matcher):
        assert team_matcher.match_team("ZZZZZ Unknown", "Euroleague") is None

    def test_no_match_unknown_league(self, team_matcher):
        # No teams registered for this league
        assert team_matcher.match_team("Barcelona", "Unknown League") is None

    def test_caching(self, team_matcher):
        # First call
        result1 = team_matcher.match_team("Barcelona", "Euroleague")
        # Second call should return from cache
        result2 = team_matcher.match_team("Barcelona", "Euroleague")
        assert result1 == result2 == "Barcelona"
        assert ("Euroleague", "Barcelona") in team_matcher._cache

    def test_cache_miss_stored(self, team_matcher):
        # A failed match should also be cached (as None)
        team_matcher.match_team("ZZZZZ", "Euroleague")
        assert ("Euroleague", "ZZZZZ") in team_matcher._cache
        assert team_matcher._cache[("Euroleague", "ZZZZZ")] is None

    def test_match_game(self, team_matcher):
        home, away = team_matcher.match_game("Barcelona", "Real Madrid", "Euroleague")
        assert home == "Barcelona"
        assert away == "Real Madrid"

    def test_match_game_partial_failure(self, team_matcher):
        home, away = team_matcher.match_game("Barcelona", "ZZZZZ", "Euroleague")
        assert home == "Barcelona"
        assert away is None

    def test_register_teams_deduplicates(self, team_matcher):
        team_matcher.register_teams("Test", ["A", "A", "B", "B", "B"])
        assert len(team_matcher._league_teams["Test"]) == 2

    def test_cross_league_isolation(self, team_matcher):
        # "Los Angeles Lakers" is registered for NBA, not Euroleague
        assert team_matcher.match_team("Los Angeles Lakers", "NBA") == "Los Angeles Lakers"
        assert team_matcher.match_team("Los Angeles Lakers", "Euroleague") is None
