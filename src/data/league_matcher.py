"""Match SportsPlus league/team names to Flashscore equivalents using fuzzy matching."""

import logging
from pathlib import Path

import yaml
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parents[2] / "config" / "league_mappings.yaml"


def load_league_config(path: Path = CONFIG_PATH) -> dict:
    """Load league mappings from YAML config."""
    with open(path) as f:
        return yaml.safe_load(f)


class LeagueMatcher:
    """Match SportsPlus league and team names to known leagues/Flashscore data."""

    def __init__(self, config_path: Path = CONFIG_PATH):
        config = load_league_config(config_path)
        self.leagues = config.get("leagues", {})

        # Build lookup: all known league names (canonical + aliases) -> canonical
        self.league_lookup: dict[str, str] = {}
        for canonical, info in self.leagues.items():
            self.league_lookup[canonical.lower()] = canonical
            for alias in info.get("aliases", []):
                self.league_lookup[alias.lower()] = canonical

    def match_league(self, sportsplus_name: str, threshold: int = 70) -> str | None:
        """Match a SportsPlus league name to a canonical league name.

        Returns canonical league name or None if no match.
        """
        key = sportsplus_name.strip().lower()

        # Exact match (case-insensitive)
        if key in self.league_lookup:
            return self.league_lookup[key]

        # Fuzzy match against all known names
        candidates = list(self.league_lookup.keys())
        if not candidates:
            return None

        result = process.extractOne(
            key, candidates, scorer=fuzz.token_sort_ratio
        )
        if result and result[1] >= threshold:
            return self.league_lookup[result[0]]

        return None

    def get_flashscore_url(self, canonical_league: str) -> str | None:
        """Get Flashscore results URL for a canonical league name."""
        info = self.leagues.get(canonical_league, {})
        return info.get("flashscore")

    def get_source(self, canonical_league: str) -> str:
        """Get preferred data source for a league."""
        info = self.leagues.get(canonical_league, {})
        return info.get("source", "flashscore")

    def get_all_flashscore_urls(self) -> dict[str, str]:
        """Get all league->URL mappings for leagues using Flashscore."""
        urls = {}
        for name, info in self.leagues.items():
            if info.get("source") != "nba_api" and "flashscore" in info:
                urls[name] = info["flashscore"]
        return urls


class TeamMatcher:
    """Fuzzy match team names between SportsPlus and Flashscore."""

    def __init__(self):
        # Cache: (league, sportsplus_name) -> flashscore_name
        self._cache: dict[tuple[str, str], str | None] = {}
        # Known teams per league from Flashscore data
        self._league_teams: dict[str, list[str]] = {}

    def register_teams(self, league: str, teams: list[str]):
        """Register known team names from Flashscore for a league."""
        self._league_teams[league] = list(set(teams))

    def match_team(
        self,
        sportsplus_name: str,
        league: str,
        threshold: int = 65,
    ) -> str | None:
        """Match a SportsPlus team name to a Flashscore team name.

        Returns matched Flashscore name or None.
        """
        cache_key = (league, sportsplus_name)
        if cache_key in self._cache:
            return self._cache[cache_key]

        candidates = self._league_teams.get(league, [])
        if not candidates:
            return None

        # Try exact match first
        for c in candidates:
            if c.lower() == sportsplus_name.lower():
                self._cache[cache_key] = c
                return c

        # Fuzzy match
        result = process.extractOne(
            sportsplus_name, candidates, scorer=fuzz.token_sort_ratio
        )
        if result and result[1] >= threshold:
            matched = result[0]
            score = result[1]
            if score < 85:
                logger.debug(
                    f"Fuzzy match: '{sportsplus_name}' -> "
                    f"'{matched}' (score={score}, league={league})"
                )
            self._cache[cache_key] = matched
            return matched

        logger.debug(
            f"No match for '{sportsplus_name}' in {league} "
            f"(best: {result[0] if result else 'none'}, "
            f"score={result[1] if result else 0})"
        )
        self._cache[cache_key] = None
        return None

    def match_game(
        self,
        home: str,
        away: str,
        league: str,
        threshold: int = 65,
    ) -> tuple[str | None, str | None]:
        """Match both team names for a game."""
        return (
            self.match_team(home, league, threshold),
            self.match_team(away, league, threshold),
        )
