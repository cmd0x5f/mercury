"""Tests for injury report scraping."""

import json
from unittest.mock import patch, MagicMock

import pytest

from src.data.injury_scraper import (
    InjuryEntry,
    fetch_injuries_rotowire,
    _parse_bball_ref_status,
    get_injured_players,
)


MOCK_ROTOWIRE_RESPONSE = json.dumps([
    {"player": "LeBron James", "team": "LAL", "position": "SF",
     "injury": "Knee", "status": "Out", "ID": "1", "URL": "", "firstname": "LeBron",
     "lastname": "James", "rDate": ""},
    {"player": "Stephen Curry", "team": "GSW", "position": "PG",
     "injury": "Ankle", "status": "Questionable", "ID": "2", "URL": "", "firstname": "Stephen",
     "lastname": "Curry", "rDate": ""},
    {"player": "Jayson Tatum", "team": "BOS", "position": "SF",
     "injury": "Wrist", "status": "Probable", "ID": "3", "URL": "", "firstname": "Jayson",
     "lastname": "Tatum", "rDate": ""},
    {"player": "Zach Collins", "team": "SAS", "position": "C",
     "injury": "Toe", "status": "Out For Season", "ID": "4", "URL": "", "firstname": "Zach",
     "lastname": "Collins", "rDate": ""},
    {"player": "Chet Holmgren", "team": "OKC", "position": "PF",
     "injury": "Hip", "status": "Doubtful", "ID": "5", "URL": "", "firstname": "Chet",
     "lastname": "Holmgren", "rDate": ""},
])


def _mock_urlopen(mock_data):
    """Create a mock for urllib.request.urlopen that returns mock_data."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = mock_data.encode()
    return mock_resp


class TestFetchInjuriesRotowire:
    @patch("src.data.injury_scraper.urllib.request.urlopen")
    def test_parses_json_correctly(self, mock_urlopen_fn):
        mock_urlopen_fn.return_value = _mock_urlopen(MOCK_ROTOWIRE_RESPONSE)
        entries = fetch_injuries_rotowire()
        assert len(entries) == 5
        assert entries[0].player_name == "LeBron James"
        assert entries[0].team == "LAL"
        assert entries[0].status == "out"

    @patch("src.data.injury_scraper.urllib.request.urlopen")
    def test_returns_empty_on_failure(self, mock_urlopen_fn):
        mock_urlopen_fn.side_effect = Exception("Network error")
        entries = fetch_injuries_rotowire()
        assert entries == []

    @patch("src.data.injury_scraper.urllib.request.urlopen")
    def test_status_lowercased(self, mock_urlopen_fn):
        mock_urlopen_fn.return_value = _mock_urlopen(MOCK_ROTOWIRE_RESPONSE)
        entries = fetch_injuries_rotowire()
        # Rotowire returns "Out For Season", should be lowered to "out for season"
        season_entry = [e for e in entries if e.player_name == "Zach Collins"][0]
        assert season_entry.status == "out for season"


class TestParseBballRefStatus:
    def test_out_for_season(self):
        assert _parse_bball_ref_status("Out For Season (Toe) - surgery...") == "out for season"

    def test_ruled_out(self):
        assert _parse_bball_ref_status("ruled out for Saturday's game") == "out"

    def test_did_not_play(self):
        assert _parse_bball_ref_status("did not play on Friday's game") == "out"

    def test_questionable(self):
        assert _parse_bball_ref_status("listed as Questionable for tonight") == "questionable"

    def test_doubtful(self):
        assert _parse_bball_ref_status("listed as Doubtful for Saturday") == "doubtful"

    def test_probable(self):
        assert _parse_bball_ref_status("listed as Probable for tonight") == "probable"

    def test_unknown(self):
        assert _parse_bball_ref_status("some random text") == "unknown"


class TestGetInjuredPlayers:
    @patch("src.data.injury_scraper.fetch_injuries_rotowire")
    def test_filters_out_players(self, mock_fetch):
        mock_fetch.return_value = [
            InjuryEntry("LeBron James", "LAL", "out", "Knee"),
            InjuryEntry("Jayson Tatum", "BOS", "probable", "Wrist"),
        ]
        result = get_injured_players(include_questionable=False)
        assert "LAL" in result
        assert "LeBron James" in result["LAL"]
        assert "BOS" not in result  # probable is not "out"

    @patch("src.data.injury_scraper.fetch_injuries_rotowire")
    def test_includes_questionable_when_flag_set(self, mock_fetch):
        mock_fetch.return_value = [
            InjuryEntry("Stephen Curry", "GSW", "questionable", "Ankle"),
            InjuryEntry("Jayson Tatum", "BOS", "probable", "Wrist"),
        ]
        result = get_injured_players(include_questionable=True)
        assert "GSW" in result
        assert "Stephen Curry" in result["GSW"]
        assert "BOS" not in result  # probable still excluded

    @patch("src.data.injury_scraper.fetch_injuries_rotowire")
    def test_groups_by_team(self, mock_fetch):
        mock_fetch.return_value = [
            InjuryEntry("Player A", "LAL", "out", "Knee"),
            InjuryEntry("Player B", "LAL", "out", "Ankle"),
            InjuryEntry("Player C", "BOS", "out", "Back"),
        ]
        result = get_injured_players(include_questionable=False)
        assert len(result["LAL"]) == 2
        assert len(result["BOS"]) == 1

    @patch("src.data.injury_scraper.fetch_injuries_rotowire")
    @patch("src.data.injury_scraper.fetch_injuries_bball_ref")
    def test_falls_back_to_bball_ref(self, mock_bball, mock_roto):
        mock_roto.return_value = []  # primary fails
        mock_bball.return_value = [
            InjuryEntry("LeBron James", "LAL", "out", "Knee"),
        ]
        result = get_injured_players()
        assert "LAL" in result

    @patch("src.data.injury_scraper.fetch_injuries_rotowire")
    def test_out_for_season_included(self, mock_fetch):
        mock_fetch.return_value = [
            InjuryEntry("Zach Collins", "SAS", "out for season", "Toe"),
        ]
        result = get_injured_players(include_questionable=False)
        assert "SAS" in result

    @patch("src.data.injury_scraper.fetch_injuries_rotowire")
    def test_doubtful_included_with_questionable(self, mock_fetch):
        mock_fetch.return_value = [
            InjuryEntry("Chet Holmgren", "OKC", "doubtful", "Hip"),
        ]
        result = get_injured_players(include_questionable=True)
        assert "OKC" in result
