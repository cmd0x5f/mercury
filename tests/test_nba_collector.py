"""Tests for NBA season auto-detection and collection logic."""

from datetime import datetime
from unittest.mock import patch

import pytest

from src.data.nba_collector import current_nba_season, generate_seasons


class TestCurrentNbaSeason:
    def test_during_season(self):
        """March 2026 is during the 2025-26 season."""
        with patch("src.data.nba_collector.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 7)
            assert current_nba_season() == "2025-26"

    def test_season_start(self):
        """October is when the new season begins."""
        with patch("src.data.nba_collector.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2025, 10, 15)
            assert current_nba_season() == "2025-26"

    def test_offseason_summer(self):
        """July 2026 is offseason but still '2025-26' season."""
        with patch("src.data.nba_collector.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 7, 1)
            assert current_nba_season() == "2025-26"

    def test_january(self):
        """January 2026 is mid-season for 2025-26."""
        with patch("src.data.nba_collector.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 15)
            assert current_nba_season() == "2025-26"

    def test_century_rollover(self):
        """Handles 2099-00 season format correctly."""
        with patch("src.data.nba_collector.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2099, 11, 1)
            assert current_nba_season() == "2099-00"

    def test_returns_string_format(self):
        season = current_nba_season()
        assert isinstance(season, str)
        parts = season.split("-")
        assert len(parts) == 2
        assert len(parts[0]) == 4
        assert len(parts[1]) == 2


class TestGenerateSeasons:
    def test_generates_from_first_year_to_current(self):
        with patch("src.data.nba_collector.current_nba_season", return_value="2025-26"):
            seasons = generate_seasons(first_year=2022)
        assert seasons == ["2022-23", "2023-24", "2024-25", "2025-26"]

    def test_single_season(self):
        with patch("src.data.nba_collector.current_nba_season", return_value="2025-26"):
            seasons = generate_seasons(first_year=2025)
        assert seasons == ["2025-26"]

    def test_includes_current_season(self):
        """The current season should always be included."""
        seasons = generate_seasons()
        current = current_nba_season()
        assert current in seasons

    def test_seasons_are_contiguous(self):
        """No gaps between seasons."""
        seasons = generate_seasons(first_year=2020)
        for i in range(1, len(seasons)):
            prev_end = int(seasons[i - 1].split("-")[0])
            curr_start = int(seasons[i].split("-")[0])
            assert curr_start == prev_end + 1
