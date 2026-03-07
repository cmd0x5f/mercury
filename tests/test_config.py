"""Tests for centralized config loading."""

from pathlib import Path

import pytest

from src.config import get, load_settings


class TestConfig:
    def test_loads_settings_yaml(self):
        settings = load_settings()
        assert "nba" in settings
        assert "model" in settings
        assert "betting" in settings
        assert "sportsplus" in settings

    def test_get_existing_value(self):
        assert get("betting", "kelly_fraction") == 0.25
        assert get("betting", "edge_threshold") == 0.05
        assert get("model", "elo_k_factor") == 20
        assert get("model", "rolling_window") == 10
        assert get("model", "elo_start") == 1500

    def test_get_missing_key_returns_default(self):
        assert get("betting", "nonexistent_key", 42) == 42

    def test_get_missing_section_returns_default(self):
        assert get("nonexistent_section", "key", "fallback") == "fallback"

    def test_get_nba_first_season_year(self):
        year = get("nba", "first_season_year")
        assert isinstance(year, int)
        assert 2000 <= year <= 2030

    def test_loads_missing_file_returns_empty(self):
        settings = load_settings(Path("/nonexistent/settings.yaml"))
        assert settings == {}
