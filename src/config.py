"""Centralized configuration loaded from config/settings.yaml."""

from pathlib import Path

import yaml

SETTINGS_PATH = Path(__file__).parents[1] / "config" / "settings.yaml"


def load_settings(path: Path = SETTINGS_PATH) -> dict:
    """Load settings from YAML. Returns empty dict if file missing."""
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


# Load once at import time — modules read from this dict
_settings = load_settings()


def get(section: str, key: str, default=None):
    """Get a setting value. Example: get('betting', 'kelly_fraction', 0.25)"""
    return _settings.get(section, {}).get(key, default)
