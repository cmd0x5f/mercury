"""Feature builder — combines all feature modules into a model-ready DataFrame."""

import pandas as pd

from src.features.context import compute_context_features
from src.features.form import compute_rolling_margins
from src.features.team_strength import compute_elo_ratings

# Columns used as model input features
FEATURE_COLS = [
    "elo_diff",
    "home_elo",
    "away_elo",
    "home_avg_margin",
    "away_avg_margin",
    "home_avg_scored",
    "away_avg_scored",
    "home_rest_days",
    "away_rest_days",
    "home_game_num",
    "away_game_num",
    "is_b2b_home",
    "is_b2b_away",
    "league_id",
]

TARGET_COL = "margin"  # home_score - away_score (signed)


def build_features(games: pd.DataFrame, elo_k: int = 20) -> pd.DataFrame:
    """Build all features from raw game data.

    Input must have: date, home_team, away_team, home_score, away_score
    Optionally: league (defaults to 'NBA')
    Returns DataFrame with all feature columns + target.
    """
    games = games.sort_values("date").reset_index(drop=True)

    # Ensure margin column exists
    if "margin" not in games.columns:
        games["margin"] = games["home_score"] - games["away_score"]

    # Ensure league column exists
    if "league" not in games.columns:
        games["league"] = "NBA"

    # Encode league as integer ID
    league_cats = pd.Categorical(games["league"])
    games["league_id"] = league_cats.codes

    # Compute features per league (Elo ratings are league-specific)
    parts = []
    for league, group in games.groupby("league"):
        group = group.copy()
        group = compute_elo_ratings(group, k=elo_k)
        group = compute_rolling_margins(group)
        group = compute_context_features(group)
        parts.append(group)

    result = pd.concat(parts).sort_values("date").reset_index(drop=True)

    # Store the league category mapping for later use
    result.attrs["league_categories"] = dict(
        zip(league_cats.categories, range(len(league_cats.categories)))
    )

    return result


def get_feature_matrix(games: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract X (features) and y (target) from a fully-featured games DataFrame."""
    return games[FEATURE_COLS], games[TARGET_COL]
