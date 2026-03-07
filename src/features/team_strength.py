"""Elo rating system for team strength estimation."""

import pandas as pd

from src.config import get as cfg


def expected_score(rating_a: float, rating_b: float) -> float:
    """Expected win probability for team A."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def compute_elo_ratings(
    games: pd.DataFrame,
    k: int = None,
    start_rating: float = None,
    home_advantage: float = 100,
) -> pd.DataFrame:
    """Compute Elo ratings for all teams over a sequence of games.

    Returns games DataFrame with added columns:
        home_elo, away_elo (ratings BEFORE each game)
        elo_diff (home_elo + home_advantage - away_elo)
    """
    if k is None:
        k = cfg("model", "elo_k_factor", 20)
    if start_rating is None:
        start_rating = cfg("model", "elo_start", 1500)

    ratings: dict[str, float] = {}
    home_elos, away_elos = [], []

    for _, game in games.iterrows():
        h, a = game["home_team"], game["away_team"]
        hr = ratings.get(h, start_rating)
        ar = ratings.get(a, start_rating)

        home_elos.append(hr)
        away_elos.append(ar)

        # Update: 1 = home win, 0 = away win
        actual = 1.0 if game["home_score"] > game["away_score"] else 0.0
        exp = expected_score(hr + home_advantage, ar)

        ratings[h] = hr + k * (actual - exp)
        ratings[a] = ar + k * ((1 - actual) - (1 - exp))

    games = games.copy()
    games["home_elo"] = home_elos
    games["away_elo"] = away_elos
    games["elo_diff"] = games["home_elo"] + home_advantage - games["away_elo"]
    return games
