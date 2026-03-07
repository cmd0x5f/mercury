"""Rolling form metrics — recent margin averages and trends."""

import pandas as pd

from src.config import get as cfg


def compute_rolling_margins(games: pd.DataFrame, window: int = None) -> pd.DataFrame:
    """Add rolling average absolute margin for each team (before each game).

    Returns games with added columns:
        home_avg_margin, away_avg_margin (rolling mean of absolute margins)
        home_avg_scored, away_avg_scored (rolling mean of points scored)
    """
    if window is None:
        window = cfg("model", "rolling_window", 10)

    games = games.copy().sort_values("date").reset_index(drop=True)

    # Build per-team rolling stats
    team_margins: dict[str, list[float]] = {}
    team_scores: dict[str, list[float]] = {}

    home_avg_margin, away_avg_margin = [], []
    home_avg_scored, away_avg_scored = [], []

    for _, g in games.iterrows():
        h, a = g["home_team"], g["away_team"]
        abs_m = abs(g["home_score"] - g["away_score"])

        # Record pre-game stats
        h_margins = team_margins.get(h, [])
        a_margins = team_margins.get(a, [])
        h_scores = team_scores.get(h, [])
        a_scores = team_scores.get(a, [])

        home_avg_margin.append(
            sum(h_margins[-window:]) / len(h_margins[-window:]) if h_margins else 10.0
        )
        away_avg_margin.append(
            sum(a_margins[-window:]) / len(a_margins[-window:]) if a_margins else 10.0
        )
        home_avg_scored.append(
            sum(h_scores[-window:]) / len(h_scores[-window:]) if h_scores else 110.0
        )
        away_avg_scored.append(
            sum(a_scores[-window:]) / len(a_scores[-window:]) if a_scores else 110.0
        )

        # Update post-game
        team_margins.setdefault(h, []).append(abs_m)
        team_margins.setdefault(a, []).append(abs_m)
        team_scores.setdefault(h, []).append(g["home_score"])
        team_scores.setdefault(a, []).append(g["away_score"])

    games["home_avg_margin"] = home_avg_margin
    games["away_avg_margin"] = away_avg_margin
    games["home_avg_scored"] = home_avg_scored
    games["away_avg_scored"] = away_avg_scored
    return games
