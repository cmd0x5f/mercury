"""Player availability impact features.

Computes how much player impact each team is missing in a given game
by comparing who actually played vs the team's full-strength roster.

Impact score per player is a rolling weighted combination of:
  - Minutes played (proxy for coach trust / role importance)
  - Points scored (offensive contribution)
  - Plus/minus (net impact on the court)
"""

import logging
import time

import pandas as pd

from src.config import get as cfg

logger = logging.getLogger(__name__)

# Weights for combining stats into a single impact score
W_MINUTES = cfg("model", "impact_w_minutes", 0.5)
W_POINTS = cfg("model", "impact_w_points", 0.3)
W_PLUSMINUS = cfg("model", "impact_w_plusminus", 0.2)

# Rolling window for computing player impact
IMPACT_WINDOW = cfg("model", "impact_window", 15)

# Minimum games to establish a player's impact baseline
MIN_PLAYER_GAMES = 5


def fetch_player_game_logs(season: str) -> pd.DataFrame:
    """Fetch all player game logs for a season from nba_api.

    Returns DataFrame with one row per player per game:
        player_id, player_name, team, game_id, date, minutes, points, plus_minus
    """
    from nba_api.stats.endpoints import LeagueGameLog

    logger.info(f"Fetching player game logs for {season}...")
    time.sleep(0.6)

    log = LeagueGameLog(
        season=season,
        player_or_team_abbreviation="P",
        season_type_all_star="Regular Season",
        league_id="00",
    )
    df = log.get_data_frames()[0]

    df = df.rename(columns={
        "PLAYER_ID": "player_id",
        "PLAYER_NAME": "player_name",
        "TEAM_ABBREVIATION": "team",
        "GAME_ID": "game_id",
        "GAME_DATE": "date",
        "MIN": "minutes",
        "PTS": "points",
        "PLUS_MINUS": "plus_minus",
    })
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    return df[["player_id", "player_name", "team", "game_id", "date",
               "minutes", "points", "plus_minus"]]


def compute_player_impact_scores(player_logs: pd.DataFrame) -> pd.DataFrame:
    """Compute a rolling impact score for each player before each game.

    For each player appearance, the impact score is their rolling average of:
        W_MINUTES * (min / 48) + W_POINTS * (pts / 30) + W_PLUSMINUS * (pm / 15)

    All components are normalized to roughly [0, 1] range so weights
    are meaningful. A star player (36 min, 28 pts, +8 pm) scores ~0.73.
    A bench player (12 min, 5 pts, +1 pm) scores ~0.19.

    Returns DataFrame with columns:
        player_id, player_name, team, game_id, date, impact_score
    """
    logs = player_logs.sort_values("date").copy()

    # Compute raw impact per game appearance
    logs["raw_impact"] = (
        W_MINUTES * (logs["minutes"] / 48).clip(0, 1.5)
        + W_POINTS * (logs["points"] / 30).clip(0, 2.0)
        + W_PLUSMINUS * ((logs["plus_minus"] + 15) / 30).clip(0, 1.5)
    )

    # Rolling average impact per player (before each game)
    results = []
    player_history: dict[int, list[float]] = {}

    for _, row in logs.iterrows():
        pid = row["player_id"]
        history = player_history.get(pid, [])

        if len(history) >= MIN_PLAYER_GAMES:
            window = history[-IMPACT_WINDOW:]
            impact = sum(window) / len(window)
        else:
            impact = None  # not enough history

        results.append({
            "player_id": pid,
            "player_name": row["player_name"],
            "team": row["team"],
            "game_id": row["game_id"],
            "date": row["date"],
            "impact_score": impact,
        })

        player_history.setdefault(pid, []).append(row["raw_impact"])

    return pd.DataFrame(results)


def compute_missing_impact(
    team_games: pd.DataFrame,
    player_impacts: pd.DataFrame,
) -> pd.DataFrame:
    """Compute missing player impact for each team in each game.

    For each game, identifies players who are on the team's roster (have played
    recently) but did NOT play in this game. Sums their impact scores to get
    `missing_impact` — how much firepower the team is missing.

    Args:
        team_games: games DataFrame with game_id, date, home_team, away_team
        player_impacts: output of compute_player_impact_scores()

    Returns:
        DataFrame with game_id, home_missing_impact, away_missing_impact
    """
    # Build roster tracker: for each team, track active players
    # A player is "on the roster" if they've played in the last 20 games
    ROSTER_WINDOW = 20

    impacts = player_impacts.dropna(subset=["impact_score"]).copy()
    impacts = impacts.sort_values("date")

    # Build: team -> {player_id: (impact_score, last_game_date, games_since_last)}
    team_rosters: dict[str, dict[int, dict]] = {}
    # Track game counts per team
    team_game_counts: dict[str, int] = {}

    # Pre-compute: for each (game_id, team), which players played and their impacts
    game_team_players: dict[tuple[str, str], dict[int, float]] = {}
    for _, row in impacts.iterrows():
        key = (row["game_id"], row["team"])
        if key not in game_team_players:
            game_team_players[key] = {}
        game_team_players[key][row["player_id"]] = row["impact_score"]

    # Process games in order
    games = team_games.sort_values("date").copy()
    home_missing = []
    away_missing = []

    for _, game in games.iterrows():
        gid = game["game_id"]
        home = game["home_team"]
        away = game["away_team"]

        # Get who played in this game
        home_played = game_team_players.get((gid, home), {})
        away_played = game_team_players.get((gid, away), {})

        # Calculate missing impact for each team
        home_miss = _calc_missing(team_rosters.get(home, {}), home_played, team_game_counts.get(home, 0))
        away_miss = _calc_missing(team_rosters.get(away, {}), away_played, team_game_counts.get(away, 0))

        home_missing.append(home_miss)
        away_missing.append(away_miss)

        # Update rosters with who played
        _update_roster(team_rosters, home, home_played, team_game_counts, ROSTER_WINDOW)
        _update_roster(team_rosters, away, away_played, team_game_counts, ROSTER_WINDOW)

    games = games.copy()
    games["home_missing_impact"] = home_missing
    games["away_missing_impact"] = away_missing

    return games[["game_id", "home_missing_impact", "away_missing_impact"]]


def _calc_missing(
    roster: dict[int, dict],
    played: dict[int, float],
    team_game_count: int,
) -> float:
    """Calculate total missing impact for a team in a game."""
    if team_game_count < MIN_PLAYER_GAMES:
        return 0.0  # not enough history to know who's missing

    missing = 0.0
    for pid, info in roster.items():
        if pid not in played and info["games_ago"] <= 10:
            # Player is expected to play but didn't
            missing += info["impact_score"]

    return round(missing, 4)


def _update_roster(
    team_rosters: dict[str, dict[int, dict]],
    team: str,
    played: dict[int, float],
    team_game_counts: dict[str, int],
    roster_window: int,
):
    """Update a team's roster after a game."""
    if team not in team_rosters:
        team_rosters[team] = {}
    if team not in team_game_counts:
        team_game_counts[team] = 0

    team_game_counts[team] += 1

    # Age out all players by one game
    for pid in list(team_rosters[team].keys()):
        team_rosters[team][pid]["games_ago"] += 1
        if team_rosters[team][pid]["games_ago"] > roster_window:
            del team_rosters[team][pid]

    # Update players who played
    for pid, impact in played.items():
        team_rosters[team][pid] = {
            "impact_score": impact,
            "games_ago": 0,
        }
