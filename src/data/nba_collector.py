"""Collect NBA game results via nba_api."""

import logging
import time

import pandas as pd
from nba_api.stats.endpoints import LeagueGameLog

from src.data.data_store import DataStore

logger = logging.getLogger(__name__)

SEASONS = ["2022-23", "2023-24", "2024-25"]


def fetch_season(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """Fetch all games for one NBA season. Returns one row per game."""
    logger.info(f"Fetching NBA {season} ({season_type})...")
    time.sleep(0.6)  # rate limit

    log = LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
        league_id="00",
    )
    df = log.get_data_frames()[0]

    # Each game has 2 rows (one per team). Keep home team rows only.
    # MATCHUP contains "vs." for home and "@" for away
    home = df[df["MATCHUP"].str.contains("vs.")].copy()
    away = df[df["MATCHUP"].str.contains("@")].copy()

    # Parse team abbreviations from matchup
    home["home_team"] = home["MATCHUP"].str.split(" vs. ").str[0].str.strip()
    home["away_team"] = home["MATCHUP"].str.split(" vs. ").str[1].str.strip()
    home["home_score"] = home["PTS"]

    # Merge away scores
    away_scores = away[["GAME_ID", "PTS"]].rename(columns={"PTS": "away_score"})
    games = home.merge(away_scores, on="GAME_ID")

    games = games.rename(columns={"GAME_ID": "game_id", "GAME_DATE": "date"})
    games["date"] = pd.to_datetime(games["date"]).dt.strftime("%Y-%m-%d")
    games["source"] = "nba_api"
    games["league"] = "NBA"

    return games[["source", "league", "game_id", "date",
                   "home_team", "away_team", "home_score", "away_score"]]


def collect_nba(seasons: list[str] = None, store: DataStore = None):
    """Fetch multiple seasons and store them."""
    seasons = seasons or SEASONS
    store = store or DataStore()

    all_games = []
    for season in seasons:
        df = fetch_season(season)
        logger.info(f"  {season}: {len(df)} games")
        all_games.append(df)

    combined = pd.concat(all_games, ignore_index=True)
    store.upsert_games(combined)
    logger.info(f"Total: {len(combined)} NBA games stored")
    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collect_nba()
