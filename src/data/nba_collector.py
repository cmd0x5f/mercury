"""Collect NBA game results via nba_api + Flashscore for latest games."""

import asyncio
import logging
import time
from datetime import datetime

import pandas as pd
from nba_api.stats.endpoints import LeagueGameLog

from src.config import get as cfg
from src.data.data_store import DataStore
from src.data.team_names import SPORTSPLUS_TO_NBA

logger = logging.getLogger(__name__)

FIRST_SEASON_YEAR = cfg("nba", "first_season_year", 2022)


def current_nba_season() -> str:
    """Auto-detect the current NBA season string (e.g. '2025-26').

    NBA seasons start in October, so:
    - Oct 2025 through Jun 2026 → '2025-26'
    - Jul 2026 through Sep 2026 → still '2025-26' (offseason)
    """
    now = datetime.now()
    if now.month >= 10:
        start_year = now.year
    else:
        start_year = now.year - 1
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def generate_seasons(first_year: int = None) -> list[str]:
    """Generate all NBA season strings from first_year to current season."""
    if first_year is None:
        first_year = FIRST_SEASON_YEAR

    current = current_nba_season()
    current_start = int(current.split("-")[0])

    seasons = []
    for year in range(first_year, current_start + 1):
        seasons.append(f"{year}-{(year + 1) % 100:02d}")
    return seasons


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


def _normalize_flashscore_team(name: str) -> str | None:
    """Convert a Flashscore full team name to nba_api abbreviation."""
    return SPORTSPLUS_TO_NBA.get(name)


def fetch_latest_from_flashscore() -> pd.DataFrame:
    """Scrape the latest NBA results from Flashscore and normalize team names.

    Returns a DataFrame in the same format as fetch_season() (with nba_api
    abbreviations for team names), so both sources can coexist in the DB.
    """
    from src.data.flashscore_scraper import scrape_league_results

    async def _scrape():
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            df = await scrape_league_results(
                context,
                "https://www.flashscore.com/basketball/usa/nba/results/",
                "NBA",
                max_clicks=0,
            )
            await context.close()
            await browser.close()
        return df

    df = asyncio.run(_scrape())
    if df.empty:
        return df

    # Normalize team names to nba_api abbreviations
    df["home_team"] = df["home_team"].map(_normalize_flashscore_team)
    df["away_team"] = df["away_team"].map(_normalize_flashscore_team)

    # Drop games where team names couldn't be mapped
    unmapped = df[df["home_team"].isna() | df["away_team"].isna()]
    if len(unmapped) > 0:
        logger.warning(f"Could not map {len(unmapped)} Flashscore NBA games (unknown team names)")
    df = df.dropna(subset=["home_team", "away_team"])

    # Change source to distinguish from nba_api, but keep same team name format
    df["source"] = "flashscore_nba"

    return df


def collect_nba(seasons: list[str] = None, store: DataStore = None):
    """Fetch NBA seasons from nba_api, then fill in latest games from Flashscore."""
    seasons = seasons or generate_seasons()
    store = store or DataStore()

    all_games = []
    for season in seasons:
        df = fetch_season(season)
        logger.info(f"  {season}: {len(df)} games")
        all_games.append(df)

    combined = pd.concat(all_games, ignore_index=True)
    store.upsert_games(combined)
    nba_api_latest = combined["date"].max()
    logger.info(f"nba_api: {len(combined)} games (latest: {nba_api_latest})")

    # Fill in latest games from Flashscore (today's completed games)
    logger.info("Fetching latest NBA games from Flashscore...")
    fs_df = fetch_latest_from_flashscore()
    if not fs_df.empty:
        # Only keep Flashscore games newer than what nba_api has
        fs_new = fs_df[fs_df["date"] > nba_api_latest]
        if not fs_new.empty:
            store.upsert_games(fs_new)
            logger.info(f"Flashscore: {len(fs_new)} newer games added (up to {fs_new['date'].max()})")
        else:
            logger.info("Flashscore: no games newer than nba_api")
    else:
        logger.info("Flashscore: no NBA games scraped")

    total = store.get_games(league="NBA")
    logger.info(f"Total NBA games in DB: {len(total)} (latest: {total['date'].max()})")
    return total


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collect_nba()
