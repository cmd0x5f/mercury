"""Scrape historical basketball results from Flashscore.com."""

import asyncio
import logging
import re
from datetime import datetime

import pandas as pd
from playwright.async_api import Page, async_playwright

from src.data.data_store import DataStore

logger = logging.getLogger(__name__)

BASE_URL = "https://www.flashscore.com"


async def _dismiss_cookie_banner(page: Page):
    """Accept cookies on Flashscore if the banner appears."""
    try:
        btn = page.locator("#onetrust-accept-btn-handler")
        if await btn.is_visible(timeout=3000):
            await btn.click()
            await page.wait_for_timeout(500)
    except Exception:
        pass


async def _load_more_results(page: Page, clicks: int = 5):
    """Click 'Show more matches' to load older results."""
    for _ in range(clicks):
        try:
            more_btn = page.locator("a.event__more")
            if await more_btn.is_visible(timeout=2000):
                await more_btn.click()
                await page.wait_for_timeout(1500)
            else:
                break
        except Exception:
            break


async def scrape_league_results(
    page: Page,
    league_url: str,
    league_name: str,
    load_more_clicks: int = 5,
) -> pd.DataFrame:
    """Scrape results for a single league from Flashscore.

    Args:
        page: Playwright page
        league_url: Full URL like https://www.flashscore.com/basketball/usa/nba/results/
        league_name: Canonical league name for storage
        load_more_clicks: How many times to click 'Show more'

    Returns:
        DataFrame with source, league, game_id, date,
        home_team, away_team, home_score, away_score
    """
    resp = await page.goto(league_url, wait_until="networkidle")
    if resp and resp.status == 404:
        logger.warning(f"  {league_name}: 404 at {league_url} — skipping")
        return pd.DataFrame()
    await page.wait_for_timeout(2000)
    await _dismiss_cookie_banner(page)
    await _load_more_results(page, clicks=load_more_clicks)

    # Extract game data via JS — each match row contains its own
    # event__time div with "DD.MM. HH:MM" and team/score elements
    raw = await page.evaluate("""() => {
        const games = [];
        const rows = document.querySelectorAll(
            '.event__match, [class*="event__match"]'
        );

        for (const row of rows) {
            const homeEl = row.querySelector(
                '.event__participant--home, [class*="participant--home"]'
            );
            const awayEl = row.querySelector(
                '.event__participant--away, [class*="participant--away"]'
            );
            const homeScoreEl = row.querySelector(
                '.event__score--home, [class*="score--home"]'
            );
            const awayScoreEl = row.querySelector(
                '.event__score--away, [class*="score--away"]'
            );
            const timeEl = row.querySelector(
                '.event__time, [class*="event__time"]'
            );

            if (homeEl && awayEl && homeScoreEl && awayScoreEl) {
                const homeScore = parseInt(homeScoreEl.textContent.trim());
                const awayScore = parseInt(awayScoreEl.textContent.trim());
                if (!isNaN(homeScore) && !isNaN(awayScore)) {
                    const id = row.id || row.getAttribute('id') || '';
                    const dateStr = timeEl
                        ? timeEl.textContent.trim() : '';
                    games.push({
                        game_id: id,
                        date_str: dateStr,
                        home_team: homeEl.textContent.trim(),
                        away_team: awayEl.textContent.trim(),
                        home_score: homeScore,
                        away_score: awayScore,
                    });
                }
            }
        }
        return games;
    }""")

    games = []
    for item in raw:
        # Parse date — Flashscore uses formats like "05.03.2026" or "05.03."
        date = _parse_flashscore_date(item.get("date_str", ""))
        game_id = item.get("game_id", "")
        if not game_id:
            # Generate a deterministic ID from the game data
            game_id = (
                f"fs_{league_name}_{date}_"
                f"{item['home_team']}_{item['away_team']}"
            )

        games.append({
            "source": "flashscore",
            "league": league_name,
            "game_id": game_id,
            "date": date,
            "home_team": item["home_team"],
            "away_team": item["away_team"],
            "home_score": item["home_score"],
            "away_score": item["away_score"],
        })

    df = pd.DataFrame(games)
    logger.info(f"  {league_name}: {len(df)} games scraped from Flashscore")
    return df


def _parse_flashscore_date(date_str: str) -> str:
    """Parse Flashscore date formats into YYYY-MM-DD.

    Handles: "27.02. 03:45" (DD.MM. HH:MM), "05.03.2026", "05.03."
    When year is missing, defaults to current year but rolls back if the
    resulting date is in the future (e.g. Dec games scraped in Jan).
    """
    if not date_str:
        return datetime.now().strftime("%Y-%m-%d")

    match = re.search(r"(\d{2})\.(\d{2})\.(\d{4})?", date_str)
    if match:
        day, month = int(match.group(1)), int(match.group(2))
        if match.group(3):
            year = int(match.group(3))
        else:
            year = datetime.now().year
            # Roll back year if date would be in the future
            try:
                parsed = datetime(year, month, day)
                if parsed > datetime.now():
                    year -= 1
            except ValueError:
                pass  # invalid date, just use current year
        return f"{year:04d}-{month:02d}-{day:02d}"

    return datetime.now().strftime("%Y-%m-%d")


async def scrape_multiple_leagues(
    league_urls: dict[str, str],
    headless: bool = True,
    load_more_clicks: int = 5,
    store: DataStore = None,
) -> pd.DataFrame:
    """Scrape results for multiple leagues.

    Args:
        league_urls: {league_name: flashscore_results_url}
        headless: run browser headless
        load_more_clicks: how many times to load more per league
        store: optional DataStore to persist results

    Returns:
        Combined DataFrame of all games
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page()

        all_games = []
        for league_name, url in league_urls.items():
            try:
                df = await scrape_league_results(
                    page, url, league_name,
                    load_more_clicks=load_more_clicks,
                )
                all_games.append(df)
            except Exception as e:
                logger.error(f"Failed to scrape {league_name}: {e}")

        await browser.close()

    if not all_games:
        return pd.DataFrame()

    combined = pd.concat(all_games, ignore_index=True)

    if store:
        store.upsert_games(combined)
        logger.info(f"Stored {len(combined)} games from Flashscore")

    return combined


def scrape_leagues_sync(
    league_urls: dict[str, str],
    headless: bool = True,
    load_more_clicks: int = 5,
    store: DataStore = None,
) -> pd.DataFrame:
    """Synchronous wrapper for scrape_multiple_leagues."""
    return asyncio.run(
        scrape_multiple_leagues(
            league_urls, headless=headless,
            load_more_clicks=load_more_clicks, store=store,
        )
    )
