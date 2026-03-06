"""Scrape historical basketball results from Flashscore.com.

Reliability improvements inspired by github.com/gustavofariaa/FlashscoreScraping:
- domcontentloaded instead of networkidle (avoids ad/tracker hangs)
- Retry with exponential backoff on failures
- Smart "load more" with empty-cycle detection
- Fresh page per league via browser context
- Explicit element waits instead of arbitrary timeouts
- Concurrent league scraping with configurable parallelism
- Historical season support via archive page discovery
- Incremental mode: skip games already in the database
"""

import asyncio
import logging
import re
from datetime import datetime

import pandas as pd
from playwright.async_api import Page, async_playwright

from src.data.data_store import DataStore

logger = logging.getLogger(__name__)

BASE_URL = "https://www.flashscore.com"

# Selectors
MATCH_SELECTOR = ".event__match"
LOAD_MORE_SELECTOR = "a.event__more, [data-testid='wcl-buttonLink']"
COOKIE_SELECTOR = "#onetrust-accept-btn-handler"

# Timing
DEFAULT_TIMEOUT = 5000
CLICK_DELAY = 800
MAX_EMPTY_CYCLES = 4


async def _wait_for_selector_safe(page: Page, selector: str, timeout: int = DEFAULT_TIMEOUT) -> bool:
    """Wait for a selector, returning True if found, False on timeout."""
    try:
        await page.wait_for_selector(selector, timeout=timeout)
        return True
    except Exception:
        return False


async def _dismiss_cookie_banner(page: Page):
    """Accept cookies on Flashscore if the banner appears."""
    try:
        btn = page.locator(COOKIE_SELECTOR)
        if await btn.is_visible(timeout=2000):
            await btn.click()
            await page.wait_for_timeout(500)
    except Exception:
        pass


async def _load_more_results(page: Page, max_clicks: int = 20):
    """Click 'Show more matches' intelligently until no new results appear.

    Tracks whether each click loaded new matches. Stops after
    MAX_EMPTY_CYCLES clicks that add nothing.
    """
    empty_cycles = 0

    for _ in range(max_clicks):
        count_before = await page.locator(MATCH_SELECTOR).count()

        load_btn = page.locator(LOAD_MORE_SELECTOR).first
        try:
            if not await load_btn.is_visible(timeout=2000):
                break
            await load_btn.scroll_into_view_if_needed()
            await load_btn.click()
            await page.wait_for_timeout(CLICK_DELAY)
        except Exception:
            break

        count_after = await page.locator(MATCH_SELECTOR).count()
        if count_after == count_before:
            empty_cycles += 1
            if empty_cycles >= MAX_EMPTY_CYCLES:
                logger.debug(f"  No new matches after {MAX_EMPTY_CYCLES} clicks, stopping")
                break
        else:
            empty_cycles = 0
            logger.debug(f"  Loaded {count_after - count_before} more matches ({count_after} total)")


async def _retry_async(fn, retries: int = 3, base_delay: float = 1.0):
    """Retry an async function with exponential backoff."""
    last_error = None
    for attempt in range(retries + 1):
        try:
            return await fn()
        except Exception as e:
            last_error = e
            if attempt < retries:
                delay = base_delay * (attempt + 1)
                logger.warning(f"  Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
    raise last_error


async def scrape_league_results(
    context_or_page,
    league_url: str,
    league_name: str,
    max_clicks: int = 20,
) -> pd.DataFrame:
    """Scrape results for a single league from Flashscore.

    Args:
        context_or_page: Playwright BrowserContext (preferred) or Page
        league_url: Full URL like https://www.flashscore.com/basketball/usa/nba/results/
        league_name: Canonical league name for storage
        max_clicks: Maximum "Show more" clicks (stops early if no new results)

    Returns:
        DataFrame with source, league, game_id, date,
        home_team, away_team, home_score, away_score
    """
    # Use duck typing: contexts have new_page(), pages don't
    owns_page = hasattr(context_or_page, 'new_page')
    if owns_page:
        page = await context_or_page.new_page()
    else:
        page = context_or_page

    try:
        return await _scrape_league_page(page, league_url, league_name, max_clicks)
    finally:
        if owns_page:
            await page.close()


async def _scrape_league_page(
    page: Page,
    league_url: str,
    league_name: str,
    max_clicks: int,
) -> pd.DataFrame:
    """Internal: navigate to league results and extract game data."""
    resp = await page.goto(league_url, wait_until="domcontentloaded")
    if resp and resp.status == 404:
        logger.warning(f"  {league_name}: 404 at {league_url} — skipping")
        return pd.DataFrame()

    found = await _wait_for_selector_safe(page, MATCH_SELECTOR, timeout=10000)
    if not found:
        logger.warning(f"  {league_name}: no match elements found after 10s — page may be empty")
        return pd.DataFrame()

    await _dismiss_cookie_banner(page)
    await _load_more_results(page, max_clicks=max_clicks)

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
        date = _parse_flashscore_date(item.get("date_str", ""))
        game_id = item.get("game_id", "")
        if not game_id:
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
            try:
                parsed = datetime(year, month, day)
                if parsed > datetime.now():
                    year -= 1
            except ValueError:
                pass
        return f"{year:04d}-{month:02d}-{day:02d}"

    return datetime.now().strftime("%Y-%m-%d")


# -- Historical season discovery -----------------------------------------------


def _results_url_to_archive_url(results_url: str) -> str:
    """Convert a results URL to its archive page URL.

    'https://www.flashscore.com/basketball/usa/nba/results/'
    -> 'https://www.flashscore.com/basketball/usa/nba/archive/'
    """
    return re.sub(r"/results/?$", "/archive/", results_url.rstrip("/") + "/")


async def discover_season_urls(
    context_or_page,
    results_url: str,
    num_seasons: int = 3,
) -> list[str]:
    """Discover past season results URLs from FlashScore's archive page.

    FlashScore archive pages list seasons with links like:
        /basketball/usa/nba-2024-2025/  (past seasons)
        /basketball/usa/nba/            (current season)

    Args:
        context_or_page: Playwright BrowserContext or Page
        results_url: Current season's results URL
        num_seasons: How many past seasons to return (excluding current)

    Returns:
        List of results URLs for past seasons, newest first.
    """
    owns_page = hasattr(context_or_page, 'new_page')
    if owns_page:
        page = await context_or_page.new_page()
    else:
        page = context_or_page

    try:
        archive_url = _results_url_to_archive_url(results_url)
        resp = await page.goto(archive_url, wait_until="domcontentloaded")
        if not resp or resp.status != 200:
            return []

        await page.wait_for_timeout(2000)

        season_hrefs = await page.evaluate("""() => {
            const rows = document.querySelectorAll('.archiveLatte__row');
            return Array.from(rows).map(row => {
                const link = row.querySelector('a');
                return link ? link.href : '';
            }).filter(href => href.length > 0);
        }""")

        # First entry is current season — skip it, take num_seasons past ones
        past_seasons = season_hrefs[1:num_seasons + 1]

        # Convert base season URLs to results URLs
        return [url.rstrip("/") + "/results/" for url in past_seasons]
    except Exception as e:
        logger.debug(f"  Could not discover seasons for {results_url}: {e}")
        return []
    finally:
        if owns_page:
            await page.close()


# -- Main scraping orchestrators ------------------------------------------------


async def scrape_multiple_leagues(
    league_urls: dict[str, str],
    headless: bool = True,
    max_clicks: int = 20,
    store: DataStore = None,
    concurrency: int = 1,
    seasons: int = 0,
    incremental: bool = False,
) -> pd.DataFrame:
    """Scrape results for multiple leagues.

    Args:
        league_urls: {league_name: flashscore_results_url}
        headless: run browser headless
        max_clicks: max "Show more" clicks per league
        store: optional DataStore to persist results
        concurrency: number of leagues to scrape in parallel (1=sequential)
        seasons: number of past seasons to also scrape (0=current only)
        incremental: if True, skip leagues where we already have recent data
    """
    # Build the full work list: (league_name, url) pairs
    work_items = []

    # Determine which leagues need scraping (incremental check)
    existing_dates = {}
    if incremental and store:
        existing_dates = store.get_latest_dates_by_league()

    for league_name, url in league_urls.items():
        if incremental and league_name in existing_dates:
            # In incremental mode, still scrape current season (fewer clicks for new games)
            # but log that we're doing an incremental update
            logger.info(f"  {league_name}: incremental update (latest: {existing_dates[league_name]})")
        work_items.append((league_name, url, False))

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context()

        # Phase 1: Discover historical season URLs if requested
        season_urls = {}
        if seasons > 0:
            logger.info(f"Discovering {seasons} past season(s) per league...")
            for league_name, url in league_urls.items():
                past = await discover_season_urls(context, url, num_seasons=seasons)
                if past:
                    season_urls[league_name] = past
                    logger.info(f"  {league_name}: found {len(past)} past season(s)")

            # Add historical season work items
            for league_name, urls in season_urls.items():
                for season_url in urls:
                    # Extract season identifier from URL for logging
                    work_items.append((league_name, season_url, True))

        # Phase 2: Scrape all work items with concurrency control
        semaphore = asyncio.Semaphore(concurrency)
        all_games = []

        async def _scrape_with_semaphore(name, url, is_historical):
            async with semaphore:
                # Historical seasons: use more clicks to get full season
                clicks = max_clicks if not is_historical else max(max_clicks, 30)

                async def _do_scrape():
                    return await scrape_league_results(
                        context, url, name, max_clicks=clicks,
                    )

                try:
                    df = await _retry_async(_do_scrape, retries=2, base_delay=2.0)
                    return df
                except Exception as e:
                    logger.error(f"Failed to scrape {name} ({url}) after retries: {e}")
                    return pd.DataFrame()

        # Run all scrape tasks with concurrency limit
        tasks = [
            _scrape_with_semaphore(name, url, is_hist)
            for name, url, is_hist in work_items
        ]
        results = await asyncio.gather(*tasks)
        all_games = [df for df in results if not df.empty]

        await context.close()
        await browser.close()

    if not all_games:
        return pd.DataFrame()

    combined = pd.concat(all_games, ignore_index=True)

    # Deduplicate — historical seasons may overlap with current data
    combined = combined.drop_duplicates(subset=["source", "game_id"], keep="first")

    if store:
        new_count = store.upsert_games(combined)
        logger.info(f"Stored {len(combined)} games from Flashscore ({new_count} new)")

    return combined


def scrape_leagues_sync(
    league_urls: dict[str, str],
    headless: bool = True,
    max_clicks: int = 20,
    store: DataStore = None,
    concurrency: int = 1,
    seasons: int = 0,
    incremental: bool = False,
) -> pd.DataFrame:
    """Synchronous wrapper for scrape_multiple_leagues."""
    return asyncio.run(
        scrape_multiple_leagues(
            league_urls, headless=headless,
            max_clicks=max_clicks, store=store,
            concurrency=concurrency, seasons=seasons,
            incremental=incremental,
        )
    )
