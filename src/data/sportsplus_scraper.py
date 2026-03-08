"""Scrape winning margin odds from SportsPlus.ph using Playwright.

Performance improvements:
- domcontentloaded instead of networkidle (avoids ad/tracker hangs)
- Concurrent game scraping with configurable parallelism
- Retry with exponential backoff on failures
- Retry pass for rate-limited games with reduced concurrency
- Separate page per concurrent game via browser context
"""

import asyncio
import logging
import re
from datetime import datetime

import pandas as pd
from playwright.async_api import Page, async_playwright

logger = logging.getLogger(__name__)

BASE_URL = "https://www.sportsplus.ph"
BASKETBALL_URL = f"{BASE_URL}/sbk/2/upcoming"

MARGIN_SECTION = "Any Team Winning Margin (Incl. Overtime)"
BUCKET_PATTERN = re.compile(r"^(\d+)\s*-\s*(\d+)$|^(\d+)\+$")

DEFAULT_CONCURRENCY = 3
MARKET_EXPAND_WAIT = 800
MARKET_SELECTOR = "p.market-title-content"
STAGGER_DELAY = 0.5  # seconds between launching concurrent batches


class MarketsNotLoadedError(Exception):
    """Raised when a page loads but market elements don't render (likely rate-limited)."""
    pass


async def _retry_async(fn, retries: int = 2, base_delay: float = 2.0):
    """Retry an async function with exponential backoff."""
    last_error = None
    for attempt in range(retries + 1):
        try:
            return await fn()
        except Exception as e:
            last_error = e
            if attempt < retries:
                delay = base_delay * (attempt + 1)
                logger.debug(f"  Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
    raise last_error


async def _dismiss_terms_dialog(page: Page):
    """Dismiss the PAGCOR terms overlay if present."""
    try:
        close_btn = await page.query_selector("#terms-check .close_btn")
        if close_btn:
            await close_btn.click()
            await page.wait_for_timeout(300)
    except Exception:
        pass


async def get_game_links(page: Page) -> list[dict]:
    """Get all basketball game links from the upcoming page.

    Returns list of {url, home_team, away_team, league, game_date}.
    """
    await page.goto(BASKETBALL_URL, wait_until="domcontentloaded")
    # Wait for game links to render (JS-driven)
    try:
        await page.wait_for_selector('a[href*="/sbk/m/"]', timeout=10000)
    except Exception:
        logger.warning("No game links found on upcoming page")
        return []
    await _dismiss_terms_dialog(page)

    raw = await page.evaluate("""() => {
        const results = [];
        const container = document.querySelector('.events-container')
            || document.querySelector('.event-list')
            || document.querySelector('main')
            || document.body;

        let currentLeague = 'Unknown';

        const walker = document.createTreeWalker(
            container, NodeFilter.SHOW_ELEMENT, null
        );
        const seen = new Set();
        let node;
        while (node = walker.nextNode()) {
            if (!node.closest('a[href*="/sbk/m/"]') && !node.querySelector('a')) {
                const text = node.textContent.trim();
                if (text && text.length > 1 && text.length < 80
                    && !text.match(/^[\\d.+-]+$/)
                    && !text.match(/^\\d{2}\\/\\d{2}/)
                    && !text.match(/^(Next|Tournament|Outright|Sign|HDP|Main|Player)/)
                    && !text.match(/^Basketball/)
                    && node.children.length === 0
                    && node.tagName !== 'A') {
                    currentLeague = text.replace(/\\s+/g, ' ').trim();
                }
            }

            if (node.tagName === 'A' && node.href && node.href.includes('/sbk/m/')) {
                if (seen.has(node.href)) continue;
                seen.add(node.href);
                const lines = node.innerText.split('\\n')
                    .map(s => s.trim()).filter(s => s);
                if (lines.length >= 3) {
                    results.push({
                        url: node.href,
                        home_team: lines[0],
                        away_team: lines[1],
                        date_str: lines[2],
                        league: currentLeague,
                    });
                }
            }
        }
        return results;
    }""")

    games = []
    for item in raw:
        date_match = re.search(r"(\d{2}/\d{2})", item["date_str"])
        if date_match:
            md = date_match.group(1)
            year = datetime.now().year
            game_date = f"{year}-{md[:2]}-{md[3:]}"
        else:
            game_date = datetime.now().strftime("%Y-%m-%d")

        games.append({
            "url": item["url"],
            "home_team": item["home_team"],
            "away_team": item["away_team"],
            "game_date": game_date,
            "league": item.get("league", "Unknown"),
        })

    leagues = set(g["league"] for g in games)
    logger.info(f"Found {len(games)} games across {len(leagues)} leagues: {leagues}")
    return games


async def scrape_margin_odds(context, game: dict) -> list[dict]:
    """Scrape winning margin odds for a single game using a fresh page.

    Returns list of {bucket, decimal_odds} dicts.
    Raises MarketsNotLoadedError if the page loads but markets don't render.
    """
    page = await context.new_page()
    try:
        await page.goto(game["url"], wait_until="domcontentloaded")
        await _dismiss_terms_dialog(page)

        # Wait for market sections to render (JS-driven)
        try:
            await page.wait_for_selector(MARKET_SELECTOR, timeout=8000)
        except Exception:
            raise MarketsNotLoadedError(
                f"Markets didn't load for "
                f"{game['home_team']} vs {game['away_team']}"
            )

        expanded = await page.evaluate("""() => {
            const el = [...document.querySelectorAll('p.market-title-content')]
                .find(e => e.textContent.includes('Any Team Winning Margin'));
            if (el) {
                const parent = el.closest('.market-title') || el.parentElement;
                parent.click();
                return true;
            }
            return false;
        }""")

        if not expanded:
            # Game genuinely doesn't offer this market — not retryable
            logger.debug(
                f"No margin market for "
                f"{game['home_team']} vs {game['away_team']}"
            )
            return []

        await page.wait_for_timeout(MARKET_EXPAND_WAIT)

        odds = await _extract_margins_from_text(page)

        logger.info(
            f"  [{game.get('league', '?')}] "
            f"{game['home_team']} vs {game['away_team']}: "
            f"{len(odds)} margin buckets"
        )
        return odds
    finally:
        await page.close()


async def _extract_margins_from_text(page: Page) -> list[dict]:
    """Extract margin odds by parsing visible text after section expand."""
    all_text = await page.inner_text("body")
    lines = [s.strip() for s in all_text.split("\n") if s.strip()]

    odds = []
    in_section = False

    for i, line in enumerate(lines):
        if MARGIN_SECTION in line:
            in_section = True
            continue

        if in_section:
            # Stop at next section header
            if "(" in line and ")" in line and not BUCKET_PATTERN.match(line):
                break

            match = BUCKET_PATTERN.match(line)
            if match:
                if match.group(3):
                    bucket = f"{match.group(3)}+"
                else:
                    bucket = f"{match.group(1)}-{match.group(2)}"

                # Look ahead for odds value
                for j in range(i + 1, min(i + 3, len(lines))):
                    try:
                        odds_val = float(lines[j])
                        if 1.0 < odds_val < 100.0:
                            odds.append({
                                "bucket": bucket,
                                "decimal_odds": odds_val,
                            })
                            break
                    except ValueError:
                        continue

    return odds


def _build_odds_record(game: dict, odds_item: dict) -> dict:
    """Build a single odds record from a game and odds item."""
    return {
        "scraped_at": datetime.now().isoformat(),
        "source": "sportsplus",
        "league": game.get("league", "Unknown"),
        "game_date": game["game_date"],
        "home_team": game["home_team"],
        "away_team": game["away_team"],
        "bucket": odds_item["bucket"],
        "decimal_odds": odds_item["decimal_odds"],
    }


async def _scrape_batch(
    context,
    games: list[dict],
    concurrency: int,
) -> tuple[list[dict], list[dict]]:
    """Scrape a batch of games. Returns (odds_records, failed_games)."""
    semaphore = asyncio.Semaphore(concurrency)
    all_odds = []
    failed = []

    async def _scrape_one(game: dict):
        async with semaphore:
            try:
                odds = await _retry_async(
                    lambda g=game: scrape_margin_odds(context, g),
                    retries=1,
                    base_delay=2.0,
                )
                return game, odds, None
            except MarketsNotLoadedError:
                return game, [], "rate_limited"
            except Exception as e:
                logger.error(f"Failed: {game['home_team']} vs {game['away_team']}: {e}")
                return game, [], "error"

    # Stagger task creation to avoid slamming the server
    tasks = []
    for i, game in enumerate(games):
        if i > 0 and i % concurrency == 0:
            await asyncio.sleep(STAGGER_DELAY)
        tasks.append(asyncio.create_task(_scrape_one(game)))

    results = await asyncio.gather(*tasks)

    for game, odds, error in results:
        if error == "rate_limited":
            failed.append(game)
        elif odds:
            for o in odds:
                if "decimal_odds" in o:
                    all_odds.append(_build_odds_record(game, o))

    return all_odds, failed


async def scrape_all_margins(
    headless: bool = True,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> pd.DataFrame:
    """Scrape margin odds for all upcoming basketball games.

    Uses two passes:
      1. Main pass at full concurrency
      2. Retry pass for rate-limited games at concurrency=1 with longer waits

    Args:
        headless: Run browser in headless mode.
        concurrency: Number of game pages to scrape in parallel.

    Returns DataFrame with columns:
        scraped_at, source, league, game_date,
        home_team, away_team, bucket, decimal_odds
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context()

        # Phase 1: Get all game links (single page, sequential)
        index_page = await context.new_page()
        games = await get_game_links(index_page)
        await index_page.close()

        if not games:
            await browser.close()
            return pd.DataFrame()

        # Phase 2: Main scrape pass
        all_odds, failed = await _scrape_batch(context, games, concurrency)
        logger.info(
            f"Pass 1: {len(all_odds)} odds scraped, "
            f"{len(failed)} games need retry"
        )

        # Phase 3: Retry failed games one at a time with longer waits
        if failed:
            logger.info(f"Retrying {len(failed)} rate-limited games (concurrency=1)...")
            await asyncio.sleep(3)  # cooldown before retry
            retry_odds, still_failed = await _scrape_batch(context, failed, concurrency=1)
            all_odds.extend(retry_odds)

            if still_failed:
                logger.warning(
                    f"{len(still_failed)} games failed after retry: "
                    + ", ".join(f"{g['home_team']} vs {g['away_team']}" for g in still_failed[:5])
                    + ("..." if len(still_failed) > 5 else "")
                )

        await context.close()
        await browser.close()

    df = pd.DataFrame(all_odds)
    logger.info(
        f"Scraped {len(df)} total margin odds "
        f"across {len(games)} games"
    )
    return df


def scrape_odds_sync(
    headless: bool = True,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> pd.DataFrame:
    """Synchronous wrapper for scrape_all_margins."""
    return asyncio.run(scrape_all_margins(headless=headless, concurrency=concurrency))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = scrape_odds_sync(headless=False)
    print(df.to_string())
