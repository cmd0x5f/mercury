"""Scrape winning margin odds from SportsPlus.ph using Playwright."""

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


async def _dismiss_terms_dialog(page: Page):
    """Dismiss the PAGCOR terms overlay if present."""
    close_btn = await page.query_selector("#terms-check .close_btn")
    if close_btn:
        await close_btn.click()
        await page.wait_for_timeout(500)


async def get_game_links(page: Page) -> list[dict]:
    """Get all basketball game links from the upcoming page.

    Returns list of {url, home_team, away_team, league, game_date}.
    League is detected from section headers on the page.
    """
    await page.goto(BASKETBALL_URL, wait_until="networkidle")
    await page.wait_for_timeout(2000)
    await _dismiss_terms_dialog(page)

    # Use JS to walk the DOM and extract league headers + game links in order.
    # League headers are text nodes (not inside <a>) that appear between game groups.
    raw = await page.evaluate("""() => {
        const results = [];
        // The main content area contains league headers and game link blocks
        const container = document.querySelector('.events-container')
            || document.querySelector('.event-list')
            || document.querySelector('main')
            || document.body;

        let currentLeague = 'Unknown';

        // Walk through all elements in order
        const walker = document.createTreeWalker(
            container, NodeFilter.SHOW_ELEMENT, null
        );
        const seen = new Set();
        let node;
        while (node = walker.nextNode()) {
            // League header: standalone text not inside a game link
            if (!node.closest('a[href*="/sbk/m/"]') && !node.querySelector('a')) {
                const text = node.textContent.trim();
                // League names are short, no digits-heavy, no odds patterns
                if (text && text.length > 1 && text.length < 80
                    && !text.match(/^[\\d.+-]+$/)
                    && !text.match(/^\\d{2}\\/\\d{2}/)
                    && !text.match(/^(Next|Tournament|Outright|Sign|HDP|Main|Player)/)
                    && !text.match(/^Basketball/)
                    && node.children.length === 0
                    && node.tagName !== 'A') {
                    // Looks like a league name
                    currentLeague = text.replace(/\\s+/g, ' ').trim();
                }
            }

            // Game link
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


async def scrape_margin_odds(page: Page, game: dict) -> list[dict]:
    """Scrape winning margin odds for a single game.

    Returns list of {bucket, decimal_odds} dicts.
    """
    await page.goto(game["url"], wait_until="networkidle")
    await page.wait_for_timeout(1500)

    # Dismiss PAGCOR terms overlay that blocks clicks
    await _dismiss_terms_dialog(page)

    # Expand the "Any Team Winning Margin" section via JS click
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
        logger.warning(
            f"Margin section not found for "
            f"{game['home_team']} vs {game['away_team']}"
        )
        return []

    await page.wait_for_timeout(1500)

    odds = await _extract_margins_from_text(page)

    logger.info(
        f"  [{game.get('league', '?')}] "
        f"{game['home_team']} vs {game['away_team']}: "
        f"{len(odds)} margin buckets found"
    )
    return odds


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


async def scrape_all_margins(headless: bool = True) -> pd.DataFrame:
    """Scrape margin odds for all upcoming basketball games.

    Returns DataFrame with columns:
        scraped_at, source, league, game_date,
        home_team, away_team, bucket, decimal_odds
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page()

        games = await get_game_links(page)
        all_odds = []

        for game in games:
            odds = await scrape_margin_odds(page, game)
            for o in odds:
                if "decimal_odds" in o:
                    all_odds.append({
                        "scraped_at": datetime.now().isoformat(),
                        "source": "sportsplus",
                        "league": game.get("league", "Unknown"),
                        "game_date": game["game_date"],
                        "home_team": game["home_team"],
                        "away_team": game["away_team"],
                        "bucket": o["bucket"],
                        "decimal_odds": o["decimal_odds"],
                    })

        await browser.close()

    df = pd.DataFrame(all_odds)
    logger.info(
        f"Scraped {len(df)} total margin odds "
        f"across {len(games)} games"
    )
    return df


def scrape_odds_sync(headless: bool = True) -> pd.DataFrame:
    """Synchronous wrapper for scrape_all_margins."""
    return asyncio.run(scrape_all_margins(headless=headless))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = scrape_odds_sync(headless=False)
    print(df.to_string())
