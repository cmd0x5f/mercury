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

    Returns list of {url, home_team, away_team, league, game_date}
    """
    await page.goto(BASKETBALL_URL, wait_until="networkidle")
    await page.wait_for_timeout(2000)
    await _dismiss_terms_dialog(page)

    games = []

    # Get all game links
    game_links = await page.query_selector_all("a[href*='/sbk/m/']")

    for link in game_links:
        href = await link.get_attribute("href")
        text = await link.inner_text()
        lines = [s.strip() for s in text.strip().split("\n") if s.strip()]

        if len(lines) < 3:
            continue

        # First two non-empty lines are team names, third has date/time
        home_team = lines[0]
        away_team = lines[1]
        date_str = lines[2]

        # Parse date like "03/05 (Thu) 08:00"
        date_match = re.search(r"(\d{2}/\d{2})", date_str)
        if date_match:
            md = date_match.group(1)
            year = datetime.now().year
            game_date = f"{year}-{md[:2]}-{md[3:]}"
        else:
            game_date = datetime.now().strftime("%Y-%m-%d")

        url = href if href.startswith("http") else BASE_URL + href

        games.append({
            "url": url,
            "home_team": home_team,
            "away_team": away_team,
            "game_date": game_date,
        })

    logger.info(f"Found {len(games)} basketball games")
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
    # (Playwright's normal click fails because the element is far down
    # the page and scrollIntoView doesn't work with the nested scroll container)
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
        logger.warning(f"Margin section not found for {game['home_team']} vs {game['away_team']}")
        return []

    await page.wait_for_timeout(1500)

    odds = await _extract_margins_from_text(page)

    logger.info(
        f"  {game['home_team']} vs {game['away_team']}: "
        f"{len(odds)} margin buckets found"
    )
    return odds


async def _extract_margins_from_text(page: Page) -> list[dict]:
    """Fallback: extract margin odds by parsing visible text after section expand."""
    all_text = await page.inner_text("body")
    lines = [s.strip() for s in all_text.split("\n") if s.strip()]

    odds = []
    in_section = False

    for i, line in enumerate(lines):
        if MARGIN_SECTION in line:
            in_section = True
            continue

        if in_section:
            # Stop at next section header (contains parentheses and multiple words)
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
                            odds.append({"bucket": bucket, "decimal_odds": odds_val})
                            break
                    except ValueError:
                        continue

    return odds


async def scrape_all_margins(headless: bool = True) -> pd.DataFrame:
    """Scrape margin odds for all upcoming basketball games.

    Returns DataFrame with columns:
        scraped_at, source, league, game_date, home_team, away_team, bucket, decimal_odds
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
                        "league": "NBA",  # TODO: detect league from page
                        "game_date": game["game_date"],
                        "home_team": game["home_team"],
                        "away_team": game["away_team"],
                        "bucket": o["bucket"],
                        "decimal_odds": o["decimal_odds"],
                    })

        await browser.close()

    df = pd.DataFrame(all_odds)
    logger.info(f"Scraped {len(df)} total margin odds across {len(games)} games")
    return df


def scrape_odds_sync(headless: bool = True) -> pd.DataFrame:
    """Synchronous wrapper for scrape_all_margins."""
    return asyncio.run(scrape_all_margins(headless=headless))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = scrape_odds_sync(headless=False)
    print(df.to_string())
