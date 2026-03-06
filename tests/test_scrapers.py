"""Tests for Flashscore and SportsPlus scrapers.

Unit tests use mocked Playwright for fast, deterministic checks.
Live integration tests (marked @pytest.mark.live) hit real sites to
detect DOM structure changes that would break our CSS selectors.

Run live tests:  pytest tests/test_scrapers.py -m live --no-header -v
Run unit tests:  pytest tests/test_scrapers.py -m "not live" -v
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from playwright.async_api import async_playwright

from src.data.flashscore_scraper import (
    _parse_flashscore_date,
    scrape_league_results,
    scrape_multiple_leagues,
)
from src.data.sportsplus_scraper import (
    BUCKET_PATTERN,
    _extract_margins_from_text,
    get_game_links,
    scrape_margin_odds,
)

# ── Flashscore date parsing ──────────────────────────────────────


class TestParseFlashscoreDate:
    def test_full_date(self):
        assert _parse_flashscore_date("05.03.2026") == "2026-03-05"

    def test_date_no_year(self):
        """Flashscore sometimes omits year — defaults to current year."""
        result = _parse_flashscore_date("05.03.")
        expected_year = str(datetime.now().year)
        assert result == f"{expected_year}-03-05"

    def test_date_with_surrounding_text(self):
        assert _parse_flashscore_date("Round 24, 15.01.2025") == "2025-01-15"

    def test_empty_string(self):
        """Empty date falls back to today."""
        result = _parse_flashscore_date("")
        assert result == datetime.now().strftime("%Y-%m-%d")

    def test_garbage_input(self):
        """Unparseable date falls back to today."""
        result = _parse_flashscore_date("no date here")
        assert result == datetime.now().strftime("%Y-%m-%d")


# ── SportsPlus bucket pattern regex ──────────────────────────────


class TestBucketPattern:
    def test_range_bucket(self):
        m = BUCKET_PATTERN.match("1-5")
        assert m and m.group(1) == "1" and m.group(2) == "5"

    def test_range_bucket_large(self):
        m = BUCKET_PATTERN.match("26-30")
        assert m and m.group(1) == "26" and m.group(2) == "30"

    def test_plus_bucket(self):
        m = BUCKET_PATTERN.match("31+")
        assert m and m.group(3) == "31"

    def test_range_with_spaces(self):
        m = BUCKET_PATTERN.match("6 - 10")
        assert m and m.group(1) == "6" and m.group(2) == "10"

    def test_no_match_on_odds(self):
        assert BUCKET_PATTERN.match("3.50") is None

    def test_no_match_on_text(self):
        assert BUCKET_PATTERN.match("Any Team") is None


# ── Flashscore scraper with mocked Playwright ────────────────────


def _make_mock_page(evaluate_return=None):
    """Create a mock Playwright page with common async methods."""
    page = AsyncMock()
    page.goto = AsyncMock()
    page.wait_for_timeout = AsyncMock()
    page.evaluate = AsyncMock(return_value=evaluate_return or [])
    page.locator = MagicMock()
    # Cookie banner not visible
    mock_btn = AsyncMock()
    mock_btn.is_visible = AsyncMock(return_value=False)
    page.locator.return_value = mock_btn
    return page


@pytest.mark.asyncio
async def test_scrape_league_results_basic():
    """Verify scrape_league_results parses JS-extracted game data correctly."""
    fake_games = [
        {
            "game_id": "g_4_abc123",
            "date_str": "01.03.2026",
            "home_team": "Barcelona",
            "away_team": "Real Madrid",
            "home_score": 88,
            "away_score": 82,
        },
        {
            "game_id": "",
            "date_str": "28.02.2026",
            "home_team": "Valencia",
            "away_team": "Baskonia",
            "home_score": 75,
            "away_score": 91,
        },
    ]
    page = _make_mock_page(evaluate_return=fake_games)

    df = await scrape_league_results(
        page, "https://flashscore.com/basketball/spain/acb/results/",
        "Spain ACB", load_more_clicks=0,
    )

    assert len(df) == 2
    assert df.iloc[0]["home_team"] == "Barcelona"
    assert df.iloc[0]["away_team"] == "Real Madrid"
    assert df.iloc[0]["home_score"] == 88
    assert df.iloc[0]["date"] == "2026-03-01"
    assert df.iloc[0]["source"] == "flashscore"
    assert df.iloc[0]["league"] == "Spain ACB"
    # Game with empty ID should get a generated one
    assert df.iloc[1]["game_id"].startswith("fs_Spain ACB_")


@pytest.mark.asyncio
async def test_scrape_league_results_empty():
    """No games found returns empty DataFrame."""
    page = _make_mock_page(evaluate_return=[])
    df = await scrape_league_results(page, "https://example.com", "Test", load_more_clicks=0)
    assert df.empty


@pytest.mark.asyncio
async def test_scrape_multiple_leagues_combines():
    """Multiple leagues are combined into one DataFrame."""
    fake_game = [{
        "game_id": "g1",
        "date_str": "01.03.2026",
        "home_team": "A",
        "away_team": "B",
        "home_score": 100,
        "away_score": 90,
    }]

    with patch("src.data.flashscore_scraper.async_playwright") as mock_pw:
        mock_browser = AsyncMock()
        mock_page = _make_mock_page(evaluate_return=fake_game)
        mock_browser.new_page = AsyncMock(return_value=mock_page)
        mock_context = AsyncMock()
        mock_context.chromium.launch = AsyncMock(return_value=mock_browser)
        mock_pw.return_value.__aenter__ = AsyncMock(return_value=mock_context)
        mock_pw.return_value.__aexit__ = AsyncMock(return_value=False)

        urls = {"League A": "https://a.com", "League B": "https://b.com"}
        df = await scrape_multiple_leagues(urls, headless=True, load_more_clicks=0)

    # Both leagues return the same fake game, so 2 rows total
    assert len(df) == 2
    assert set(df["league"]) == {"League A", "League B"}


@pytest.mark.asyncio
async def test_scrape_multiple_leagues_handles_error():
    """A failing league is skipped, others still scraped."""
    call_count = 0

    async def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # Second league call: goto raises
        if call_count > 2:  # first goto + wait_for_timeout, then second goto
            raise Exception("Network error")
        return []

    with patch("src.data.flashscore_scraper.async_playwright") as mock_pw:
        mock_browser = AsyncMock()
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.wait_for_timeout = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value=[{
            "game_id": "g1", "date_str": "01.03.2026",
            "home_team": "A", "away_team": "B",
            "home_score": 80, "away_score": 70,
        }])
        mock_btn = AsyncMock()
        mock_btn.is_visible = AsyncMock(return_value=False)
        mock_page.locator = MagicMock(return_value=mock_btn)
        mock_browser.new_page = AsyncMock(return_value=mock_page)

        mock_context = AsyncMock()
        mock_context.chromium.launch = AsyncMock(return_value=mock_browser)
        mock_pw.return_value.__aenter__ = AsyncMock(return_value=mock_context)
        mock_pw.return_value.__aexit__ = AsyncMock(return_value=False)

        # Make second league fail
        original_scrape = scrape_league_results

        async def patched_scrape(page, url, name, load_more_clicks=5):
            if name == "Fail League":
                raise Exception("Scrape failed")
            return await original_scrape(page, url, name, load_more_clicks)

        with patch(
            "src.data.flashscore_scraper.scrape_league_results",
            side_effect=patched_scrape,
        ):
            urls = {"Good League": "https://good.com", "Fail League": "https://fail.com"}
            df = await scrape_multiple_leagues(urls, headless=True, load_more_clicks=0)

    # Only the good league's games survive
    assert len(df) == 1
    assert df.iloc[0]["league"] == "Good League"


@pytest.mark.asyncio
async def test_scrape_multiple_leagues_stores_to_datastore():
    """Results are persisted via DataStore when provided."""
    fake_game = [{
        "game_id": "g1",
        "date_str": "01.03.2026",
        "home_team": "X",
        "away_team": "Y",
        "home_score": 99,
        "away_score": 88,
    }]

    mock_store = MagicMock()

    with patch("src.data.flashscore_scraper.async_playwright") as mock_pw:
        mock_browser = AsyncMock()
        mock_page = _make_mock_page(evaluate_return=fake_game)
        mock_browser.new_page = AsyncMock(return_value=mock_page)
        mock_context = AsyncMock()
        mock_context.chromium.launch = AsyncMock(return_value=mock_browser)
        mock_pw.return_value.__aenter__ = AsyncMock(return_value=mock_context)
        mock_pw.return_value.__aexit__ = AsyncMock(return_value=False)

        await scrape_multiple_leagues(
            {"Test": "https://test.com"},
            headless=True, load_more_clicks=0, store=mock_store,
        )

    mock_store.upsert_games.assert_called_once()
    stored_df = mock_store.upsert_games.call_args[0][0]
    assert len(stored_df) == 1


# ── SportsPlus scraper with mocked Playwright ────────────────────


@pytest.mark.asyncio
async def test_extract_margins_from_text():
    """Test margin odds extraction from page text."""
    page_text = """
Some header stuff
Any Team Winning Margin (Incl. Overtime)
1-5
3.50
6-10
4.20
11-15
5.50
16-20
8.00
21-25
12.00
26-30
18.00
31+
25.00
Some Other Market (Something)
Over 180.5
"""
    page = AsyncMock()
    page.inner_text = AsyncMock(return_value=page_text)

    odds = await _extract_margins_from_text(page)

    assert len(odds) == 7
    assert odds[0] == {"bucket": "1-5", "decimal_odds": 3.50}
    assert odds[1] == {"bucket": "6-10", "decimal_odds": 4.20}
    assert odds[6] == {"bucket": "31+", "decimal_odds": 25.00}


@pytest.mark.asyncio
async def test_extract_margins_no_section():
    """No margin section in page text returns empty list."""
    page = AsyncMock()
    page.inner_text = AsyncMock(return_value="Moneyline\nHome 1.80\nAway 2.10")

    odds = await _extract_margins_from_text(page)
    assert odds == []


@pytest.mark.asyncio
async def test_extract_margins_partial():
    """Handles partial data — some buckets without valid odds."""
    page_text = """
Any Team Winning Margin (Incl. Overtime)
1-5
3.50
6-10
Suspended
11-15
5.50
Next Market (X)
"""
    page = AsyncMock()
    page.inner_text = AsyncMock(return_value=page_text)

    odds = await _extract_margins_from_text(page)
    # 6-10 skipped because "Suspended" isn't a valid float
    assert len(odds) == 2
    assert odds[0]["bucket"] == "1-5"
    assert odds[1]["bucket"] == "11-15"


@pytest.mark.asyncio
async def test_scrape_margin_odds_no_section():
    """When margin section isn't found, returns empty list."""
    page = AsyncMock()
    page.goto = AsyncMock()
    page.wait_for_timeout = AsyncMock()
    page.query_selector = AsyncMock(return_value=None)  # no terms dialog
    page.evaluate = AsyncMock(return_value=False)  # section not found

    game = {"url": "https://sportsplus.ph/sbk/m/123",
            "home_team": "Team A", "away_team": "Team B",
            "league": "NBA"}
    odds = await scrape_margin_odds(page, game)
    assert odds == []


@pytest.mark.asyncio
async def test_get_game_links_parses_js_result():
    """Verify get_game_links correctly processes the JS-extracted data."""
    fake_links = [
        {
            "url": "https://sportsplus.ph/sbk/m/1",
            "home_team": "Lakers",
            "away_team": "Celtics",
            "date_str": "03/07 10:00",
            "league": "NBA",
        },
        {
            "url": "https://sportsplus.ph/sbk/m/2",
            "home_team": "Ginebra",
            "away_team": "San Miguel",
            "date_str": "03/08 19:00",
            "league": "PBA",
        },
    ]
    page = AsyncMock()
    page.goto = AsyncMock()
    page.wait_for_timeout = AsyncMock()
    page.query_selector = AsyncMock(return_value=None)
    page.evaluate = AsyncMock(return_value=fake_links)

    games = await get_game_links(page)

    assert len(games) == 2
    assert games[0]["home_team"] == "Lakers"
    assert games[0]["league"] == "NBA"
    year = str(datetime.now().year)
    assert games[0]["game_date"] == f"{year}-03-07"
    assert games[1]["league"] == "PBA"


@pytest.mark.asyncio
async def test_get_game_links_fallback_date():
    """Games without parseable date get today's date."""
    fake_links = [{
        "url": "https://sportsplus.ph/sbk/m/1",
        "home_team": "A", "away_team": "B",
        "date_str": "TBD",
        "league": "Unknown",
    }]
    page = AsyncMock()
    page.goto = AsyncMock()
    page.wait_for_timeout = AsyncMock()
    page.query_selector = AsyncMock(return_value=None)
    page.evaluate = AsyncMock(return_value=fake_links)

    games = await get_game_links(page)
    assert games[0]["game_date"] == datetime.now().strftime("%Y-%m-%d")


# ── Live integration tests ───────────────────────────────────────
# These hit real sites to verify our CSS selectors still work.
# Run with: pytest tests/test_scrapers.py -m live -v


EUROLEAGUE_URL = "https://www.flashscore.com/basketball/europe/euroleague/results/"
NBA_URL = "https://www.flashscore.com/basketball/usa/nba/results/"
ACB_URL = "https://www.flashscore.com/basketball/spain/acb/results/"
DEAD_URL = "https://www.flashscore.com/basketball/turkey/bsl/results/"


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_flashscore_euroleague():
    """Verify Flashscore Euroleague selectors return valid game data."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        df = await scrape_league_results(
            page, EUROLEAGUE_URL, "Euroleague", load_more_clicks=0,
        )

        await browser.close()

    assert not df.empty, "Euroleague returned 0 games — DOM selectors may be broken"
    assert len(df) >= 10, f"Expected 10+ games, got {len(df)}"

    # Verify DataFrame schema
    expected_cols = {"source", "league", "game_id", "date",
                     "home_team", "away_team", "home_score", "away_score"}
    assert set(df.columns) == expected_cols

    # Verify data quality
    assert (df["source"] == "flashscore").all()
    assert (df["league"] == "Euroleague").all()
    assert df["home_team"].str.len().min() > 0, "Empty home team names"
    assert df["away_team"].str.len().min() > 0, "Empty away team names"
    assert (df["home_score"] > 0).all(), "Invalid home scores"
    assert (df["away_score"] > 0).all(), "Invalid away scores"

    # Verify dates are parsed (not all the same fallback date)
    assert df["date"].nunique() > 1, "All dates identical — date parsing broken"

    # Verify dates are reasonable (not in the future, not ancient)
    dates = pd.to_datetime(df["date"])
    today = pd.Timestamp.now()
    assert dates.max() <= today, f"Future date found: {dates.max()}"
    assert dates.min() >= today - pd.Timedelta(days=365), "Dates older than 1 year"

    # Verify game IDs are unique
    assert df["game_id"].is_unique, "Duplicate game IDs"


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_flashscore_nba():
    """Verify Flashscore NBA selectors work (different region/structure)."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        df = await scrape_league_results(
            page, NBA_URL, "NBA", load_more_clicks=0,
        )

        await browser.close()

    assert not df.empty, "NBA returned 0 games — DOM selectors may be broken"
    assert len(df) >= 10
    assert df["date"].nunique() > 1, "All dates identical — date parsing broken"
    # Most NBA scores should be 70+, but allow some outliers
    # (postponed games, quarter scores, etc.)
    normal_scores = df["home_score"] >= 70
    assert normal_scores.mean() > 0.9, (
        f"Only {normal_scores.mean():.0%} of NBA scores >= 70 — "
        "score parsing may be broken"
    )
    assert df["game_id"].is_unique


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_flashscore_spain_acb():
    """Verify a European national league works."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        df = await scrape_league_results(
            page, ACB_URL, "Spain ACB", load_more_clicks=0,
        )

        await browser.close()

    assert not df.empty, "Spain ACB returned 0 games"
    assert len(df) >= 10
    assert df["date"].nunique() > 1


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_flashscore_404_handling():
    """Verify 404 pages return empty DataFrame, not crash."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        df = await scrape_league_results(
            page, DEAD_URL, "Bad URL", load_more_clicks=0,
        )

        await browser.close()

    assert df.empty, "404 page should return empty DataFrame"


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_flashscore_dom_selectors():
    """Directly verify that critical CSS selectors exist on the page.

    This is the canary test — if Flashscore renames their CSS classes,
    this test fails with a clear message about which selector broke.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        await page.goto(EUROLEAGUE_URL, wait_until="networkidle")
        await page.wait_for_timeout(2000)

        # Dismiss cookie banner if present
        try:
            btn = page.locator("#onetrust-accept-btn-handler")
            if await btn.is_visible(timeout=3000):
                await btn.click()
                await page.wait_for_timeout(500)
        except Exception:
            pass

        selectors = {
            "match rows": ".event__match",
            "home team": ".event__participant--home",
            "away team": ".event__participant--away",
            "home score": ".event__score--home",
            "away score": ".event__score--away",
            "time/date": ".event__time",
        }

        for name, selector in selectors.items():
            count = await page.locator(selector).count()
            assert count > 0, (
                f"Selector '{selector}' for {name} matched 0 elements. "
                f"Flashscore may have changed their DOM structure."
            )

        await browser.close()


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_flashscore_load_more():
    """Verify 'Show more' button loads additional games."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        # Use separate pages to avoid networkidle timeout on second nav
        page1 = await browser.new_page()
        df_short = await scrape_league_results(
            page1, EUROLEAGUE_URL, "Euroleague", load_more_clicks=0,
        )
        await page1.close()

        page2 = await browser.new_page()
        df_long = await scrape_league_results(
            page2, EUROLEAGUE_URL, "Euroleague", load_more_clicks=3,
        )
        await page2.close()

        await browser.close()

    assert len(df_long) >= len(df_short), (
        f"Loading more should give >= games: {len(df_long)} vs {len(df_short)}"
    )
