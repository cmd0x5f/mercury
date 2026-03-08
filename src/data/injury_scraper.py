"""Scrape NBA injury reports for player availability.

Primary source: Rotowire JSON API (most up-to-date, updated throughout the day).
Fallback: Basketball Reference HTML table (simple, reliable).
Manual: CLI --out flag for user-specified missing players.
"""

import json
import logging
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger(__name__)

ROTOWIRE_URL = "https://www.rotowire.com/basketball/tables/injury-report.php?team=ALL&pos=ALL"
BBALL_REF_URL = "https://www.basketball-reference.com/friv/injuries.fcgi"

# Statuses that mean a player is definitely OUT
OUT_STATUSES = {"out", "out for season", "out indefinitely"}
# Statuses that are uncertain — treat as out for conservative modeling
QUESTIONABLE_STATUSES = {"day to day", "questionable", "doubtful"}


@dataclass
class InjuryEntry:
    player_name: str
    team: str  # nba_api abbreviation (e.g., "LAL")
    status: str  # "out", "questionable", etc.
    injury: str  # body part / description


def fetch_injuries_rotowire() -> list[InjuryEntry]:
    """Fetch injury data from Rotowire's JSON API (primary source).

    Returns JSON with fields: player, team (abbreviation), position,
    injury (body part), status (Out/Questionable/etc).
    """
    try:
        req = urllib.request.Request(
            ROTOWIRE_URL,
            headers={"User-Agent": "Mozilla/5.0 MercuryBot/1.0"},
        )
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read())
    except Exception as e:
        logger.warning(f"Failed to fetch Rotowire injuries: {e}")
        return []

    entries = []
    for item in data:
        entries.append(InjuryEntry(
            player_name=item.get("player", ""),
            team=item.get("team", ""),
            status=item.get("status", "").lower(),
            injury=item.get("injury", ""),
        ))

    logger.info(f"Rotowire: {len(entries)} injury entries fetched")
    return entries


def fetch_injuries_bball_ref() -> list[InjuryEntry]:
    """Fallback: scrape Basketball Reference injury table (simple HTML)."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("beautifulsoup4 not installed — Basketball Reference fallback unavailable")
        return []

    from src.data.team_names import SPORTSPLUS_TO_NBA

    try:
        req = urllib.request.Request(
            BBALL_REF_URL,
            headers={"User-Agent": "Mozilla/5.0 MercuryBot/1.0"},
        )
        resp = urllib.request.urlopen(req, timeout=15)
        html = resp.read()
    except Exception as e:
        logger.warning(f"Failed to fetch Basketball Reference injuries: {e}")
        return []

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        logger.warning("No injury table found on Basketball Reference")
        return []

    entries = []
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue

        player_name = cells[0].get_text(strip=True)
        team_full = cells[1].get_text(strip=True)
        description = cells[3].get_text(strip=True)

        team_abbr = SPORTSPLUS_TO_NBA.get(team_full)
        if not team_abbr:
            continue

        status = _parse_bball_ref_status(description)
        entries.append(InjuryEntry(
            player_name=player_name,
            team=team_abbr,
            status=status,
            injury=description[:80],
        ))

    logger.info(f"Basketball Reference: {len(entries)} injury entries scraped")
    return entries


def _parse_bball_ref_status(description: str) -> str:
    """Extract status from Basketball Reference description text."""
    d = description.lower()
    if "out for season" in d:
        return "out for season"
    if "out indefinitely" in d:
        return "out indefinitely"
    if "ruled out" in d or "listed as out" in d or "did not play" in d or "did not return" in d:
        return "out"
    if "doubtful" in d:
        return "doubtful"
    if "questionable" in d:
        return "questionable"
    if "day to day" in d or "day-to-day" in d:
        return "day to day"
    if "probable" in d:
        return "probable"
    return "unknown"


def get_injured_players(
    include_questionable: bool = True,
    manual_out: list[str] | None = None,
) -> dict[str, list[str]]:
    """Get currently injured/out players grouped by team abbreviation.

    Tries Rotowire first, falls back to Basketball Reference.

    Args:
        include_questionable: If True, treat Questionable/Doubtful as out.
        manual_out: Additional player names to mark as out (from CLI --out flag).

    Returns:
        {team_abbr: [player_name, ...]} for players who are out.
    """
    # Try primary source (Rotowire)
    entries = fetch_injuries_rotowire()

    # Fallback to Basketball Reference
    if not entries:
        logger.info("Rotowire failed, trying Basketball Reference fallback...")
        entries = fetch_injuries_bball_ref()

    # Filter to players who are actually out
    out_players: dict[str, list[str]] = {}
    valid_statuses = OUT_STATUSES.copy()
    if include_questionable:
        valid_statuses |= QUESTIONABLE_STATUSES

    for entry in entries:
        if entry.status in valid_statuses and entry.team:
            out_players.setdefault(entry.team, []).append(entry.player_name)

    # Add manual overrides
    if manual_out:
        _add_manual_players(out_players, manual_out)

    total = sum(len(v) for v in out_players.values())
    logger.info(f"Total players marked as out: {total} across {len(out_players)} teams")
    return out_players


def _add_manual_players(
    out_players: dict[str, list[str]],
    manual_names: list[str],
) -> None:
    """Add manually specified players to the out list.

    Resolves team from the player logs DB.
    """
    from src.data.data_store import DataStore

    store = DataStore()
    player_logs = store.get_player_logs()

    for name in manual_names:
        name = name.strip()
        if not name:
            continue

        # Check if already in injury list (avoid duplicates)
        already_listed = any(
            name.lower() in [p.lower() for p in players]
            for players in out_players.values()
        )
        if already_listed:
            logger.debug(f"Manual out: {name} already in injury report, skipping")
            continue

        # Try to find team from player logs
        if not player_logs.empty:
            matches = player_logs[
                player_logs["player_name"].str.contains(name, case=False, na=False)
            ]
            if not matches.empty:
                team = matches.sort_values("date").iloc[-1]["team"]
                out_players.setdefault(team, []).append(name)
                logger.info(f"Manual out: {name} ({team})")
                continue

        logger.warning(f"Manual out: {name} — team unknown, won't affect impact calculation")
