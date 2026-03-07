"""CLI for the basketball winning margin predictor."""

import logging
import sys

import click

from src.betting.kelly import calculate_stakes
from src.config import get as cfg
from src.betting.tracker import BetTracker
from src.betting.value_calculator import find_value_bets
from src.data.data_store import DataStore
from src.data.nba_collector import collect_nba
from src.data.sportsplus_scraper import scrape_odds_sync
from src.data.team_names import normalize_team
from src.features.builder import build_features
from src.model.margin_model import MarginModel

logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def main(verbose):
    """Basketball Winning Margin Predictor — find +EV bets."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@main.command()
def collect():
    """Fetch NBA game data (auto-detects seasons) + latest from Flashscore."""
    from src.data.nba_collector import generate_seasons

    store = DataStore()
    seasons = generate_seasons()
    click.echo(f"Collecting NBA seasons: {', '.join(seasons)}")
    df = collect_nba(seasons, store)
    click.echo(f"NBA: {len(df)} total games in DB (latest: {df['date'].max()})")


@main.command("collect-leagues")
@click.option("--headless/--no-headless", default=True)
@click.option("--clicks", "-c", default=20, help="Max 'Show more' clicks per league (stops early when exhausted)")
@click.option("--parallel", "-p", default=1, help="Number of leagues to scrape concurrently")
@click.option("--seasons", "-s", default=0, help="Number of past seasons to also scrape (0=current only)")
@click.option("--incremental/--full", default=False, help="Skip re-scraping games already in DB")
def collect_leagues(headless, clicks, parallel, seasons, incremental):
    """Scrape historical results from Flashscore for all mapped leagues."""
    from src.data.flashscore_scraper import scrape_leagues_sync
    from src.data.league_matcher import LeagueMatcher

    store = DataStore()
    matcher = LeagueMatcher()
    urls = matcher.get_all_flashscore_urls()

    if not urls:
        click.echo("No Flashscore URLs configured in league_mappings.yaml")
        sys.exit(1)

    mode = "incremental" if incremental else "full"
    season_msg = f" + {seasons} past season(s)" if seasons else ""
    para_msg = f" ({parallel}x parallel)" if parallel > 1 else ""
    click.echo(f"Scraping {len(urls)} leagues from Flashscore [{mode}]{season_msg}{para_msg}...")

    df = scrape_leagues_sync(
        urls, headless=headless, max_clicks=clicks, store=store,
        concurrency=parallel, seasons=seasons, incremental=incremental,
    )

    if df.empty:
        click.echo("No games scraped")
    else:
        leagues = df["league"].nunique()
        click.echo(f"Collected {len(df)} games across {leagues} leagues")


@main.command()
@click.option("--league", "-l", default=None, help="Train on single league (default: all)")
def train(league):
    """Train the margin prediction model."""
    store = DataStore()
    games = store.get_games(league=league)

    if games.empty:
        click.echo("No games in database. Run 'collect' or 'collect-leagues' first.")
        sys.exit(1)

    leagues = games["league"].nunique()
    click.echo(f"Training on {len(games)} games across {leagues} league(s)...")
    model = MarginModel()
    model.train(games)
    model.save()
    click.echo(f"Model trained (global σ={model.sigma:.2f}) and saved")
    if model.league_sigmas:
        inv = {v: k for k, v in model.league_categories.items()}
        for lid, s in model.league_sigmas.items():
            click.echo(f"  {inv.get(lid, lid)}: σ={s:.2f}")


@main.command()
@click.option("--league", "-l", default=None, help="Evaluate single league (default: all)")
def evaluate(league):
    """Run walk-forward evaluation on historical data."""
    store = DataStore()
    games = store.get_games(league=league)

    if games.empty:
        click.echo("No games in database. Run 'collect' first.")
        sys.exit(1)

    click.echo(f"Evaluating on {len(games)} games with walk-forward CV...")
    model = MarginModel()
    results = model.evaluate(games)

    click.echo("\nResults:")
    click.echo(f"  MAE:  {results['mae']:.2f} points")
    click.echo(f"  RMSE: {results['rmse']:.2f} points")
    click.echo(f"  σ:    {results['sigma']:.2f}")
    click.echo("\nBucket accuracy (most-likely bucket correct):")
    for bucket, acc in results["bucket_accuracy"].items():
        click.echo(f"  {bucket:>5}: {acc:.1%}")


@main.command()
@click.option("--headless/--no-headless", default=True)
def scrape(headless):
    """Scrape winning margin odds from SportsPlus."""
    store = DataStore()
    df = scrape_odds_sync(headless=headless)

    if df.empty:
        click.echo("No odds found")
        return

    store.upsert_odds(df)
    leagues = df["league"].nunique()
    games_count = df["home_team"].nunique()
    click.echo(f"Scraped {len(df)} odds for {games_count} games across {leagues} leagues")


@main.command()
@click.option("--bankroll", "-b", default=cfg("betting", "default_bankroll", 10000),
              help="Current bankroll")
@click.option("--min-edge", "-e", default=cfg("betting", "edge_threshold", 0.05),
              help="Minimum edge threshold")
@click.option("--kelly", "-k", default=cfg("betting", "kelly_fraction", 0.25),
              help="Kelly fraction")
@click.option("--date", "-d", default=None, help="Game date YYYY-MM-DD (default: latest)")
def picks(bankroll, min_edge, kelly, date):
    """Show +EV picks by comparing model vs book odds."""
    from src.data.league_matcher import LeagueMatcher, TeamMatcher

    store = DataStore()
    model = MarginModel()

    try:
        model.load()
    except FileNotFoundError:
        click.echo("No trained model found. Run 'train' first.")
        sys.exit(1)

    # Get odds for the specified date, or find the latest date with odds
    if date:
        odds_df = store.get_odds(game_date=date)
    else:
        all_odds = store.get_odds()
        if all_odds.empty:
            click.echo("No odds in database. Run 'scrape' first.")
            sys.exit(1)
        date = all_odds["game_date"].max()
        odds_df = all_odds[all_odds["game_date"] == date]

    if odds_df.empty:
        click.echo(f"No odds for {date}. Run 'scrape' first.")
        sys.exit(1)

    # Set up league/team matching for international games
    league_matcher = LeagueMatcher()
    team_matcher = TeamMatcher()

    # Get all historical games and build features
    all_games = store.get_games()
    if all_games.empty:
        click.echo("No games in database. Run 'collect' or 'collect-leagues' first.")
        sys.exit(1)

    featured = build_features(all_games)

    # Register known teams per league from historical data
    for lg, group in all_games.groupby("league"):
        teams = list(set(group["home_team"].tolist() + group["away_team"].tolist()))
        team_matcher.register_teams(lg, teams)

    # For each game with odds, predict bucket probs
    all_bets = []
    game_groups = odds_df.groupby(["home_team", "away_team", "league"])

    for (home, away, sp_league), group in game_groups:
        # Match league from SportsPlus to our canonical name
        canonical_league = league_matcher.match_league(sp_league)

        # NBA uses nba_api abbreviations — map via team_names
        if canonical_league == "NBA" or (not canonical_league and normalize_team(home)):
            canonical_league = "NBA"
            home_mapped = normalize_team(home)
            away_mapped = normalize_team(away)
        elif canonical_league:
            # International: fuzzy match team names to Flashscore data
            home_mapped, away_mapped = team_matcher.match_game(home, away, canonical_league)
        else:
            logger.debug(f"Unknown league '{sp_league}' for {home} vs {away}, skipping")
            continue

        if not home_mapped or not away_mapped:
            logger.debug(f"Can't map teams: {home} vs {away} ({canonical_league})")
            continue

        # Look up league_id from model
        league_id = model.get_league_id(canonical_league)

        # Find latest features for each team
        home_games = featured[
            (featured["home_team"] == home_mapped) | (featured["away_team"] == home_mapped)
        ]
        away_games = featured[
            (featured["home_team"] == away_mapped) | (featured["away_team"] == away_mapped)
        ]

        if home_games.empty or away_games.empty:
            logger.warning(f"No history for {home} ({home_mapped}) vs {away} ({away_mapped})")
            continue

        # Get the team's latest Elo/stats regardless of home/away in that game
        lh = home_games.iloc[-1]
        la = away_games.iloc[-1]

        h_elo = lh["home_elo"] if lh["home_team"] == home_mapped else lh["away_elo"]
        a_elo = la["away_elo"] if la["away_team"] == away_mapped else la["home_elo"]

        h_avg_m = lh["home_avg_margin"] if lh["home_team"] == home_mapped else lh["away_avg_margin"]
        a_avg_m = la["away_avg_margin"] if la["away_team"] == away_mapped else la["home_avg_margin"]

        h_avg_s = lh["home_avg_scored"] if lh["home_team"] == home_mapped else lh["away_avg_scored"]
        a_avg_s = la["away_avg_scored"] if la["away_team"] == away_mapped else la["home_avg_scored"]

        h_gnum = lh["home_game_num"] if lh["home_team"] == home_mapped else lh["away_game_num"]
        a_gnum = la["away_game_num"] if la["away_team"] == away_mapped else la["home_game_num"]

        features = {
            "home_elo": h_elo,
            "away_elo": a_elo,
            "elo_diff": h_elo + 100 - a_elo,
            "home_avg_margin": h_avg_m,
            "away_avg_margin": a_avg_m,
            "home_avg_scored": h_avg_s,
            "away_avg_scored": a_avg_s,
            "home_rest_days": 2,
            "away_rest_days": 2,
            "home_game_num": h_gnum,
            "away_game_num": a_gnum,
            "is_b2b_home": 0,
            "is_b2b_away": 0,
            "league_id": league_id,
        }

        model_probs = model.predict_single(features, league_id=league_id)
        book_odds = dict(zip(group["bucket"], group["decimal_odds"]))

        game_info = {
            "game_date": date,
            "league": canonical_league,
            "home_team": home,
            "away_team": away,
        }

        bets = find_value_bets(model_probs, book_odds, game_info, min_edge=min_edge)
        all_bets.extend(bets)

    if not all_bets:
        click.echo("No +EV bets found today.")
        return

    # Calculate stakes
    sized_bets = calculate_stakes(all_bets, bankroll, fraction=kelly)

    # Display
    click.echo(f"\n{'='*90}")
    click.echo(f"  PICKS — {date}  |  Bankroll: ${bankroll:,.0f}  |  Min Edge: {min_edge:.0%}")
    click.echo(f"{'='*90}")
    click.echo(
        f"{'Game':<30} {'League':>10} {'Bucket':>6} {'Model%':>7} {'Book%':>7} "
        f"{'Edge':>6} {'EV/unit':>7} {'Stake':>8}"
    )
    click.echo("-" * 90)

    for bet, stake in sized_bets:
        game = f"{bet.home_team} vs {bet.away_team}"
        if len(game) > 29:
            game = game[:26] + "..."
        league_short = bet.league[:10] if bet.league else "?"
        click.echo(
            f"{game:<30} {league_short:>10} {bet.bucket:>6} {bet.model_prob:>6.1%} "
            f"{bet.implied_prob:>6.1%} {bet.edge:>+5.1%} "
            f"{bet.ev_per_unit:>+6.2f} ${stake:>7.0f}"
        )

    total_stake = sum(s for _, s in sized_bets)
    click.echo("-" * 90)
    click.echo(f"{'Total exposure:':<60} ${total_stake:>7.0f} ({total_stake/bankroll:.1%})")
    click.echo()


@main.command()
def status():
    """Show bet tracking summary."""
    tracker = BetTracker()
    s = tracker.summary()

    if s["total_bets"] == 0:
        click.echo("No settled bets yet.")
        return

    click.echo("\nBet Tracking Summary")
    click.echo(f"  Total bets:  {s['total_bets']} ({s['wins']}W / {s['losses']}L)")
    click.echo(f"  Pending:     {s['pending']}")
    click.echo(f"  Total staked: ${s['total_staked']:,.0f}")
    click.echo(f"  P&L:         ${s['pnl']:+,.0f}")
    click.echo(f"  ROI:         {s['roi']:+.1%}")
    click.echo(f"  Avg edge:    {s['avg_edge']:.1%}")
    click.echo(f"  Avg odds:    {s['avg_odds']:.2f}")


if __name__ == "__main__":
    main()
