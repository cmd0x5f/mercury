"""CLI for the basketball winning margin predictor."""

import logging
import sys

import click

from src.betting.kelly import calculate_stakes
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
@click.option("--seasons", "-s", multiple=True, default=["2022-23", "2023-24", "2024-25"])
def collect(seasons):
    """Fetch NBA game data and store it."""
    store = DataStore()
    df = collect_nba(list(seasons), store)
    click.echo(f"Collected {len(df)} games across {len(seasons)} seasons")


@main.command()
def train():
    """Train the margin prediction model."""
    store = DataStore()
    games = store.get_games(league="NBA")

    if games.empty:
        click.echo("No games in database. Run 'collect' first.")
        sys.exit(1)

    click.echo(f"Training on {len(games)} NBA games...")
    model = MarginModel()
    model.train(games)
    model.save()
    click.echo(f"Model trained (sigma={model.sigma:.2f}) and saved")


@main.command()
def evaluate():
    """Run walk-forward evaluation on historical data."""
    store = DataStore()
    games = store.get_games(league="NBA")

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
    click.echo(f"Scraped {len(df)} odds for {df['home_team'].nunique()} games")


@main.command()
@click.option("--bankroll", "-b", default=10000, help="Current bankroll")
@click.option("--min-edge", "-e", default=0.05, help="Minimum edge threshold")
@click.option("--kelly", "-k", default=0.25, help="Kelly fraction")
@click.option("--date", "-d", default=None, help="Game date YYYY-MM-DD (default: latest)")
def picks(bankroll, min_edge, kelly, date):
    """Show +EV picks by comparing model vs book odds."""
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

    # Get all historical games for feature computation
    games = store.get_games(league="NBA")
    featured = build_features(games)

    # For each game with odds, predict bucket probs
    all_bets = []
    game_groups = odds_df.groupby(["home_team", "away_team"])

    for (home, away), group in game_groups:
        # Map SportsPlus names to nba_api abbreviations
        home_abbr = normalize_team(home)
        away_abbr = normalize_team(away)

        if not home_abbr or not away_abbr:
            logger.debug(f"Non-NBA game: {home} vs {away}, skipping")
            continue

        # Find latest features for each team
        home_games = featured[
            (featured["home_team"] == home_abbr) | (featured["away_team"] == home_abbr)
        ]
        away_games = featured[
            (featured["home_team"] == away_abbr) | (featured["away_team"] == away_abbr)
        ]

        if home_games.empty or away_games.empty:
            logger.warning(f"No history for {home} ({home_abbr}) vs {away} ({away_abbr})")
            continue

        # Get the team's latest Elo/stats regardless of home/away in that game
        lh = home_games.iloc[-1]
        la = away_games.iloc[-1]

        # Extract the correct Elo for the team (might have been home or away)
        h_elo = lh["home_elo"] if lh["home_team"] == home_abbr else lh["away_elo"]
        a_elo = la["away_elo"] if la["away_team"] == away_abbr else la["home_elo"]

        h_avg_m = lh["home_avg_margin"] if lh["home_team"] == home_abbr else lh["away_avg_margin"]
        a_avg_m = la["away_avg_margin"] if la["away_team"] == away_abbr else la["home_avg_margin"]

        h_avg_s = lh["home_avg_scored"] if lh["home_team"] == home_abbr else lh["away_avg_scored"]
        a_avg_s = la["away_avg_scored"] if la["away_team"] == away_abbr else la["home_avg_scored"]

        h_gnum = lh["home_game_num"] if lh["home_team"] == home_abbr else lh["away_game_num"]
        a_gnum = la["away_game_num"] if la["away_team"] == away_abbr else la["home_game_num"]

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
        }

        model_probs = model.predict_single(features)
        book_odds = dict(zip(group["bucket"], group["decimal_odds"]))

        game_info = {
            "game_date": date,
            "league": "NBA",
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
    click.echo(f"\n{'='*85}")
    click.echo(f"  PICKS — {date}  |  Bankroll: ${bankroll:,.0f}  |  Min Edge: {min_edge:.0%}")
    click.echo(f"{'='*85}")
    click.echo(
        f"{'Game':<35} {'Bucket':>6} {'Model%':>7} {'Book%':>7} "
        f"{'Edge':>6} {'EV/unit':>7} {'Stake':>8}"
    )
    click.echo("-" * 85)

    for bet, stake in sized_bets:
        game = f"{bet.home_team} vs {bet.away_team}"
        if len(game) > 34:
            game = game[:31] + "..."
        click.echo(
            f"{game:<35} {bet.bucket:>6} {bet.model_prob:>6.1%} "
            f"{bet.implied_prob:>6.1%} {bet.edge:>+5.1%} "
            f"{bet.ev_per_unit:>+6.2f} ${stake:>7.0f}"
        )

    total_stake = sum(s for _, s in sized_bets)
    click.echo("-" * 85)
    click.echo(f"{'Total exposure:':<55} ${total_stake:>7.0f} ({total_stake/bankroll:.1%})")
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
