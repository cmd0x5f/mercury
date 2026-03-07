"""Calculate expected value and identify +EV bets."""

from dataclasses import dataclass

from src.config import get as cfg


@dataclass
class BetOpportunity:
    game_date: str
    league: str
    home_team: str
    away_team: str
    bucket: str
    model_prob: float
    decimal_odds: float

    @property
    def implied_prob(self) -> float:
        return 1.0 / self.decimal_odds

    @property
    def edge(self) -> float:
        return self.model_prob - self.implied_prob

    @property
    def ev_per_unit(self) -> float:
        """Expected value per unit staked."""
        return self.model_prob * (self.decimal_odds - 1) - (1 - self.model_prob)

    @property
    def is_positive_ev(self) -> bool:
        return self.edge > 0


def find_value_bets(
    model_probs: dict[str, float],
    odds: dict[str, float],
    game_info: dict,
    min_edge: float = None,
) -> list[BetOpportunity]:
    """Compare model probabilities to book odds and find +EV bets.

    Args:
        model_probs: {bucket: probability} from model
        odds: {bucket: decimal_odds} from scraper
        game_info: {game_date, league, home_team, away_team}
        min_edge: minimum edge threshold to flag as a bet

    Returns:
        list of BetOpportunity with edge >= min_edge
    """
    if min_edge is None:
        min_edge = cfg("betting", "edge_threshold", 0.05)

    bets = []
    for bucket in model_probs:
        if bucket not in odds:
            continue

        bet = BetOpportunity(
            game_date=game_info["game_date"],
            league=game_info.get("league", "NBA"),
            home_team=game_info["home_team"],
            away_team=game_info["away_team"],
            bucket=bucket,
            model_prob=model_probs[bucket],
            decimal_odds=odds[bucket],
        )

        if bet.edge >= min_edge:
            bets.append(bet)

    return sorted(bets, key=lambda b: b.edge, reverse=True)
