"""Kelly criterion stake sizing."""

from src.config import get as cfg


def kelly_stake(
    prob: float,
    decimal_odds: float,
    fraction: float = None,
    min_stake_pct: float = None,
    max_stake_pct: float = None,
) -> float:
    """Calculate stake as fraction of bankroll using Kelly criterion.

    Kelly formula: f* = (bp - q) / b
    where b = decimal_odds - 1, p = win probability, q = 1 - p

    Args:
        prob: model's estimated probability of winning
        decimal_odds: book's decimal odds
        fraction: Kelly fraction (0.25 = quarter Kelly)
        min_stake_pct: minimum stake as % of bankroll
        max_stake_pct: maximum stake as % of bankroll

    Returns:
        Optimal stake as fraction of bankroll (0 if negative EV)
    """
    if fraction is None:
        fraction = cfg("betting", "kelly_fraction", 0.25)
    if min_stake_pct is None:
        min_stake_pct = cfg("betting", "min_stake_pct", 0.005)
    if max_stake_pct is None:
        max_stake_pct = cfg("betting", "max_stake_pct", 0.03)

    b = decimal_odds - 1  # net odds (profit per unit)
    q = 1 - prob

    full_kelly = (b * prob - q) / b

    if full_kelly <= 0:
        return 0.0

    stake = full_kelly * fraction
    return max(min_stake_pct, min(stake, max_stake_pct))


def calculate_stakes(
    bets: list,
    bankroll: float = None,
    fraction: float = None,
    max_exposure_pct: float = None,
) -> list[tuple]:
    """Calculate stakes for a set of bets with exposure limits.

    Returns list of (bet, stake_amount) tuples.
    """
    if bankroll is None:
        bankroll = cfg("betting", "default_bankroll", 10000)
    if fraction is None:
        fraction = cfg("betting", "kelly_fraction", 0.25)
    if max_exposure_pct is None:
        max_exposure_pct = cfg("betting", "max_exposure_pct", 0.15)

    results = []
    total_exposure = 0.0

    for bet in bets:
        pct = kelly_stake(bet.model_prob, bet.decimal_odds, fraction=fraction)
        amount = pct * bankroll

        # Check total exposure limit
        if (total_exposure + amount) / bankroll > max_exposure_pct:
            remaining = max_exposure_pct * bankroll - total_exposure
            if remaining <= 0:
                break
            amount = remaining

        results.append((bet, round(amount, 2)))
        total_exposure += amount

    return results
