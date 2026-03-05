"""Bet tracking and P&L reporting."""

from datetime import datetime

from src.data.data_store import DataStore


class BetTracker:
    def __init__(self, store: DataStore = None):
        self.store = store or DataStore()

    def record(self, bet, stake: float):
        """Record a placed bet."""
        self.store.record_bet({
            "placed_at": datetime.now().isoformat(),
            "game_date": bet.game_date,
            "league": bet.league,
            "home_team": bet.home_team,
            "away_team": bet.away_team,
            "bucket": bet.bucket,
            "decimal_odds": bet.decimal_odds,
            "model_prob": bet.model_prob,
            "edge": bet.edge,
            "stake": stake,
        })

    def settle(self, bet_id: int, actual_abs_margin: int):
        """Settle a bet based on actual game result."""
        bets = self.store.get_bets()
        bet = bets[bets["id"] == bet_id].iloc[0]

        # Determine actual bucket
        actual_bucket = margin_to_bucket(actual_abs_margin)
        won = bet["bucket"] == actual_bucket

        if won:
            pnl = bet["stake"] * (bet["decimal_odds"] - 1)
            self.store.update_bet_result(bet_id, "won", pnl)
        else:
            self.store.update_bet_result(bet_id, "lost", -bet["stake"])

    def summary(self) -> dict:
        """Get P&L summary."""
        bets = self.store.get_bets()
        settled = bets[bets["result"] != "pending"]

        if settled.empty:
            return {"total_bets": 0, "pnl": 0, "roi": 0}

        return {
            "total_bets": len(settled),
            "wins": len(settled[settled["result"] == "won"]),
            "losses": len(settled[settled["result"] == "lost"]),
            "pending": len(bets[bets["result"] == "pending"]),
            "total_staked": settled["stake"].sum(),
            "pnl": settled["pnl"].sum(),
            "roi": (
                settled["pnl"].sum() / settled["stake"].sum()
                if settled["stake"].sum() > 0 else 0
            ),
            "avg_edge": settled["edge"].mean(),
            "avg_odds": settled["decimal_odds"].mean(),
        }


def margin_to_bucket(abs_margin: int) -> str:
    """Convert absolute margin to bucket name."""
    if abs_margin <= 5:
        return "1-5"
    elif abs_margin <= 10:
        return "6-10"
    elif abs_margin <= 15:
        return "11-15"
    elif abs_margin <= 20:
        return "16-20"
    elif abs_margin <= 25:
        return "21-25"
    elif abs_margin <= 30:
        return "26-30"
    else:
        return "31+"
