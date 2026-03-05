"""Tests for betting logic — value calculator, Kelly sizing, and tracker."""

import pytest

from src.betting.kelly import calculate_stakes, kelly_stake
from src.betting.tracker import BetTracker, margin_to_bucket
from src.betting.value_calculator import BetOpportunity, find_value_bets
from src.data.data_store import DataStore

# --- Value Calculator ---

class TestBetOpportunity:
    def test_implied_prob(self):
        bet = BetOpportunity("2024-01-01", "NBA", "LAL", "BOS", "1-5",
                             model_prob=0.30, decimal_odds=4.0)
        assert bet.implied_prob == pytest.approx(0.25)

    def test_edge_positive(self):
        bet = BetOpportunity("2024-01-01", "NBA", "LAL", "BOS", "1-5",
                             model_prob=0.30, decimal_odds=4.0)
        assert bet.edge == pytest.approx(0.05)  # 30% - 25%

    def test_edge_negative(self):
        bet = BetOpportunity("2024-01-01", "NBA", "LAL", "BOS", "1-5",
                             model_prob=0.20, decimal_odds=4.0)
        assert bet.edge == pytest.approx(-0.05)

    def test_ev_per_unit_positive(self):
        # model_prob=0.30, odds=4.0: EV = 0.30*3 - 0.70 = 0.90 - 0.70 = 0.20
        bet = BetOpportunity("2024-01-01", "NBA", "LAL", "BOS", "1-5",
                             model_prob=0.30, decimal_odds=4.0)
        assert bet.ev_per_unit == pytest.approx(0.20)

    def test_is_positive_ev(self):
        pos = BetOpportunity("2024-01-01", "NBA", "LAL", "BOS", "1-5",
                             model_prob=0.30, decimal_odds=4.0)
        neg = BetOpportunity("2024-01-01", "NBA", "LAL", "BOS", "1-5",
                             model_prob=0.20, decimal_odds=4.0)
        assert pos.is_positive_ev is True
        assert neg.is_positive_ev is False


class TestFindValueBets:
    def test_finds_bets_above_threshold(self):
        model_probs = {"1-5": 0.35, "6-10": 0.25, "11-15": 0.15,
                       "16-20": 0.10, "21-25": 0.07, "26-30": 0.05, "31+": 0.03}
        odds = {"1-5": 3.50, "6-10": 4.00, "11-15": 6.00,
                "16-20": 10.0, "21-25": 15.0, "26-30": 20.0, "31+": 25.0}
        game_info = {"game_date": "2024-01-01", "home_team": "LAL", "away_team": "BOS"}

        bets = find_value_bets(model_probs, odds, game_info, min_edge=0.05)
        # All returned bets should have edge >= 5%
        for bet in bets:
            assert bet.edge >= 0.05

    def test_returns_empty_when_no_edge(self):
        model_probs = {"1-5": 0.25, "6-10": 0.20}
        odds = {"1-5": 4.0, "6-10": 5.0}  # implied = 25%, 20% — no edge
        game_info = {"game_date": "2024-01-01", "home_team": "LAL", "away_team": "BOS"}

        bets = find_value_bets(model_probs, odds, game_info, min_edge=0.05)
        assert len(bets) == 0

    def test_sorted_by_edge_descending(self):
        model_probs = {"1-5": 0.40, "6-10": 0.35}
        odds = {"1-5": 3.50, "6-10": 3.50}  # implied = 28.6%
        game_info = {"game_date": "2024-01-01", "home_team": "LAL", "away_team": "BOS"}

        bets = find_value_bets(model_probs, odds, game_info, min_edge=0.05)
        assert len(bets) == 2
        assert bets[0].edge >= bets[1].edge

    def test_skips_buckets_without_odds(self):
        model_probs = {"1-5": 0.40, "6-10": 0.30}
        odds = {"1-5": 3.0}  # only one bucket has odds
        game_info = {"game_date": "2024-01-01", "home_team": "LAL", "away_team": "BOS"}

        bets = find_value_bets(model_probs, odds, game_info, min_edge=0.05)
        for bet in bets:
            assert bet.bucket == "1-5"


# --- Kelly Sizing ---

class TestKellyStake:
    def test_positive_ev_returns_stake(self):
        # prob=0.30, odds=4.0: kelly = (3*0.30 - 0.70)/3 = 0.20/3 ≈ 0.0667
        # quarter kelly ≈ 0.0167
        stake = kelly_stake(0.30, 4.0, fraction=0.25)
        assert stake == pytest.approx(0.0167, abs=0.001)

    def test_negative_ev_returns_zero(self):
        stake = kelly_stake(0.20, 4.0, fraction=0.25)
        assert stake == 0.0

    def test_respects_max_stake(self):
        # Very high edge should be capped
        stake = kelly_stake(0.80, 2.0, fraction=1.0, max_stake_pct=0.03)
        assert stake <= 0.03

    def test_respects_min_stake(self):
        # Small edge still gets minimum stake
        stake = kelly_stake(0.26, 4.0, fraction=0.25, min_stake_pct=0.005)
        assert stake >= 0.005

    def test_full_kelly_vs_quarter(self):
        full = kelly_stake(0.30, 4.0, fraction=1.0, max_stake_pct=1.0)
        quarter = kelly_stake(0.30, 4.0, fraction=0.25, max_stake_pct=1.0)
        assert quarter < full


class TestCalculateStakes:
    def test_exposure_limit_enforced(self):
        bets = [
            BetOpportunity("2024-01-01", "NBA", "LAL", "BOS", f"bucket{i}",
                           model_prob=0.40, decimal_odds=3.50)
            for i in range(10)
        ]
        # With max_exposure=15%, bankroll=10000, max 1500 total
        results = calculate_stakes(bets, bankroll=10000, max_exposure_pct=0.15)
        total = sum(s for _, s in results)
        assert total <= 1500.01  # small float tolerance


# --- Margin to Bucket ---

class TestMarginToBucket:
    @pytest.mark.parametrize("margin,expected", [
        (1, "1-5"),
        (3, "1-5"),
        (5, "1-5"),
        (6, "6-10"),
        (10, "6-10"),
        (11, "11-15"),
        (15, "11-15"),
        (16, "16-20"),
        (20, "16-20"),
        (21, "21-25"),
        (25, "21-25"),
        (26, "26-30"),
        (30, "26-30"),
        (31, "31+"),
        (50, "31+"),
    ])
    def test_margin_to_bucket(self, margin, expected):
        assert margin_to_bucket(margin) == expected


# --- Bet Tracker ---

class TestBetTracker:
    @pytest.fixture
    def tracker(self, tmp_path):
        store = DataStore(db_path=tmp_path / "test.db")
        return BetTracker(store=store)

    def test_record_and_summary(self, tracker):
        bet = BetOpportunity("2024-01-01", "NBA", "LAL", "BOS", "1-5",
                             model_prob=0.35, decimal_odds=3.50)
        tracker.record(bet, stake=100.0)
        s = tracker.summary()
        assert s["total_bets"] == 0  # none settled yet

    def test_settle_winning_bet(self, tracker):
        bet = BetOpportunity("2024-01-01", "NBA", "LAL", "BOS", "1-5",
                             model_prob=0.35, decimal_odds=3.50)
        tracker.record(bet, stake=100.0)

        bets = tracker.store.get_bets()
        bet_id = int(bets.iloc[0]["id"])

        # Actual margin of 3 falls in 1-5 bucket — win!
        tracker.settle(bet_id, actual_abs_margin=3)

        s = tracker.summary()
        assert s["wins"] == 1
        assert s["pnl"] == pytest.approx(250.0)  # 100 * (3.50 - 1)

    def test_settle_losing_bet(self, tracker):
        bet = BetOpportunity("2024-01-01", "NBA", "LAL", "BOS", "1-5",
                             model_prob=0.35, decimal_odds=3.50)
        tracker.record(bet, stake=100.0)

        bets = tracker.store.get_bets()
        bet_id = int(bets.iloc[0]["id"])

        # Actual margin of 12 falls in 11-15 bucket — loss!
        tracker.settle(bet_id, actual_abs_margin=12)

        s = tracker.summary()
        assert s["losses"] == 1
        assert s["pnl"] == pytest.approx(-100.0)
