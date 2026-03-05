"""Tests for SQLite data store."""


import pandas as pd
import pytest

from src.data.data_store import DataStore


@pytest.fixture
def store(tmp_path):
    return DataStore(db_path=tmp_path / "test.db")


@pytest.fixture
def sample_games():
    return pd.DataFrame([
        {"source": "nba_api", "league": "NBA", "game_id": "001",
         "date": "2024-01-01", "home_team": "LAL", "away_team": "BOS",
         "home_score": 110, "away_score": 105},
        {"source": "nba_api", "league": "NBA", "game_id": "002",
         "date": "2024-01-02", "home_team": "GSW", "away_team": "MIA",
         "home_score": 98, "away_score": 112},
        {"source": "nba_api", "league": "NBA", "game_id": "003",
         "date": "2024-01-03", "home_team": "LAL", "away_team": "GSW",
         "home_score": 120, "away_score": 100},
    ])


class TestDataStore:
    def test_upsert_and_retrieve_games(self, store, sample_games):
        store.upsert_games(sample_games)
        result = store.get_games()
        assert len(result) == 3
        assert "margin" in result.columns
        assert "abs_margin" in result.columns

    def test_computed_margin_columns(self, store, sample_games):
        store.upsert_games(sample_games)
        result = store.get_games()
        # LAL 110 - BOS 105 = +5
        lal_bos = result[result["game_id"] == "001"].iloc[0]
        assert lal_bos["margin"] == 5
        assert lal_bos["abs_margin"] == 5
        # GSW 98 - MIA 112 = -14
        gsw_mia = result[result["game_id"] == "002"].iloc[0]
        assert gsw_mia["margin"] == -14
        assert gsw_mia["abs_margin"] == 14

    def test_upsert_ignores_duplicates(self, store, sample_games):
        store.upsert_games(sample_games)
        store.upsert_games(sample_games)  # insert again
        result = store.get_games()
        assert len(result) == 3  # still 3, not 6

    def test_filter_by_league(self, store, sample_games):
        store.upsert_games(sample_games)
        nba = store.get_games(league="NBA")
        assert len(nba) == 3
        empty = store.get_games(league="VTB")
        assert len(empty) == 0

    def test_filter_by_min_date(self, store, sample_games):
        store.upsert_games(sample_games)
        recent = store.get_games(min_date="2024-01-02")
        assert len(recent) == 2

    def test_upsert_and_retrieve_odds(self, store):
        odds_df = pd.DataFrame([
            {"scraped_at": "2024-01-01T10:00:00", "source": "sportsplus",
             "league": "NBA", "game_date": "2024-01-01",
             "home_team": "LAL", "away_team": "BOS",
             "bucket": "1-5", "decimal_odds": 3.50},
            {"scraped_at": "2024-01-01T10:00:00", "source": "sportsplus",
             "league": "NBA", "game_date": "2024-01-01",
             "home_team": "LAL", "away_team": "BOS",
             "bucket": "6-10", "decimal_odds": 4.20},
        ])
        store.upsert_odds(odds_df)
        result = store.get_odds(game_date="2024-01-01")
        assert len(result) == 2

    def test_odds_upsert_updates_on_conflict(self, store):
        odds1 = pd.DataFrame([{
            "scraped_at": "2024-01-01T10:00:00", "source": "sportsplus",
            "league": "NBA", "game_date": "2024-01-01",
            "home_team": "LAL", "away_team": "BOS",
            "bucket": "1-5", "decimal_odds": 3.50,
        }])
        odds2 = pd.DataFrame([{
            "scraped_at": "2024-01-01T12:00:00", "source": "sportsplus",
            "league": "NBA", "game_date": "2024-01-01",
            "home_team": "LAL", "away_team": "BOS",
            "bucket": "1-5", "decimal_odds": 3.80,  # updated odds
        }])
        store.upsert_odds(odds1)
        store.upsert_odds(odds2)
        result = store.get_odds()
        assert len(result) == 1
        assert result.iloc[0]["decimal_odds"] == 3.80

    def test_record_and_retrieve_bets(self, store):
        store.record_bet({
            "placed_at": "2024-01-01T10:00:00",
            "game_date": "2024-01-01",
            "league": "NBA",
            "home_team": "LAL",
            "away_team": "BOS",
            "bucket": "1-5",
            "decimal_odds": 3.50,
            "model_prob": 0.35,
            "edge": 0.064,
            "stake": 100.0,
        })
        bets = store.get_bets()
        assert len(bets) == 1
        assert bets.iloc[0]["result"] == "pending"

    def test_update_bet_result(self, store):
        store.record_bet({
            "placed_at": "2024-01-01T10:00:00",
            "game_date": "2024-01-01",
            "league": "NBA",
            "home_team": "LAL",
            "away_team": "BOS",
            "bucket": "1-5",
            "decimal_odds": 3.50,
            "model_prob": 0.35,
            "edge": 0.064,
            "stake": 100.0,
        })
        bets = store.get_bets()
        bet_id = int(bets.iloc[0]["id"])
        store.update_bet_result(bet_id, "won", 250.0)

        settled = store.get_bets(status="won")
        assert len(settled) == 1
        assert settled.iloc[0]["pnl"] == 250.0
