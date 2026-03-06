"""SQLite storage for games, odds, and bet tracking."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

DEFAULT_DB = Path(__file__).parents[2] / "data" / "sportsbetting.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,          -- 'nba_api', 'flashscore', 'sofascore'
    league TEXT NOT NULL,          -- 'NBA', 'VTB', etc.
    game_id TEXT NOT NULL,         -- source-specific ID
    date TEXT NOT NULL,            -- YYYY-MM-DD
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_score INTEGER NOT NULL,
    away_score INTEGER NOT NULL,
    margin INTEGER GENERATED ALWAYS AS (home_score - away_score) STORED,
    abs_margin INTEGER GENERATED ALWAYS AS (ABS(home_score - away_score)) STORED,
    UNIQUE(source, game_id)
);

CREATE INDEX IF NOT EXISTS idx_games_league_date ON games(league, date);
CREATE INDEX IF NOT EXISTS idx_games_teams ON games(home_team, away_team);

CREATE TABLE IF NOT EXISTS odds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scraped_at TEXT NOT NULL,      -- ISO timestamp
    source TEXT NOT NULL,          -- 'sportsplus'
    league TEXT NOT NULL,
    game_date TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    bucket TEXT NOT NULL,          -- '1-5', '6-10', ..., '31+'
    decimal_odds REAL NOT NULL,
    UNIQUE(source, game_date, home_team, away_team, bucket)
);

CREATE TABLE IF NOT EXISTS bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    placed_at TEXT NOT NULL,
    game_date TEXT NOT NULL,
    league TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    bucket TEXT NOT NULL,
    decimal_odds REAL NOT NULL,
    model_prob REAL NOT NULL,
    edge REAL NOT NULL,
    stake REAL NOT NULL,
    result TEXT DEFAULT 'pending', -- 'pending', 'won', 'lost'
    pnl REAL DEFAULT 0
);
"""


class DataStore:
    def __init__(self, db_path: Path = DEFAULT_DB):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def upsert_games(self, df: pd.DataFrame) -> int:
        """Insert games, skipping duplicates. Returns count of newly inserted rows.

        Expects columns:
        source, league, game_id, date, home_team, away_team, home_score, away_score
        """
        with self._conn() as conn:
            before = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
            df.to_sql("_games_staging", conn, if_exists="replace", index=False)
            conn.execute("""
                INSERT OR IGNORE INTO games
                    (source, league, game_id, date, home_team, away_team, home_score, away_score)
                SELECT source, league, game_id, date, home_team, away_team, home_score, away_score
                FROM _games_staging
            """)
            conn.execute("DROP TABLE IF EXISTS _games_staging")
            after = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
            return after - before

    def get_latest_dates_by_league(self) -> dict[str, str]:
        """Return {league_name: latest_date} for all leagues in the database."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT league, MAX(date) FROM games GROUP BY league"
            ).fetchall()
            return {row[0]: row[1] for row in rows}

    def get_games(self, league: str = None, min_date: str = None) -> pd.DataFrame:
        query = "SELECT * FROM games WHERE 1=1"
        params = []
        if league:
            query += " AND league = ?"
            params.append(league)
        if min_date:
            query += " AND date >= ?"
            params.append(min_date)
        query += " ORDER BY date"
        with self._conn() as conn:
            return pd.read_sql(query, conn, params=params)

    def upsert_odds(self, df: pd.DataFrame):
        """Insert odds, updating on conflict."""
        with self._conn() as conn:
            for _, row in df.iterrows():
                conn.execute("""
                    INSERT INTO odds
                        (scraped_at, source, league, game_date,
                         home_team, away_team, bucket, decimal_odds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source, game_date, home_team, away_team, bucket)
                    DO UPDATE SET
                        decimal_odds = excluded.decimal_odds,
                        scraped_at = excluded.scraped_at
                """, (row.scraped_at, row.source, row.league, row.game_date,
                      row.home_team, row.away_team, row.bucket, row.decimal_odds))

    def get_odds(self, game_date: str = None) -> pd.DataFrame:
        query = "SELECT * FROM odds WHERE 1=1"
        params = []
        if game_date:
            query += " AND game_date = ?"
            params.append(game_date)
        with self._conn() as conn:
            return pd.read_sql(query, conn, params=params)

    def record_bet(self, bet: dict):
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO bets (placed_at, game_date, league, home_team, away_team,
                                  bucket, decimal_odds, model_prob, edge, stake)
                VALUES (:placed_at, :game_date, :league, :home_team, :away_team,
                        :bucket, :decimal_odds, :model_prob, :edge, :stake)
            """, bet)

    def update_bet_result(self, bet_id: int, result: str, pnl: float):
        with self._conn() as conn:
            conn.execute(
                "UPDATE bets SET result = ?, pnl = ? WHERE id = ?",
                (result, pnl, bet_id),
            )

    def get_bets(self, status: str = None) -> pd.DataFrame:
        query = "SELECT * FROM bets WHERE 1=1"
        params = []
        if status:
            query += " AND result = ?"
            params.append(status)
        with self._conn() as conn:
            return pd.read_sql(query, conn, params=params)
