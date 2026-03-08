"""Auto-tune pipeline: per-league backend selection + hyperparameter optimization.

For each league:
  1. Evaluate all backends via walk-forward CV → pick lowest MAE
  2. Tune the winning backend's hyperparams with Optuna
  3. Return a LeagueConfig with (backend_name, best_params)
"""

import logging
from dataclasses import dataclass, field

import pandas as pd

from src.model.backends import available_backends
from src.model.margin_model import MarginModel, MIN_STANDALONE_GAMES
from src.model.tuner import tune as run_tune

logger = logging.getLogger(__name__)


@dataclass
class LeagueConfig:
    """Per-league backend selection + tuned hyperparameters."""
    backend_name: str
    backend_params: dict = field(default_factory=dict)
    cv_score: float = 0.0  # best CV MAE before tuning
    tuned_score: float = 0.0  # CV MAE after tuning


def _evaluate_backend_for_league(
    backend_name: str,
    league_games: pd.DataFrame,
    metric: str = "mae",
    elo_k: int = 20,
    player_impact: pd.DataFrame = None,
) -> float:
    """Run walk-forward CV for a single backend on a single league's data."""
    model = MarginModel(backend_name=backend_name)
    results = model.evaluate(league_games, elo_k=elo_k, player_impact=player_impact)
    return results[metric]


def auto_tune(
    games: pd.DataFrame,
    tune_trials: int = 20,
    metric: str = "mae",
    elo_k: int = 20,
    player_impact: pd.DataFrame = None,
    backends_to_try: list[str] | None = None,
) -> dict[str, LeagueConfig]:
    """Run the full auto-tune pipeline: compare backends per league, then tune the winner.

    Args:
        games: All historical game data (multi-league).
        tune_trials: Optuna trials per league for the winning backend.
        metric: Metric to minimize ("mae" or "rmse").
        elo_k: Elo K-factor.
        player_impact: Optional player impact DataFrame.
        backends_to_try: List of backend names to compare (default: all available).

    Returns:
        Dict mapping league_name → LeagueConfig with selected backend + tuned params.
    """
    if "league" not in games.columns:
        games = games.copy()
        games["league"] = "NBA"

    backends = backends_to_try or [b for b in available_backends() if b != "lightgbm"]

    league_counts = games["league"].value_counts()
    # Only auto-tune leagues with enough data for meaningful CV
    eligible = league_counts[league_counts >= MIN_STANDALONE_GAMES].index.tolist()
    small_leagues = league_counts[league_counts < MIN_STANDALONE_GAMES].index.tolist()

    logger.info(f"Auto-tuning {len(eligible)} leagues, {len(small_leagues)} pooled as fallback")
    logger.info(f"Backends to compare: {backends}")

    configs: dict[str, LeagueConfig] = {}

    # --- Phase 1: Compare backends per league ---
    for league in eligible:
        lg_games = games[games["league"] == league].copy()

        lg_impact = None
        if player_impact is not None and league == "NBA" and "game_id" in lg_games.columns:
            lg_impact = player_impact[player_impact["game_id"].isin(lg_games["game_id"])]

        logger.info(f"\n{'='*60}")
        logger.info(f"League: {league} ({len(lg_games)} games)")
        logger.info(f"{'='*60}")

        # Evaluate each backend
        scores: dict[str, float] = {}
        for backend_name in backends:
            try:
                score = _evaluate_backend_for_league(
                    backend_name, lg_games, metric=metric,
                    elo_k=elo_k, player_impact=lg_impact,
                )
                scores[backend_name] = score
                logger.info(f"  {backend_name:>10}: {metric}={score:.4f}")
            except Exception as e:
                logger.warning(f"  {backend_name:>10}: FAILED ({e})")

        if not scores:
            logger.warning(f"  All backends failed for {league}, using xgboost default")
            configs[league] = LeagueConfig(backend_name="xgboost")
            continue

        # Pick the best
        best_backend = min(scores, key=scores.get)
        best_score = scores[best_backend]
        logger.info(f"  Winner: {best_backend} ({metric}={best_score:.4f})")

        configs[league] = LeagueConfig(
            backend_name=best_backend,
            cv_score=best_score,
        )

    # --- Phase 2: Tune the winning backend per league ---
    if tune_trials > 0:
        for league in eligible:
            cfg = configs[league]
            lg_games = games[games["league"] == league].copy()

            lg_impact = None
            if player_impact is not None and league == "NBA" and "game_id" in lg_games.columns:
                lg_impact = player_impact[player_impact["game_id"].isin(lg_games["game_id"])]

            logger.info(f"\nTuning {cfg.backend_name} for {league} ({tune_trials} trials)...")

            try:
                result = run_tune(
                    backend_name=cfg.backend_name,
                    games=lg_games,
                    n_trials=tune_trials,
                    metric=metric,
                    elo_k=elo_k,
                    player_impact=lg_impact,
                )
                cfg.backend_params = result["best_params"]
                cfg.tuned_score = result["best_value"]
                logger.info(
                    f"  {league}: {metric} {cfg.cv_score:.4f} → {cfg.tuned_score:.4f} "
                    f"(Δ={cfg.tuned_score - cfg.cv_score:+.4f})"
                )
            except Exception as e:
                logger.warning(f"  Tuning failed for {league}: {e}")
                cfg.tuned_score = cfg.cv_score

    # --- Phase 3: Handle small leagues (pooled fallback) ---
    if small_leagues:
        pooled_games = games[games["league"].isin(small_leagues)].copy()
        logger.info(f"\nFallback ({len(small_leagues)} small leagues, {len(pooled_games)} games)")

        # Quick comparison for the pooled model
        scores = {}
        for backend_name in backends:
            try:
                score = _evaluate_backend_for_league(
                    backend_name, pooled_games, metric=metric, elo_k=elo_k,
                )
                scores[backend_name] = score
                logger.info(f"  {backend_name:>10}: {metric}={score:.4f}")
            except Exception as e:
                logger.warning(f"  {backend_name:>10}: FAILED ({e})")

        if scores:
            best = min(scores, key=scores.get)
            configs["__fallback__"] = LeagueConfig(
                backend_name=best, cv_score=scores[best]
            )
            logger.info(f"  Fallback winner: {best} ({metric}={scores[best]:.4f})")
        else:
            configs["__fallback__"] = LeagueConfig(backend_name="xgboost")

    return configs
