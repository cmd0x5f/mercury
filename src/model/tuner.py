"""Hyperparameter tuning via Optuna — uses walk-forward CV as the objective."""

import logging

import optuna
import pandas as pd

from src.model.backends import get_backend_class
from src.model.margin_model import MarginModel

logger = logging.getLogger(__name__)


def tune(
    backend_name: str,
    games: pd.DataFrame,
    n_trials: int = 30,
    metric: str = "mae",
    elo_k: int = 20,
    player_impact: pd.DataFrame = None,
) -> dict:
    """Run Optuna hyperparameter search for a given backend.

    Args:
        backend_name: Which backend to tune (e.g. "xgboost", "ridge").
        games: Historical game data.
        n_trials: Number of Optuna trials to run.
        metric: Metric to minimize — "mae" or "rmse".
        elo_k: Elo K-factor passed to evaluate().
        player_impact: Optional player impact DataFrame.

    Returns:
        Dict with "best_params", "best_value", and "study" (the Optuna study).
    """
    backend_cls = get_backend_class(backend_name)

    def objective(trial: optuna.Trial) -> float:
        hyperparams = backend_cls.search_space(trial)
        model = MarginModel(backend_name=backend_name, backend_params=hyperparams)
        results = model.evaluate(games, elo_k=elo_k, player_impact=player_impact)
        value = results[metric]
        logger.info(
            f"Trial {trial.number}: {metric}={value:.4f} | {hyperparams}"
        )
        return value

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    logger.info(
        f"Best {metric}: {study.best_value:.4f} "
        f"(trial {study.best_trial.number})"
    )

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "study": study,
    }
