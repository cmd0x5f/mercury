"""Per-league margin predictors + folded normal bucket probabilities.

Supports pluggable ML backends (XGBoost, Ridge, RF, LightGBM) via the
backends registry. Each backend declares its preprocessing needs."""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.features.builder import FEATURE_COLS, build_features, get_feature_matrix
from src.model.backends import DEFAULT_BACKEND, get_backend_class
from src.model.backends.base import BaseBackend
from src.model.calibration import PlattCalibrator
from src.model.distribution import BUCKET_NAMES, bucket_probabilities
from src.model.preprocessor import Preprocessor

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parents[2] / "data" / "models"

# Leagues with fewer games than this get pooled into a shared "small leagues" model
MIN_STANDALONE_GAMES = 500


class _LeagueModel:
    """A single league's trained model."""

    def __init__(self):
        self.backend: BaseBackend | None = None
        self.preprocessor: Preprocessor | None = None
        self.sigma: float = 12.0
        self.calibrator: PlattCalibrator | None = None
        self.n_games: int = 0


class MarginModel:
    """Manages per-league models. Each league with enough data gets its
    own model, σ, and Platt calibrator. Small leagues share a pooled model.

    Supports per-league backend selection via league_configs. If league_configs
    is provided, each league can use a different ML backend + hyperparams.
    """

    def __init__(
        self,
        backend_name: str = DEFAULT_BACKEND,
        backend_params: dict | None = None,
        league_configs: dict | None = None,
    ):
        self.backend_name = backend_name
        self.backend_params = backend_params or {}
        # Per-league overrides: {league_name: {"backend_name": str, "backend_params": dict}}
        self.league_configs = league_configs or {}
        self.league_models: dict[str, _LeagueModel] = {}  # league_name -> model
        self.fallback: _LeagueModel | None = None  # pooled model for small leagues
        self.league_categories: dict[str, int] = {}  # league_name -> league_id

    @property
    def sigma(self) -> float:
        """Global σ — average across all league models."""
        sigmas = [lm.sigma for lm in self.league_models.values()]
        if self.fallback:
            sigmas.append(self.fallback.sigma)
        return float(np.mean(sigmas)) if sigmas else 12.0

    @property
    def league_sigmas(self) -> dict[int, float]:
        """Per-league σ for backward compat with CLI display."""
        result = {}
        for name, lm in self.league_models.items():
            lid = self.league_categories.get(name)
            if lid is not None:
                result[lid] = lm.sigma
        return result

    def _get_league_model(self, league_name: str) -> _LeagueModel:
        """Get the model for a league, falling back to pooled model."""
        if league_name in self.league_models:
            return self.league_models[league_name]
        if self.fallback:
            return self.fallback
        raise ValueError(f"No model for league '{league_name}' and no fallback model")

    def _create_backend(self, league_name: str | None = None) -> BaseBackend:
        """Create a fresh backend instance, respecting per-league overrides."""
        if league_name and league_name in self.league_configs:
            cfg = self.league_configs[league_name]
            name = cfg.get("backend_name", self.backend_name)
            params = cfg.get("backend_params", {})
            return get_backend_class(name)(**params)
        return get_backend_class(self.backend_name)(**self.backend_params)

    def train(self, games: pd.DataFrame, elo_k: int = 20, player_impact: pd.DataFrame = None):
        """Train separate models per league.

        Leagues with >= MIN_STANDALONE_GAMES get their own model.
        Smaller leagues are pooled into a shared fallback model.
        """
        if "league" not in games.columns:
            games = games.copy()
            games["league"] = "NBA"

        league_counts = games["league"].value_counts()
        standalone_leagues = league_counts[league_counts >= MIN_STANDALONE_GAMES].index.tolist()
        small_leagues = league_counts[league_counts < MIN_STANDALONE_GAMES].index.tolist()

        logger.info(
            f"{len(standalone_leagues)} standalone leagues, "
            f"{len(small_leagues)} pooled into fallback"
        )

        # Build league_categories from all games
        all_leagues = sorted(games["league"].unique())
        self.league_categories = {name: i for i, name in enumerate(all_leagues)}

        # Train standalone league models
        for league in standalone_leagues:
            lg_games = games[games["league"] == league].copy()

            # Only pass player impact for NBA (we only have NBA player logs)
            lg_impact = None
            if player_impact is not None and league == "NBA" and "game_id" in lg_games.columns:
                lg_impact = player_impact[
                    player_impact["game_id"].isin(lg_games["game_id"])
                ]

            lm = self._train_single(lg_games, elo_k, lg_impact, league_name=league)
            lm.n_games = len(lg_games)
            self.league_models[league] = lm
            backend_used = lm.backend.name()
            logger.info(f"  {league}: {len(lg_games)} games, σ={lm.sigma:.2f} [{backend_used}]")

        # Train fallback model on pooled small leagues
        if small_leagues:
            small_games = games[games["league"].isin(small_leagues)].copy()
            self.fallback = self._train_single(small_games, elo_k, league_name="__fallback__")
            self.fallback.n_games = len(small_games)
            backend_used = self.fallback.backend.name()
            logger.info(
                f"  Fallback ({len(small_leagues)} leagues): "
                f"{len(small_games)} games, σ={self.fallback.sigma:.2f} [{backend_used}]"
            )

    def _train_single(
        self, games: pd.DataFrame, elo_k: int, player_impact: pd.DataFrame = None,
        league_name: str | None = None,
    ) -> _LeagueModel:
        """Train a single model on a set of games."""
        featured = build_features(games, elo_k=elo_k, player_impact=player_impact)

        # Drop warmup games (first 200 or 1/3, whichever is smaller)
        warmup = min(200, len(featured) // 3)
        featured = featured.iloc[warmup:].reset_index(drop=True)

        X, y = get_feature_matrix(featured)

        lm = _LeagueModel()
        backend = self._create_backend(league_name=league_name)
        preprocessor = Preprocessor(backend.preprocessing_config())
        X_proc = preprocessor.fit_transform(X)

        backend.fit(X_proc, y.values)
        lm.backend = backend
        lm.preprocessor = preprocessor

        preds = backend.predict(X_proc)
        residuals = y.values - preds
        lm.sigma = float(np.std(residuals))

        # Fit Platt calibrator
        lm.calibrator = PlattCalibrator()
        raw_probs = []
        actual_buckets = []
        for pred_mu, actual_margin in zip(preds, y.values):
            raw_probs.append(bucket_probabilities(float(pred_mu), lm.sigma))
            actual_abs = abs(actual_margin)
            actual_bucket = "31+"
            for name, low, high in [("1-5",1,5),("6-10",6,10),("11-15",11,15),
                                     ("16-20",16,20),("21-25",21,25),("26-30",26,30)]:
                if low <= actual_abs <= high:
                    actual_bucket = name
                    break
            actual_buckets.append(actual_bucket)
        lm.calibrator.fit(raw_probs, actual_buckets)

        return lm

    def _predict_with_lm(self, lm: _LeagueModel, X: pd.DataFrame) -> np.ndarray:
        """Run prediction through preprocessor + backend."""
        X_proc = lm.preprocessor.transform(X)
        return lm.backend.predict(X_proc)

    def _maybe_calibrate(self, lm: _LeagueModel, probs: dict[str, float]) -> dict[str, float]:
        """Apply Platt calibration if a calibrator is fitted."""
        if lm.calibrator and lm.calibrator.calibrators:
            return lm.calibrator.calibrate(probs)
        return probs

    def predict_margin(self, features: pd.DataFrame) -> np.ndarray:
        """Predict signed margin for games. Uses first available league model."""
        lm = next(iter(self.league_models.values()), self.fallback)
        return self._predict_with_lm(lm, features[FEATURE_COLS])

    def predict_buckets(self, features: pd.DataFrame, league_name: str = None) -> list[dict[str, float]]:
        """Predict bucket probabilities for games."""
        lm = self._get_league_model(league_name) if league_name else next(
            iter(self.league_models.values()), self.fallback
        )
        mus = self._predict_with_lm(lm, features[FEATURE_COLS])
        return [
            self._maybe_calibrate(lm, bucket_probabilities(float(mu), lm.sigma))
            for mu in mus
        ]

    def predict_single(self, features: dict, league_id: int | None = None) -> dict[str, float]:
        """Predict bucket probabilities for a single game."""
        # Resolve league name from league_id
        league_name = None
        if league_id is not None:
            inv = {v: k for k, v in self.league_categories.items()}
            league_name = inv.get(league_id)

        lm = self._get_league_model(league_name) if league_name else (
            next(iter(self.league_models.values()), self.fallback)
        )

        df = pd.DataFrame([features])
        mu = float(self._predict_with_lm(lm, df[FEATURE_COLS])[0])
        return self._maybe_calibrate(lm, bucket_probabilities(mu, lm.sigma))

    def evaluate(self, games: pd.DataFrame, elo_k: int = 20, player_impact: pd.DataFrame = None) -> dict:
        """Walk-forward evaluation per league, then aggregate results."""
        if "league" not in games.columns:
            games = games.copy()
            games["league"] = "NBA"

        league_counts = games["league"].value_counts()
        all_residuals = []
        bucket_hits = {b: {"correct": 0, "total": 0} for b in BUCKET_NAMES}
        league_results = {}

        for league in games["league"].unique():
            lg_games = games[games["league"] == league].copy()

            lg_impact = None
            if player_impact is not None and league == "NBA" and "game_id" in lg_games.columns:
                lg_impact = player_impact[
                    player_impact["game_id"].isin(lg_games["game_id"])
                ]

            featured = build_features(lg_games, elo_k=elo_k, player_impact=lg_impact)
            warmup = min(200, len(featured) // 3)
            featured = featured.iloc[warmup:].reset_index(drop=True)

            if len(featured) < 50:
                continue

            tscv = TimeSeriesSplit(n_splits=5)
            lg_residuals = []

            for train_idx, test_idx in tscv.split(featured):
                X_train, y_train = get_feature_matrix(featured.iloc[train_idx])
                X_test, y_test = get_feature_matrix(featured.iloc[test_idx])

                backend = self._create_backend(league_name=league)
                preprocessor = Preprocessor(backend.preprocessing_config())
                X_train_proc = preprocessor.fit_transform(X_train)
                X_test_proc = preprocessor.transform(X_test)

                backend.fit(X_train_proc, y_train.values)
                preds = backend.predict(X_test_proc)
                residuals = y_test.values - preds
                lg_residuals.extend(residuals)

                train_preds = backend.predict(X_train_proc)
                train_sigma = float(np.std(y_train.values - train_preds))

                for pred_mu, actual_margin in zip(preds, y_test.values):
                    probs = bucket_probabilities(float(pred_mu), train_sigma)
                    predicted_bucket = max(probs, key=probs.get)
                    actual_abs = abs(actual_margin)
                    actual_bucket = "31+"
                    for name, low, high in [("1-5",1,5),("6-10",6,10),("11-15",11,15),
                                             ("16-20",16,20),("21-25",21,25),("26-30",26,30)]:
                        if low <= actual_abs <= high:
                            actual_bucket = name
                            break
                    bucket_hits[actual_bucket]["total"] += 1
                    if predicted_bucket == actual_bucket:
                        bucket_hits[actual_bucket]["correct"] += 1

            all_residuals.extend(lg_residuals)
            lg_sigma = float(np.std(lg_residuals))
            league_results[league] = {
                "sigma": lg_sigma,
                "mae": float(np.mean(np.abs(lg_residuals))),
                "n_games": len(lg_games),
            }

        mae = float(np.mean(np.abs(all_residuals)))
        rmse = float(np.sqrt(np.mean(np.array(all_residuals) ** 2)))
        sigma_est = float(np.std(all_residuals))

        return {
            "mae": mae,
            "rmse": rmse,
            "sigma": sigma_est,
            "bucket_accuracy": {
                b: d["correct"] / d["total"] if d["total"] > 0 else 0
                for b, d in bucket_hits.items()
            },
            "per_league": league_results,
        }

    def save(self, path: Path = None):
        path = path or MODEL_DIR / "margin_model.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "version": 3,
                "backend_name": self.backend_name,  # default backend
                "league_models": {
                    name: {
                        "backend_name": lm.backend.name(),
                        "backend_params": lm.backend.get_params(),
                        "preprocessor_state": lm.preprocessor.get_state(),
                        "sigma": lm.sigma,
                        "calibrator": lm.calibrator,
                        "n_games": lm.n_games,
                    }
                    for name, lm in self.league_models.items()
                },
                "fallback": {
                    "backend_name": self.fallback.backend.name(),
                    "backend_params": self.fallback.backend.get_params(),
                    "preprocessor_state": self.fallback.preprocessor.get_state(),
                    "sigma": self.fallback.sigma,
                    "calibrator": self.fallback.calibrator,
                    "n_games": self.fallback.n_games,
                } if self.fallback else None,
                "league_categories": self.league_categories,
            }, f)
        logger.info(f"Model saved to {path} ({len(self.league_models)} league models)")

    def load(self, path: Path = None):
        path = path or MODEL_DIR / "margin_model.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.league_categories = data.get("league_categories", {})
        self.backend_name = data.get("backend_name", DEFAULT_BACKEND)

        self.league_models = {}
        for name, lm_data in data.get("league_models", {}).items():
            lm = _LeagueModel()
            lm_backend_name = lm_data.get("backend_name", self.backend_name)
            lm_cls = get_backend_class(lm_backend_name)
            lm.backend = lm_cls.from_params(lm_data["backend_params"])
            lm.preprocessor = Preprocessor.from_state(lm_data["preprocessor_state"])
            lm.sigma = lm_data["sigma"]
            lm.calibrator = lm_data.get("calibrator")
            lm.n_games = lm_data.get("n_games", 0)
            self.league_models[name] = lm

        fb_data = data.get("fallback")
        if fb_data:
            self.fallback = _LeagueModel()
            fb_backend_name = fb_data.get("backend_name", self.backend_name)
            fb_cls = get_backend_class(fb_backend_name)
            self.fallback.backend = fb_cls.from_params(fb_data["backend_params"])
            self.fallback.preprocessor = Preprocessor.from_state(fb_data["preprocessor_state"])
            self.fallback.sigma = fb_data["sigma"]
            self.fallback.calibrator = fb_data.get("calibrator")
            self.fallback.n_games = fb_data.get("n_games", 0)

        league_info = ", ".join(
            f"{name}[{lm.backend.name()}](σ={lm.sigma:.2f})"
            for name, lm in self.league_models.items()
        )
        logger.info(f"Loaded model ({len(self.league_models)} leagues): {league_info}")

    def get_league_id(self, league_name: str) -> int:
        """Look up a league_id from a league name. Returns 0 if unknown."""
        return self.league_categories.get(league_name, 0)
