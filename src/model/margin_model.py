"""XGBoost margin predictor + folded normal bucket probabilities."""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

from src.features.builder import FEATURE_COLS, build_features, get_feature_matrix
from src.model.distribution import BUCKET_NAMES, bucket_probabilities, bucket_probabilities_batch

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parents[2] / "data" / "models"


class MarginModel:
    """Predicts signed margin with XGBoost, then converts to bucket probabilities
    via a folded normal distribution."""

    # Minimum games needed per league before using league-specific σ
    MIN_LEAGUE_GAMES = 50

    def __init__(self):
        self.model: xgb.XGBRegressor | None = None
        self.sigma: float = 12.0  # global fallback σ
        self.league_sigmas: dict[int, float] = {}  # league_id -> σ
        self.league_categories: dict[str, int] = {}  # league_name -> league_id

    def _get_sigma(self, league_id: int | None = None) -> float:
        """Get σ for a league, falling back to global σ."""
        if league_id is not None and league_id in self.league_sigmas:
            return self.league_sigmas[league_id]
        return self.sigma

    def train(self, games: pd.DataFrame, elo_k: int = 20):
        """Train on historical games.

        Args:
            games: raw games DataFrame with scores
        """
        featured = build_features(games, elo_k=elo_k)

        # Store league category mapping
        self.league_categories = featured.attrs.get("league_categories", {})

        # Drop early games where features are unreliable (first ~30 games per team)
        # Use games after the first 200 as training data
        featured = featured.iloc[200:].reset_index(drop=True)

        X, y = get_feature_matrix(featured)

        self.model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
        )
        self.model.fit(X, y, verbose=False)

        # Estimate global sigma from training residuals
        preds = self.model.predict(X)
        residuals = y.values - preds
        self.sigma = float(np.std(residuals))

        # Estimate per-league sigma
        self.league_sigmas = {}
        if "league_id" in featured.columns:
            for lid in featured["league_id"].unique():
                mask = featured.iloc[: len(X)]["league_id"] == lid
                if mask.sum() >= self.MIN_LEAGUE_GAMES:
                    league_resid = residuals[mask.values]
                    self.league_sigmas[int(lid)] = float(np.std(league_resid))

        league_info = ", ".join(
            f"{name}={self.league_sigmas.get(lid, self.sigma):.2f}"
            for name, lid in sorted(self.league_categories.items(), key=lambda x: x[1])
        ) if self.league_categories else ""
        logger.info(
            f"Trained on {len(X)} games, global σ={self.sigma:.2f}"
            + (f" | per-league σ: {league_info}" if league_info else "")
        )

    def predict_margin(self, features: pd.DataFrame) -> np.ndarray:
        """Predict signed margin for games."""
        return self.model.predict(features[FEATURE_COLS])

    def predict_buckets(self, features: pd.DataFrame) -> list[dict[str, float]]:
        """Predict bucket probabilities for games."""
        mus = self.predict_margin(features)
        # Use per-league σ if available
        if "league_id" in features.columns and self.league_sigmas:
            results = []
            for mu, lid in zip(mus, features["league_id"].values):
                sigma = self._get_sigma(int(lid))
                results.append(bucket_probabilities(float(mu), sigma))
            return results
        return bucket_probabilities_batch(mus, self.sigma)

    def predict_single(self, features: dict, league_id: int | None = None) -> dict[str, float]:
        """Predict bucket probabilities for a single game."""
        df = pd.DataFrame([features])
        mu = float(self.model.predict(df[FEATURE_COLS])[0])
        sigma = self._get_sigma(league_id if league_id is not None else features.get("league_id"))
        return bucket_probabilities(mu, sigma)

    def evaluate(self, games: pd.DataFrame, elo_k: int = 20) -> dict:
        """Walk-forward evaluation: train on games up to each point, predict next."""
        featured = build_features(games, elo_k=elo_k)
        featured = featured.iloc[200:].reset_index(drop=True)

        tscv = TimeSeriesSplit(n_splits=5)
        all_residuals = []
        bucket_hits = {b: {"correct": 0, "total": 0} for b in BUCKET_NAMES}

        for train_idx, test_idx in tscv.split(featured):
            X_train, y_train = get_feature_matrix(featured.iloc[train_idx])
            X_test, y_test = get_feature_matrix(featured.iloc[test_idx])

            model = xgb.XGBRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=1.0, random_state=42,
            )
            model.fit(X_train, y_train, verbose=False)
            preds = model.predict(X_test)
            residuals = y_test.values - preds
            all_residuals.extend(residuals)

            train_residuals = y_train.values - model.predict(X_train)
            sigma = float(np.std(train_residuals))

            # Per-league σ for evaluation
            eval_league_sigmas = {}
            if "league_id" in featured.columns:
                train_data = featured.iloc[train_idx]
                for lid in train_data["league_id"].unique():
                    mask = train_data["league_id"] == lid
                    if mask.sum() >= self.MIN_LEAGUE_GAMES:
                        lid_resid = train_residuals[mask.values]
                        eval_league_sigmas[int(lid)] = float(np.std(lid_resid))

            # Check bucket accuracy
            if "league_id" in featured.columns:
                test_league_ids = featured.iloc[test_idx]["league_id"].values
            else:
                test_league_ids = [None] * len(test_idx)
            for i, (pred_mu, actual_margin) in enumerate(zip(preds, y_test.values)):
                lid = int(test_league_ids[i]) if test_league_ids[i] is not None else None
                game_sigma = eval_league_sigmas.get(lid, sigma) if lid is not None else sigma
                probs = bucket_probabilities(float(pred_mu), game_sigma)
                predicted_bucket = max(probs, key=probs.get)
                actual_abs = abs(actual_margin)

                # Find actual bucket
                actual_bucket = "31+"
                for name, low, high in [("1-5",1,5),("6-10",6,10),("11-15",11,15),
                                         ("16-20",16,20),("21-25",21,25),("26-30",26,30)]:
                    if low <= actual_abs <= high:
                        actual_bucket = name
                        break

                bucket_hits[actual_bucket]["total"] += 1
                if predicted_bucket == actual_bucket:
                    bucket_hits[actual_bucket]["correct"] += 1

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
        }

    def save(self, path: Path = None):
        path = path or MODEL_DIR / "margin_model.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "sigma": self.sigma,
                "league_sigmas": self.league_sigmas,
                "league_categories": self.league_categories,
            }, f)
        logger.info(f"Model saved to {path}")

    def load(self, path: Path = None):
        path = path or MODEL_DIR / "margin_model.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.sigma = data["sigma"]
        self.league_sigmas = data.get("league_sigmas", {})
        self.league_categories = data.get("league_categories", {})
        logger.info(f"Model loaded from {path}, global σ={self.sigma:.2f}")
        if self.league_sigmas:
            inv = {v: k for k, v in self.league_categories.items()}
            info = ", ".join(
                f"{inv.get(lid, lid)}={s:.2f}" for lid, s in self.league_sigmas.items()
            )
            logger.info(f"  Per-league σ: {info}")

    def get_league_id(self, league_name: str) -> int:
        """Look up a league_id from a league name. Returns 0 if unknown."""
        return self.league_categories.get(league_name, 0)
