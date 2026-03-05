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

    def __init__(self):
        self.model: xgb.XGBRegressor | None = None
        self.sigma: float = 12.0  # default, estimated from residuals

    def train(self, games: pd.DataFrame, elo_k: int = 20):
        """Train on historical games.

        Args:
            games: raw games DataFrame with scores
        """
        featured = build_features(games, elo_k=elo_k)

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

        # Estimate sigma from training residuals
        preds = self.model.predict(X)
        residuals = y.values - preds
        self.sigma = float(np.std(residuals))
        logger.info(f"Trained on {len(X)} games, sigma={self.sigma:.2f}")

    def predict_margin(self, features: pd.DataFrame) -> np.ndarray:
        """Predict signed margin for games."""
        return self.model.predict(features[FEATURE_COLS])

    def predict_buckets(self, features: pd.DataFrame) -> list[dict[str, float]]:
        """Predict bucket probabilities for games."""
        mus = self.predict_margin(features)
        return bucket_probabilities_batch(mus, self.sigma)

    def predict_single(self, features: dict) -> dict[str, float]:
        """Predict bucket probabilities for a single game."""
        df = pd.DataFrame([features])
        mu = float(self.model.predict(df[FEATURE_COLS])[0])
        return bucket_probabilities(mu, self.sigma)

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

            sigma = float(np.std(y_train.values - model.predict(X_train)))

            # Check bucket accuracy
            for i, (pred_mu, actual_margin) in enumerate(zip(preds, y_test.values)):
                probs = bucket_probabilities(float(pred_mu), sigma)
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
            pickle.dump({"model": self.model, "sigma": self.sigma}, f)
        logger.info(f"Model saved to {path}")

    def load(self, path: Path = None):
        path = path or MODEL_DIR / "margin_model.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.sigma = data["sigma"]
        logger.info(f"Model loaded from {path}, sigma={self.sigma:.2f}")
