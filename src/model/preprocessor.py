"""Configurable preprocessing layer driven by backend's PreprocessingConfig."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.model.backends.base import PreprocessingConfig


class Preprocessor:
    """Transforms a feature DataFrame into a numpy array, optionally scaling
    and one-hot encoding columns based on the backend's config."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self._scaler: StandardScaler | None = None
        self._onehot_categories: dict[str, list] | None = None  # col -> sorted unique vals
        self._feature_columns: list[str] | None = None

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Learn transforms from X and return the transformed array."""
        X = X.copy()
        self._feature_columns = list(X.columns)

        # One-hot encode specified columns
        if self.config.onehot_columns:
            self._onehot_categories = {}
            for col in self.config.onehot_columns:
                if col in X.columns:
                    cats = sorted(X[col].unique())
                    self._onehot_categories[col] = cats
            X = self._apply_onehot(X)

        arr = X.values.astype(np.float32)

        # Scale if requested
        if self.config.scale_features:
            self._scaler = StandardScaler()
            arr = self._scaler.fit_transform(arr).astype(np.float32)

        return arr

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply learned transforms to new data."""
        X = X.copy()

        if self.config.onehot_columns and self._onehot_categories:
            X = self._apply_onehot(X)

        arr = X.values.astype(np.float32)

        if self.config.scale_features and self._scaler is not None:
            arr = self._scaler.transform(arr).astype(np.float32)

        return arr

    def _apply_onehot(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace categorical columns with one-hot columns."""
        for col in list(self.config.onehot_columns or []):
            if col not in X.columns:
                continue
            cats = self._onehot_categories[col]
            for cat in cats:
                X[f"{col}_{cat}"] = (X[col] == cat).astype(np.float32)
            X = X.drop(columns=[col])
        return X

    def get_state(self) -> dict:
        """Serialize preprocessor state for saving."""
        state = {
            "config": {
                "scale_features": self.config.scale_features,
                "onehot_columns": self.config.onehot_columns,
            },
            "feature_columns": self._feature_columns,
            "onehot_categories": self._onehot_categories,
        }
        if self._scaler is not None:
            state["scaler"] = {
                "mean": self._scaler.mean_.tolist(),
                "scale": self._scaler.scale_.tolist(),
                "var": self._scaler.var_.tolist(),
                "n_features": self._scaler.n_features_in_,
                "n_samples": int(self._scaler.n_samples_seen_),
            }
        return state

    @classmethod
    def from_state(cls, state: dict) -> Preprocessor:
        """Reconstruct a fitted Preprocessor from saved state."""
        config = PreprocessingConfig(**state["config"])
        proc = cls(config)
        proc._feature_columns = state.get("feature_columns")
        proc._onehot_categories = state.get("onehot_categories")
        if "scaler" in state:
            s = state["scaler"]
            scaler = StandardScaler()
            scaler.mean_ = np.array(s["mean"])
            scaler.scale_ = np.array(s["scale"])
            scaler.var_ = np.array(s["var"])
            scaler.n_features_in_ = s["n_features"]
            scaler.n_samples_seen_ = s["n_samples"]
            proc._scaler = scaler
        return proc
