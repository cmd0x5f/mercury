"""Random Forest backend."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.model.backends.base import BaseBackend, PreprocessingConfig

_DEFAULTS = dict(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=5,
)


class RandomForestBackend(BaseBackend):
    def __init__(self, **hyperparams):
        self._hyperparams = {**_DEFAULTS, **hyperparams}
        self._model: RandomForestRegressor | None = None

    def name(self) -> str:
        return "rf"

    def preprocessing_config(self) -> PreprocessingConfig:
        return PreprocessingConfig(scale_features=False, onehot_columns=None)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model = RandomForestRegressor(
            **self._hyperparams,
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def get_params(self) -> dict:
        return {"model": self._model, "hyperparams": self._hyperparams}

    @classmethod
    def from_params(cls, params: dict) -> RandomForestBackend:
        hp = params.get("hyperparams", _DEFAULTS)
        backend = cls(**hp)
        backend._model = params["model"]
        return backend

    @classmethod
    def search_space(cls, trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        }
