"""Ridge regression backend — simple linear baseline."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from src.model.backends.base import BaseBackend, PreprocessingConfig

_DEFAULTS = dict(alpha=1.0)


class RidgeBackend(BaseBackend):
    def __init__(self, **hyperparams):
        self._hyperparams = {**_DEFAULTS, **hyperparams}
        self._model: Ridge | None = None

    def name(self) -> str:
        return "ridge"

    def preprocessing_config(self) -> PreprocessingConfig:
        return PreprocessingConfig(scale_features=True, onehot_columns=["league_id"])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model = Ridge(**self._hyperparams, random_state=42)
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def get_params(self) -> dict:
        return {"model": self._model, "hyperparams": self._hyperparams}

    @classmethod
    def from_params(cls, params: dict) -> RidgeBackend:
        hp = params.get("hyperparams", _DEFAULTS)
        backend = cls(**hp)
        backend._model = params["model"]
        return backend

    @classmethod
    def search_space(cls, trial) -> dict:
        return {
            "alpha": trial.suggest_float("alpha", 0.001, 100.0, log=True),
        }
