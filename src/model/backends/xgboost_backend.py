"""XGBoost backend — port of the original hardcoded model."""

from __future__ import annotations

import numpy as np
import xgboost as xgb

from src.model.backends.base import BaseBackend, PreprocessingConfig

_DEFAULTS = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=1.0,
)


class XGBoostBackend(BaseBackend):
    def __init__(self, **hyperparams):
        self._hyperparams = {**_DEFAULTS, **hyperparams}
        self._model: xgb.XGBRegressor | None = None

    def name(self) -> str:
        return "xgboost"

    def preprocessing_config(self) -> PreprocessingConfig:
        return PreprocessingConfig(scale_features=False, onehot_columns=None)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model = xgb.XGBRegressor(
            **self._hyperparams,
            random_state=42,
        )
        self._model.fit(X, y, verbose=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def get_params(self) -> dict:
        return {"model": self._model, "hyperparams": self._hyperparams}

    @classmethod
    def from_params(cls, params: dict) -> XGBoostBackend:
        hp = params.get("hyperparams", _DEFAULTS)
        backend = cls(**hp)
        backend._model = params["model"]
        return backend

    @classmethod
    def search_space(cls, trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
        }
