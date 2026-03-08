"""Abstract base class for ML model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class PreprocessingConfig:
    """Tells the Preprocessor what transforms to apply before this backend."""
    scale_features: bool = False
    onehot_columns: list[str] | None = None  # e.g. ["league_id"]


class BaseBackend(ABC):
    """Interface that every ML backend must implement."""

    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'xgboost'."""

    @abstractmethod
    def preprocessing_config(self) -> PreprocessingConfig:
        """Return the preprocessing this backend needs."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train on preprocessed features."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for preprocessed features."""

    @abstractmethod
    def get_params(self) -> dict:
        """Return picklable state for serialization."""

    @classmethod
    @abstractmethod
    def from_params(cls, params: dict) -> BaseBackend:
        """Reconstruct a fitted backend from saved params."""

    @classmethod
    def search_space(cls, trial) -> dict:
        """Return hyperparams sampled from an Optuna trial.

        Override in subclasses to define tunable hyperparameters.
        Returns a dict that can be passed as **kwargs to __init__.
        """
        return {}
