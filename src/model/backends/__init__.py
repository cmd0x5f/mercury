"""Backend registry — maps names to backend classes with lazy imports."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.model.backends.base import BaseBackend

DEFAULT_BACKEND = "xgboost"

_REGISTRY: dict[str, tuple[str, str]] = {
    "xgboost": ("src.model.backends.xgboost_backend", "XGBoostBackend"),
    "ridge": ("src.model.backends.ridge_backend", "RidgeBackend"),
    "rf": ("src.model.backends.rf_backend", "RandomForestBackend"),
    "lightgbm": ("src.model.backends.lightgbm_backend", "LightGBMBackend"),
}


def get_backend_class(name: str) -> type[BaseBackend]:
    """Look up a backend class by name. Imports lazily."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown backend '{name}'. Available: {', '.join(sorted(_REGISTRY))}"
        )
    module_path, class_name = _REGISTRY[name]
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def available_backends() -> list[str]:
    """Return names of all registered backends."""
    return sorted(_REGISTRY.keys())


def register(name: str, module_path: str, class_name: str) -> None:
    """Register a new backend."""
    _REGISTRY[name] = (module_path, class_name)
