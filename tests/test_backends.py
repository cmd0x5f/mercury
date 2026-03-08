"""Tests for backend registry, preprocessor, and individual backends."""

import numpy as np
import pandas as pd
import pytest

from src.model.backends import available_backends, get_backend_class, DEFAULT_BACKEND
from src.model.backends.base import PreprocessingConfig
from src.model.preprocessor import Preprocessor


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_default_backend(self):
        assert DEFAULT_BACKEND == "xgboost"

    def test_available_backends(self):
        names = available_backends()
        assert "xgboost" in names
        assert "ridge" in names
        assert "rf" in names
        assert "lightgbm" in names

    def test_get_backend_class(self):
        cls = get_backend_class("xgboost")
        assert cls.__name__ == "XGBoostBackend"

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend_class("nonexistent")


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class TestPreprocessor:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "feat_a": [1.0, 2.0, 3.0, 4.0],
            "feat_b": [10.0, 20.0, 30.0, 40.0],
            "league_id": [0, 1, 0, 2],
        })

    def test_passthrough_no_transforms(self, sample_df):
        """Tree backends: just converts to float32 ndarray."""
        config = PreprocessingConfig(scale_features=False, onehot_columns=None)
        proc = Preprocessor(config)
        result = proc.fit_transform(sample_df)
        assert result.dtype == np.float32
        assert result.shape == (4, 3)

    def test_scaling(self, sample_df):
        config = PreprocessingConfig(scale_features=True, onehot_columns=None)
        proc = Preprocessor(config)
        result = proc.fit_transform(sample_df)
        # Scaled features should have ~zero mean
        assert abs(result[:, 0].mean()) < 1e-6
        assert abs(result[:, 1].mean()) < 1e-6

    def test_onehot_expansion(self, sample_df):
        config = PreprocessingConfig(scale_features=False, onehot_columns=["league_id"])
        proc = Preprocessor(config)
        result = proc.fit_transform(sample_df)
        # 2 original + 3 one-hot (league 0, 1, 2) = 5 columns
        assert result.shape == (4, 5)

    def test_scaling_plus_onehot(self, sample_df):
        config = PreprocessingConfig(scale_features=True, onehot_columns=["league_id"])
        proc = Preprocessor(config)
        result = proc.fit_transform(sample_df)
        assert result.shape == (4, 5)

    def test_transform_matches_fit_transform(self, sample_df):
        config = PreprocessingConfig(scale_features=True, onehot_columns=["league_id"])
        proc = Preprocessor(config)
        result1 = proc.fit_transform(sample_df)
        result2 = proc.transform(sample_df)
        np.testing.assert_allclose(result1, result2, atol=1e-6)

    def test_serialization_roundtrip(self, sample_df):
        config = PreprocessingConfig(scale_features=True, onehot_columns=["league_id"])
        proc = Preprocessor(config)
        original = proc.fit_transform(sample_df)

        state = proc.get_state()
        proc2 = Preprocessor.from_state(state)
        restored = proc2.transform(sample_df)
        np.testing.assert_allclose(original, restored, atol=1e-6)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_regression_data(n=200, seed=42):
    """Simple linear regression data for testing backends."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 5).astype(np.float32)
    y = X @ np.array([3, -1, 0.5, 2, -0.5]) + rng.randn(n) * 0.5
    return X, y


# ---------------------------------------------------------------------------
# XGBoost Backend
# ---------------------------------------------------------------------------

class TestXGBoostBackend:
    def test_fit_predict(self):
        cls = get_backend_class("xgboost")
        backend = cls()
        X, y = _make_regression_data()
        backend.fit(X, y)
        preds = backend.predict(X)
        assert preds.shape == (200,)
        # Should fit well on training data
        assert np.corrcoef(y, preds)[0, 1] > 0.9

    def test_params_roundtrip(self):
        cls = get_backend_class("xgboost")
        backend = cls()
        X, y = _make_regression_data()
        backend.fit(X, y)
        preds1 = backend.predict(X)

        params = backend.get_params()
        backend2 = cls.from_params(params)
        preds2 = backend2.predict(X)
        np.testing.assert_array_equal(preds1, preds2)

    def test_preprocessing_config(self):
        cls = get_backend_class("xgboost")
        config = cls().preprocessing_config()
        assert config.scale_features is False
        assert config.onehot_columns is None


# ---------------------------------------------------------------------------
# Ridge Backend
# ---------------------------------------------------------------------------

class TestRidgeBackend:
    def test_fit_predict(self):
        cls = get_backend_class("ridge")
        backend = cls()
        X, y = _make_regression_data()
        backend.fit(X, y)
        preds = backend.predict(X)
        assert preds.shape == (200,)
        assert np.corrcoef(y, preds)[0, 1] > 0.9

    def test_params_roundtrip(self):
        cls = get_backend_class("ridge")
        backend = cls()
        X, y = _make_regression_data()
        backend.fit(X, y)
        preds1 = backend.predict(X)

        params = backend.get_params()
        backend2 = cls.from_params(params)
        preds2 = backend2.predict(X)
        np.testing.assert_allclose(preds1, preds2)

    def test_preprocessing_config(self):
        cls = get_backend_class("ridge")
        config = cls().preprocessing_config()
        assert config.scale_features is True
        assert config.onehot_columns == ["league_id"]


# ---------------------------------------------------------------------------
# Random Forest Backend
# ---------------------------------------------------------------------------

class TestRFBackend:
    def test_fit_predict(self):
        cls = get_backend_class("rf")
        backend = cls()
        X, y = _make_regression_data()
        backend.fit(X, y)
        preds = backend.predict(X)
        assert preds.shape == (200,)
        assert np.corrcoef(y, preds)[0, 1] > 0.9

    def test_params_roundtrip(self):
        cls = get_backend_class("rf")
        backend = cls()
        X, y = _make_regression_data()
        backend.fit(X, y)
        preds1 = backend.predict(X)

        params = backend.get_params()
        backend2 = cls.from_params(params)
        preds2 = backend2.predict(X)
        np.testing.assert_allclose(preds1, preds2, atol=1e-12)


# ---------------------------------------------------------------------------
# LightGBM Backend
# ---------------------------------------------------------------------------

class TestLightGBMBackend:
    @pytest.fixture(autouse=True)
    def _skip_if_no_lightgbm(self):
        pytest.importorskip("lightgbm")

    def test_fit_predict(self):
        cls = get_backend_class("lightgbm")
        backend = cls()
        X, y = _make_regression_data()
        backend.fit(X, y)
        preds = backend.predict(X)
        assert preds.shape == (200,)
        assert np.corrcoef(y, preds)[0, 1] > 0.9

    def test_params_roundtrip(self):
        cls = get_backend_class("lightgbm")
        backend = cls()
        X, y = _make_regression_data()
        backend.fit(X, y)
        preds1 = backend.predict(X)

        params = backend.get_params()
        backend2 = cls.from_params(params)
        preds2 = backend2.predict(X)
        np.testing.assert_allclose(preds1, preds2)


# ---------------------------------------------------------------------------
# Search spaces + custom hyperparams
# ---------------------------------------------------------------------------

class TestSearchSpace:
    def test_xgboost_search_space_keys(self):
        import optuna
        study = optuna.create_study()
        trial = study.ask()
        cls = get_backend_class("xgboost")
        params = cls.search_space(trial)
        assert "n_estimators" in params
        assert "learning_rate" in params
        assert "max_depth" in params

    def test_ridge_search_space_keys(self):
        import optuna
        study = optuna.create_study()
        trial = study.ask()
        cls = get_backend_class("ridge")
        params = cls.search_space(trial)
        assert "alpha" in params
        assert params["alpha"] > 0

    def test_custom_hyperparams_override_defaults(self):
        cls = get_backend_class("xgboost")
        backend = cls(n_estimators=50, max_depth=3)
        assert backend._hyperparams["n_estimators"] == 50
        assert backend._hyperparams["max_depth"] == 3
        # defaults still present for unset params
        assert "learning_rate" in backend._hyperparams

    def test_custom_hyperparams_roundtrip_save_load(self):
        cls = get_backend_class("xgboost")
        backend = cls(n_estimators=50, max_depth=3)
        X, y = _make_regression_data()
        backend.fit(X, y)

        params = backend.get_params()
        restored = cls.from_params(params)
        assert restored._hyperparams["n_estimators"] == 50
        assert restored._hyperparams["max_depth"] == 3
