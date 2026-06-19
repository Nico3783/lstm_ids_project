"""
tests/test_models.py
Unit tests for LSTM model architecture and baseline model definitions.
Verifies build, compile, forward-pass, and parameter counts.
"""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestLSTMModel:

    def test_build_output_shape(self):
        from src.models.lstm_model import build_lstm_model
        model = build_lstm_model(input_shape=(10, 41), n_classes=5, lstm_units=[64, 32])
        assert model.output_shape == (None, 5)

    def test_parameter_count_positive(self):
        from src.models.lstm_model import build_lstm_model
        model = build_lstm_model(input_shape=(10, 41), n_classes=5)
        assert model.count_params() > 0

    def test_forward_pass_probabilities_sum_to_one(self):
        from src.models.lstm_model import build_lstm_model
        model = build_lstm_model(input_shape=(10, 41), n_classes=5, lstm_units=[32, 16])
        X = np.random.rand(8, 10, 41).astype(np.float32)
        probs = model.predict(X, verbose=0)
        assert probs.shape == (8, 5)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_values_in_range(self):
        from src.models.lstm_model import build_lstm_model
        model = build_lstm_model(input_shape=(10, 41), n_classes=5, lstm_units=[32, 16])
        X = np.random.rand(4, 10, 41).astype(np.float32)
        probs = model.predict(X, verbose=0)
        assert probs.min() >= 0.0 and probs.max() <= 1.0

    def test_rnn_builds(self):
        from src.models.lstm_model import build_rnn_model
        model = build_rnn_model(input_shape=(10, 41), n_classes=5)
        assert model.output_shape == (None, 5)

    def test_multiclass_output_dimensions(self):
        from src.models.lstm_model import build_lstm_model
        for n_cls in [2, 5, 10]:
            model = build_lstm_model(input_shape=(5, 10), n_classes=n_cls, lstm_units=[16])
            assert model.output_shape == (None, n_cls)

    def test_model_summary_string(self):
        from src.models.lstm_model import build_lstm_model, get_model_summary_string
        model = build_lstm_model(input_shape=(10, 10), n_classes=3, lstm_units=[16, 8])
        summary = get_model_summary_string(model)
        assert isinstance(summary, str) and len(summary) > 100

    def test_model_config_dict(self):
        from src.models.lstm_model import build_lstm_model, get_model_config_dict
        model = build_lstm_model(input_shape=(10, 10), n_classes=3, lstm_units=[16, 8])
        cfg = get_model_config_dict(model)
        assert "total_parameters" in cfg
        assert cfg["total_parameters"] > 0


class TestBaselineModels:

    def test_random_forest_builds(self):
        from src.models.baseline_models import build_random_forest
        model = build_random_forest(n_estimators=10)
        assert hasattr(model, "fit") and model.n_estimators == 10

    def test_svm_builds_with_probability(self):
        from src.models.baseline_models import build_svm
        model = build_svm()
        assert hasattr(model, "fit") and model.probability is True

    def test_logistic_regression_builds(self):
        from src.models.baseline_models import build_logistic_regression
        model = build_logistic_regression()
        assert hasattr(model, "fit")

    def test_random_forest_train_predict(self):
        from src.models.baseline_models import (
            build_random_forest, train_baseline, predict_baseline
        )
        X = np.random.rand(200, 41).astype(np.float32)
        y = np.random.randint(0, 5, 200).astype(np.int64)
        model = build_random_forest(n_estimators=5)
        model = train_baseline(model, X, y, "rf_test")
        y_pred, y_prob = predict_baseline(model, X)
        assert y_pred.shape == (200,)
        assert y_prob.shape == (200, 5)

    def test_baseline_accepts_3d_input(self):
        from src.models.baseline_models import (
            build_random_forest, train_baseline, predict_baseline
        )
        X_flat = np.random.rand(200, 41).astype(np.float32)
        y = np.random.randint(0, 5, 200).astype(np.int64)
        model = build_random_forest(n_estimators=5)
        model = train_baseline(model, X_flat, y, "rf_3d")
        X_3d = np.random.rand(50, 10, 41).astype(np.float32)
        y_pred, _ = predict_baseline(model, X_3d, "rf_3d")
        assert y_pred.shape == (50,)

    def test_train_all_baselines_returns_dict(self):
        from src.models.baseline_models import train_all_baselines
        X = np.random.rand(300, 10).astype(np.float32)
        y = np.random.randint(0, 5, 300).astype(np.int64)
        config = {
            "random_forest": {"n_estimators": 5},
            "svm": {"max_iter": 100},
            "logistic_regression": {"max_iter": 100},
        }
        fitted = train_all_baselines(X, y, config=config)
        assert set(fitted.keys()) == {"random_forest", "svm", "logistic_regression"}


class TestModelFactory:

    def test_create_lstm_model(self):
        from src.models.model_factory import create_model
        model = create_model(model_type="lstm", input_shape=(10, 41), n_classes=5)
        assert model.output_shape == (None, 5)

    def test_create_rnn_model(self):
        from src.models.model_factory import create_model
        model = create_model(model_type="rnn", input_shape=(10, 41), n_classes=5)
        assert model.output_shape == (None, 5)

    def test_create_random_forest(self):
        from src.models.model_factory import create_model
        model = create_model(model_type="random_forest")
        assert hasattr(model, "fit")

    def test_invalid_model_type_raises(self):
        from src.models.model_factory import create_model
        with pytest.raises(ValueError, match="Unknown model type"):
            create_model(model_type="transformer")


class TestModelUtils:

    def test_count_parameters(self):
        from src.models.lstm_model import build_lstm_model
        from src.models.model_utils import count_parameters
        model = build_lstm_model(input_shape=(10, 10), n_classes=3, lstm_units=[16, 8])
        params = count_parameters(model)
        assert params["total"] > 0 and params["trainable"] <= params["total"]

    def test_build_model_metadata_keys(self):
        from src.models.lstm_model import build_lstm_model
        from src.models.model_utils import build_model_metadata
        model = build_lstm_model(input_shape=(10, 10), n_classes=3, lstm_units=[16, 8])
        meta = build_model_metadata(
            model=model, dataset="nsl_kdd", n_classes=3,
            input_shape=(10, 10), window_size=10, n_features=10,
        )
        for key in ["model_name", "dataset", "n_classes", "parameters"]:
            assert key in meta