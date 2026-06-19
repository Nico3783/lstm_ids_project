
# src/models/model_factory.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Unified model creation dispatcher that returns any
#          supported model (LSTM, RNN, baselines) from a
#          single ``create_model()`` call, reading all
#          hyperparameters from the AppConfig singleton.
#
#          This decouples the training module from the model
#          definitions — train.py calls create_model() and
#          never imports lstm_model.py directly, making it
#          trivial to swap architectures via config.yaml.

from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.utils.constants import (
    SUPPORTED_MODELS,
    NSL_KDD_NUM_CLASSES,
    DEFAULT_WINDOW_SIZE,
    NSL_KDD_NUM_FEATURES,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_model(
    model_type: str = "lstm",
    input_shape: Optional[Tuple[int, int]] = None,
    n_classes: int = NSL_KDD_NUM_CLASSES,
    config: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """
    Create and return a model of the specified type.

    Reads architecture hyperparameters from *config* (the
    ``AppConfig`` singleton) when provided, with *kwargs*
    taking precedence for any individual overrides.

    Parameters
    ----------
    model_type : str
        One of ``"lstm"``, ``"rnn"``, ``"random_forest"``,
        ``"svm"``, ``"logistic_regression"``.
    input_shape : tuple of (window_size, n_features), optional
        Required for deep learning models (LSTM, RNN).
        Inferred from config when not given.
    n_classes : int
        Number of output classes.
    config : AppConfig, optional
        Project configuration singleton.  Loaded from disk
        when not provided.
    **kwargs
        Direct hyperparameter overrides (e.g.
        ``learning_rate=0.0001``).

    Returns
    -------
    Any
        Compiled Keras model or unfitted sklearn estimator.

    Raises
    ------
    ValueError
        If *model_type* is not recognised.
    """
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Supported: {SUPPORTED_MODELS}"
        )

    # Load config if not provided
    if config is None:
        from src.config import get_config
        config = get_config()

    # Resolve input shape from config if not given
    if input_shape is None:
        window = config.sequence.window_size
        n_feat  = kwargs.pop("n_features", NSL_KDD_NUM_FEATURES)
        input_shape = (window, n_feat)

    logger.info(
        "Creating model — type: '%s', input: %s, classes: %d.",
        model_type, input_shape, n_classes,
    )

    if model_type == "lstm":
        return _create_lstm(input_shape, n_classes, config, **kwargs)
    elif model_type == "rnn":
        return _create_rnn(input_shape, n_classes, config, **kwargs)
    elif model_type == "random_forest":
        return _create_random_forest(config, **kwargs)
    elif model_type == "svm":
        return _create_svm(config, **kwargs)
    elif model_type == "logistic_regression":
        return _create_logistic_regression(config, **kwargs)

    raise ValueError(f"Unhandled model type: {model_type}")


# Private Factories

def _create_lstm(
    input_shape: Tuple[int, int],
    n_classes: int,
    config: Any,
    **kwargs: Any,
) -> Any:
    from src.models.lstm_model import build_lstm_model

    m = config.model
    lstm_units = kwargs.pop(
        "lstm_units",
        [layer.units for layer in m.lstm_layers],
    )
    return build_lstm_model(
        input_shape=input_shape,
        n_classes=n_classes,
        lstm_units=lstm_units,
        dropout_rate=kwargs.pop("dropout_rate", m.lstm_layers[0].dropout),
        dense_units=kwargs.pop(
            "dense_units",
            m.dense_layers[0].units if m.dense_layers else 32,
        ),
        l2_lambda=kwargs.pop(
            "l2_lambda",
            m.dense_layers[0].l2_regularization if m.dense_layers else 0.001,
        ),
        learning_rate=kwargs.pop("learning_rate", m.learning_rate),
        **kwargs,
    )


def _create_rnn(
    input_shape: Tuple[int, int],
    n_classes: int,
    config: Any,
    **kwargs: Any,
) -> Any:
    from src.models.lstm_model import build_rnn_model

    m = config.model
    return build_rnn_model(
        input_shape=input_shape,
        n_classes=n_classes,
        rnn_units=kwargs.pop("rnn_units", 64),
        dropout_rate=kwargs.pop("dropout_rate", m.lstm_layers[0].dropout),
        learning_rate=kwargs.pop("learning_rate", m.learning_rate),
        **kwargs,
    )


def _create_random_forest(config: Any, **kwargs: Any) -> Any:
    from src.models.baseline_models import build_random_forest

    raw_rf = config.raw.get("baselines", {}).get("random_forest", {})
    return build_random_forest(
        n_estimators=kwargs.pop(
            "n_estimators", raw_rf.get("n_estimators", 100)
        ),
        max_depth=kwargs.pop(
            "max_depth", raw_rf.get("max_depth", None)
        ),
        class_weight=kwargs.pop(
            "class_weight", raw_rf.get("class_weight", "balanced")
        ),
        **kwargs,
    )


def _create_svm(config: Any, **kwargs: Any) -> Any:
    from src.models.baseline_models import build_svm

    raw_svm = config.raw.get("baselines", {}).get("svm", {})
    return build_svm(
        kernel=kwargs.pop("kernel", raw_svm.get("kernel", "rbf")),
        C=kwargs.pop("C", raw_svm.get("C", 1.0)),
        gamma=kwargs.pop("gamma", raw_svm.get("gamma", "scale")),
        max_iter=kwargs.pop("max_iter", raw_svm.get("max_iter", 1000)),
        **kwargs,
    )


def _create_logistic_regression(config: Any, **kwargs: Any) -> Any:
    from src.models.baseline_models import build_logistic_regression

    raw_lr = config.raw.get("baselines", {}).get(
        "logistic_regression", {}
    )
    return build_logistic_regression(
        max_iter=kwargs.pop(
            "max_iter", raw_lr.get("max_iter", 1000)
        ),
        solver=kwargs.pop(
            "solver", raw_lr.get("solver", "lbfgs")
        ),
        **kwargs,
    )