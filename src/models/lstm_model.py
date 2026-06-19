
# src/models/lstm_model.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Define and build the LSTM model architecture
#          described in Chapter 3, Section 3.5.3.
#
#          Architecture (Chapter 3, Section 3.5.3):
#            Input  : (batch, window_size, n_features)
#            LSTM   : 128 units, return_sequences=True,
#                     tanh/sigmoid, Dropout(0.2)
#            LSTM   : 64  units, return_sequences=False,
#                     tanh/sigmoid, Dropout(0.2)
#            Dense  : 32 units, ReLU, L2(λ=0.001)
#            BatchNormalization
#            Dense  : n_classes, Softmax
#
#          Compiled with:
#            Optimizer : Adam (lr=0.001)
#            Loss      : Categorical Cross-Entropy
#            Metrics   : Accuracy, Precision, Recall

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from src.utils.constants import (
    DEFAULT_WINDOW_SIZE,
    NSL_KDD_NUM_FEATURES,
    NSL_KDD_NUM_CLASSES,
    LSTM_LAYER_1_UNITS,
    LSTM_LAYER_2_UNITS,
    DENSE_UNITS,
    DROPOUT_RATE,
    L2_LAMBDA,
    DEFAULT_LEARNING_RATE,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Model Builder

def build_lstm_model(
    input_shape: Tuple[int, int],
    n_classes: int = NSL_KDD_NUM_CLASSES,
    lstm_units: Optional[List[int]] = None,
    dropout_rate: float = DROPOUT_RATE,
    dense_units: int = DENSE_UNITS,
    l2_lambda: float = L2_LAMBDA,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    metrics: Optional[List] = None,
) -> "tf.keras.Model":
    """
    Build and compile the stacked LSTM model.

    Architecture is aligned exactly with Chapter 3,
    Section 3.5.3 — LSTM Model Architecture Design.

    Parameters
    ----------
    input_shape : tuple of (window_size, n_features)
        Shape of a single input sequence (no batch dimension).
        For NSL-KDD: (10, 41).
    n_classes : int
        Number of output classes.
        NSL-KDD: 5 | CICIDS2017: 15 | UNSW-NB15: 10.
    lstm_units : list of int, optional
        Units per LSTM layer.  Defaults to [128, 64].
    dropout_rate : float
        Dropout rate applied after each LSTM layer.
        Default: 0.2 (Chapter 3).
    dense_units : int
        Units in the hidden Dense layer.  Default: 32.
    l2_lambda : float
        L2 regularisation coefficient for the Dense layer.
        Default: 0.001 (Chapter 3).
    learning_rate : float
        Initial Adam learning rate.  Default: 0.001.
    metrics : list, optional
        Keras metric objects/strings.  Defaults to
        [accuracy, Precision, Recall].

    Returns
    -------
    tf.keras.Model
        Compiled Keras model ready for training.

    Examples
    --------
    >>> model = build_lstm_model(
    ...     input_shape=(10, 41),
    ...     n_classes=5,
    ... )
    >>> model.summary()
    """
    # import tensorflow as tf
    from tensorflow.keras import layers, regularizers  # type: ignore

    if lstm_units is None:
        lstm_units = [LSTM_LAYER_1_UNITS, LSTM_LAYER_2_UNITS]

    if len(lstm_units) < 1:
        raise ValueError("lstm_units must contain at least one value.")

    if metrics is None:
        metrics = [
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]

    logger.info("Building LSTM model ...")
    logger.info("  Input shape  : %s", input_shape)
    logger.info("  LSTM units   : %s", lstm_units)
    logger.info("  Dropout rate : %.2f", dropout_rate)
    logger.info("  Dense units  : %d", dense_units)
    logger.info("  L2 lambda    : %.4f", l2_lambda)
    logger.info("  N classes    : %d", n_classes)
    logger.info("  Learning rate: %s", learning_rate)

    model = tf.keras.Sequential(name="LSTM_IDS")

    # ---- Input layer ----
    model.add(
        layers.Input(
            shape=input_shape,
            name="input",
        )
    )

    # ---- LSTM layers ----
    for i, units in enumerate(lstm_units):
        is_last_lstm = i == len(lstm_units) - 1
        return_sequences = not is_last_lstm

        model.add(
            layers.LSTM(
                units=units,
                return_sequences=return_sequences,
                activation="tanh",
                recurrent_activation="sigmoid",
                dropout=dropout_rate,
                recurrent_dropout=0.0,   # CPU-compatible
                name=f"lstm_{i + 1}",
            )
        )
        model.add(
            layers.Dropout(
                rate=dropout_rate,
                name=f"dropout_{i + 1}",
            )
        )
        logger.info(
            "  + LSTM(%d, return_sequences=%s) + Dropout(%.2f)",
            units, return_sequences, dropout_rate,
        )

    # ---- Dense hidden layer ----
    model.add(
        layers.Dense(
            units=dense_units,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_lambda),
            name="dense_hidden",
        )
    )
    logger.info(
        "  + Dense(%d, relu, L2=%.4f)", dense_units, l2_lambda
    )

    # ---- Batch Normalisation ----
    model.add(
        layers.BatchNormalization(name="batch_norm")
    )
    logger.info("  + BatchNormalization")

    # ---- Output layer ----
    model.add(
        layers.Dense(
            units=n_classes,
            activation="softmax",
            name="output",
        )
    )
    logger.info("  + Dense(%d, softmax)", n_classes)

    # ---- Compile ----
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        name="adam",
    )
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=metrics,
    )

    total_params = model.count_params()
    logger.info(
        "LSTM model built — total parameters: {:,}".format(total_params)
    )
    return model


# Standard RNN Baseline (for comparison)

def build_rnn_model(
    input_shape: Tuple[int, int],
    n_classes: int = NSL_KDD_NUM_CLASSES,
    rnn_units: int = 64,
    dropout_rate: float = DROPOUT_RATE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> "tf.keras.Model":
    """
    Build a standard (vanilla) RNN model for comparison
    against the LSTM.

    Used as the RNN baseline in Chapter 4 model comparisons.

    Parameters
    ----------
    input_shape : tuple of (window_size, n_features)
    n_classes : int
    rnn_units : int
        Number of SimpleRNN units.  Default: 64.
    dropout_rate : float
    learning_rate : float

    Returns
    -------
    tf.keras.Model
        Compiled Keras SimpleRNN model.
    """
    # import tensorflow as tf
    from tensorflow.keras import layers  # type: ignore

    logger.info(
        "Building standard RNN model — units: %d, classes: %d",
        rnn_units, n_classes,
    )

    model = tf.keras.Sequential(name="RNN_IDS")
    model.add(layers.Input(shape=input_shape, name="input"))
    model.add(
        layers.SimpleRNN(
            units=rnn_units,
            return_sequences=False,
            activation="tanh",
            dropout=dropout_rate,
            name="rnn_1",
        )
    )
    model.add(layers.Dropout(rate=dropout_rate, name="dropout_1"))
    model.add(layers.Dense(32, activation="relu", name="dense_hidden"))
    model.add(layers.BatchNormalization(name="batch_norm"))
    model.add(
        layers.Dense(n_classes, activation="softmax", name="output")
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    logger.info(
        "RNN model built — parameters: {:,}".format(model.count_params())
    )
    return model


# Model Summary Utilities

def get_model_summary_string(model: "tf.keras.Model") -> str:
    """
    Return the Keras model.summary() output as a string
    for saving to file or logging.

    Parameters
    ----------
    model : tf.keras.Model

    Returns
    -------
    str
    """
    lines: List[str] = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)


def log_model_summary(model: "tf.keras.Model") -> None:
    """
    Log the model architecture summary using the project
    logger so it appears in ``reports/logs/training.log``
    and in Chapter 4 screenshots.

    Parameters
    ----------
    model : tf.keras.Model
    """
    summary = get_model_summary_string(model)
    logger.info("Model Summary:\n%s", summary)


def save_model_summary(
    model: "tf.keras.Model",
    output_path: Optional[str] = None,
) -> str:
    """
    Save the model summary to a text file.

    Parameters
    ----------
    model : tf.keras.Model
    output_path : str, optional
        File path.  Defaults to
        ``reports/metrics/model_summary.txt``.

    Returns
    -------
    str
        Path to the saved file.
    """
    from src.utils.paths import METRICS_DIR, ensure_dir
    from pathlib import Path

    path = Path(output_path or (METRICS_DIR / "model_summary.txt"))
    ensure_dir(path)
    summary = get_model_summary_string(model)
    path.write_text(summary, encoding="utf-8")
    logger.info("Model summary saved: %s", path)
    return str(path)


def _get_layer_output_shape(layer: Any) -> str:
    """Safely get layer output shape, handling Keras 3 PyTorch backend limitations."""
    try:
        # Avoid direct attribute access to prevent PyTorch's __getattr__ raising AttributeError
        if hasattr(layer, "output_shape") and layer.output_shape is not None:
            return str(layer.output_shape)
    except Exception:
        pass
    try:
        if hasattr(layer, "output") and layer.output is not None and hasattr(layer.output, "shape"):
            return str(layer.output.shape)
    except Exception:
        pass
    return "unknown"


def get_model_config_dict(model: "tf.keras.Model") -> Dict:
    """
    Return a JSON-serialisable dictionary describing the
    model architecture and compile settings.

    Used to populate ``models/final/model_metadata.json``.

    Parameters
    ----------
    model : tf.keras.Model

    Returns
    -------
    dict
    """
    # import tensorflow as tf

    config = {
        "model_name": model.name,
        "total_parameters": int(model.count_params()) if hasattr(model, "count_params") else 0,
        "trainable_parameters": int(
            sum(
                int(np.prod(w.shape))
                for w in model.trainable_weights
            )
        ) if hasattr(model, "trainable_weights") else 0,
        "n_layers": len(model.layers) if hasattr(model, "layers") else 0,
        "layers": [
            {
                "name": layer.name if hasattr(layer, "name") else "unknown",
                "type": layer.__class__.__name__,
                "output_shape": _get_layer_output_shape(layer),
                "parameters": int(layer.count_params()) if hasattr(layer, "count_params") else 0,
            }
            for layer in model.layers
        ] if hasattr(model, "layers") else [],
        "optimizer": model.optimizer.__class__.__name__
        if getattr(model, "optimizer", None)
        else "unknown",
        "learning_rate": float(model.optimizer.learning_rate)
        if getattr(model, "optimizer", None) and hasattr(model.optimizer, "learning_rate")
        else None,
        "loss": model.loss if isinstance(getattr(model, "loss", None), str) else str(getattr(model, "loss", None)),
    }
    return config