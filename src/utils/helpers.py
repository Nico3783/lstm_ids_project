
# src/utils/helpers.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: General-purpose utility functions used across
#          every stage of the pipeline — seeding, timing,
#          formatting, dataset statistics, memory reporting,
#          and other cross-cutting concerns that do not
#          belong in any single domain module.

import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


# Reproducibility

def set_global_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and TensorFlow to
    ensure fully reproducible results across runs.

    Must be called before any dataset loading, model
    construction, or training begins.  In ``run_pipeline.py``
    this is the very first function executed.

    Parameters
    ----------
    seed : int
        Integer seed value.  Defaults to the project constant
        ``RANDOM_SEED`` (42).

    Notes
    -----
    TensorFlow GPU operations may still introduce non-
    determinism at the CUDA level even with seeds set.
    For fully deterministic GPU training set the environment
    variable ``TF_DETERMINISTIC_OPS=1`` before importing
    TensorFlow.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Import TensorFlow lazily to avoid slow startup when only
    # utility functions are needed (e.g. during unit tests).
    try:
        import tensorflow as tf  # type: ignore
        tf.random.set_seed(seed)
        logger.debug("TensorFlow random seed set to %d.", seed)
    except ImportError:
        logger.warning(
            "TensorFlow not found — skipping TF seed initialisation."
        )

    logger.info("Global random seed set to %d.", seed)


# Timing Utilities

class Timer:
    """
    Context manager and standalone timer for measuring the
    elapsed wall-clock time of any pipeline stage.

    Used throughout the pipeline to log how long each major
    step takes — data loading, preprocessing, training,
    evaluation — which is valuable information for Chapter 4
    discussion of computational cost.

    Examples
    --------
    >>> with Timer("Data preprocessing"):
    ...     preprocess_data()
    # Logs: "Data preprocessing completed in 12.34 s."

    >>> t = Timer("Model training")
    >>> t.start()
    >>> train_model()
    >>> elapsed = t.stop()
    """

    def __init__(self, label: str = "") -> None:
        self.label = label
        self._start: float = 0.0
        self.elapsed: float = 0.0

    def start(self) -> "Timer":
        """Start the timer."""
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        """
        Stop the timer and return elapsed seconds.

        Returns
        -------
        float
            Elapsed time in seconds.
        """
        self.elapsed = time.perf_counter() - self._start
        if self.label:
            logger.info(
                "%s completed in %s.",
                self.label,
                format_duration(self.elapsed),
            )
        return self.elapsed

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds as a human-readable string.

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        Formatted string, e.g. ``"2 h 3 min 12.45 s"``,
        ``"1 min 5.30 s"``, or ``"8.21 s"``.

    Examples
    --------
    >>> format_duration(7392.45)
    '2 h 3 min 12.45 s'
    >>> format_duration(65.3)
    '1 min 5.30 s'
    >>> format_duration(8.21)
    '8.21 s'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours > 0:
        return f"{hours} h {minutes} min {secs:.2f} s"
    if minutes > 0:
        return f"{minutes} min {secs:.2f} s"
    return f"{secs:.2f} s"


# Dataset Statistics Helpers

def compute_class_distribution(
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, int]:
    """
    Compute the count of each unique class in *labels*.

    Parameters
    ----------
    labels : np.ndarray
        1-D integer array of class labels.
    class_names : list of str, optional
        Human-readable names for each class index.  When
        provided the keys in the returned dict use the names
        instead of raw integers.

    Returns
    -------
    dict
        ``{class_label: count}`` sorted by class label.

    Examples
    --------
    >>> compute_class_distribution(
    ...     np.array([0, 0, 1, 2, 1]),
    ...     class_names=["Normal", "DoS", "Probe"]
    ... )
    {'Normal': 2, 'DoS': 2, 'Probe': 1}
    """
    unique, counts = np.unique(labels, return_counts=True)
    distribution: Dict[str, int] = {}
    for cls, count in zip(unique, counts):
        if class_names is not None and int(cls) < len(class_names):
            key = class_names[int(cls)]
        else:
            key = str(int(cls))
        distribution[key] = int(count)
    return distribution


def compute_class_weights(
    labels: np.ndarray,
) -> Dict[int, float]:
    """
    Compute inverse-frequency class weights as described in
    Chapter 3, Section 3.5.4.

    The weight for class *c* is:

        weight_c = n_samples / (n_classes * count_c)

    This is the same formula used by scikit-learn's
    ``compute_class_weight('balanced', ...)``.

    Parameters
    ----------
    labels : np.ndarray
        1-D integer array of training labels (not one-hot).

    Returns
    -------
    dict
        ``{class_int: weight_float}`` for every class present
        in *labels*.

    Examples
    --------
    >>> compute_class_weights(np.array([0, 0, 0, 1, 2]))
    {0: 0.556, 1: 1.667, 2: 1.667}
    """
    unique, counts = np.unique(labels, return_counts=True)
    n_samples = len(labels)
    n_classes = len(unique)
    weights: Dict[int, float] = {}
    for cls, count in zip(unique, counts):
        weights[int(cls)] = float(n_samples / (n_classes * count))
    logger.info("Class weights computed: %s", weights)
    return weights


def summarise_dataset(
    X: np.ndarray,
    y: np.ndarray,
    split_name: str = "dataset",
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Return a summary dictionary for a data split.

    Collects shape, sample count, feature count, class
    distribution, and class balance ratio — the values that
    populate the Dataset Summary table in Chapter 4.

    Parameters
    ----------
    X : np.ndarray
        Feature array of shape ``(n_samples, ...)``
    y : np.ndarray
        Label array of shape ``(n_samples,)``
    split_name : str
        Label for the split, e.g. ``"train"``, ``"test"``.
    class_names : list of str, optional
        Human-readable class names.

    Returns
    -------
    dict
        Summary statistics for the split.
    """
    distribution = compute_class_distribution(y, class_names)
    counts = list(distribution.values())
    majority = max(counts)
    minority = min(counts)
    imbalance_ratio = round(majority / minority, 2) if minority > 0 else float("inf")

    summary = {
        "split": split_name,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[-1]),
        "input_shape": str(X.shape),
        "n_classes": len(distribution),
        "class_distribution": distribution,
        "majority_class_count": majority,
        "minority_class_count": minority,
        "imbalance_ratio": imbalance_ratio,
    }
    return summary


# Memory and System Helpers

def get_memory_usage_mb() -> float:
    """
    Return the current process resident memory usage in MB.

    Falls back to 0.0 if the ``psutil`` package is not
    installed (it is not a required dependency).

    Returns
    -------
    float
        Memory usage in megabytes.
    """
    try:
        import psutil  # type: ignore
        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / (1024 ** 2), 2)
    except ImportError:
        return 0.0


def log_memory_usage(label: str = "") -> None:
    """
    Log the current process memory usage.

    Parameters
    ----------
    label : str, optional
        Context label prepended to the log message.
    """
    mb = get_memory_usage_mb()
    if mb > 0:
        prefix = f"[{label}] " if label else ""
        logger.info("%sMemory usage: %.2f MB", prefix, mb)


# Array / Data Helpers

def flatten_sequences(X: np.ndarray) -> np.ndarray:
    """
    Flatten a 3-D sequence array ``(samples, timesteps, features)``
    to 2-D ``(samples, timesteps * features)`` for use with
    baseline models that do not accept sequential input.

    Parameters
    ----------
    X : np.ndarray
        3-D array of shape ``(n_samples, window_size, n_features)``.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(n_samples, window_size * n_features)``.

    Raises
    ------
    ValueError
        If *X* is not 3-dimensional.
    """
    if X.ndim != 3:
        raise ValueError(
            f"Expected 3-D array (samples, timesteps, features), "
            f"got shape {X.shape}."
        )
    n_samples = X.shape[0]
    return X.reshape(n_samples, -1)


def one_hot_to_labels(y_onehot: np.ndarray) -> np.ndarray:
    """
    Convert a one-hot encoded label matrix back to a 1-D
    integer label array.

    Parameters
    ----------
    y_onehot : np.ndarray
        2-D array of shape ``(n_samples, n_classes)``.

    Returns
    -------
    np.ndarray
        1-D integer array of shape ``(n_samples,)``.
    """
    return np.argmax(y_onehot, axis=1)


def labels_to_one_hot(
    y: np.ndarray,
    n_classes: Optional[int] = None,
) -> np.ndarray:
    """
    Convert a 1-D integer label array to a one-hot matrix.

    Parameters
    ----------
    y : np.ndarray
        1-D integer label array of shape ``(n_samples,)``.
    n_classes : int, optional
        Total number of classes.  Inferred from *y* when None.

    Returns
    -------
    np.ndarray
        2-D float32 array of shape ``(n_samples, n_classes)``.
    """
    if n_classes is None:
        n_classes = int(np.max(y)) + 1
    one_hot = np.zeros((len(y), n_classes), dtype=np.float32)
    one_hot[np.arange(len(y)), y.astype(int)] = 1.0
    return one_hot


def safe_divide(numerator: float, denominator: float) -> float:
    """
    Return numerator / denominator, or 0.0 if denominator is zero.

    Parameters
    ----------
    numerator : float
    denominator : float

    Returns
    -------
    float
    """
    return numerator / denominator if denominator != 0 else 0.0


# JSON / Dict Helpers

def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Serialise *data* to a JSON file at *path*.

    Creates parent directories automatically.  Values that
    are not natively JSON-serialisable (e.g. ``np.int64``,
    ``np.float32``) are converted via ``_json_serialiser``.

    Parameters
    ----------
    data : dict
        Data to serialise.
    path : str or Path
        Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=4, default=_json_serialiser)
    logger.debug("JSON saved to %s", path)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load and return a JSON file as a Python dictionary.

    Parameters
    ----------
    path : str or Path
        Source file path.

    Returns
    -------
    dict

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_serialiser(obj: Any) -> Any:
    """
    JSON serialiser for NumPy scalar types and other
    non-standard Python objects.

    Parameters
    ----------
    obj : Any
        Object to serialise.

    Returns
    -------
    Any
        A JSON-compatible representation.

    Raises
    ------
    TypeError
        If the object type is not handled.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable.")


# Formatting Helpers

def format_metrics_table(metrics: Dict[str, float]) -> str:
    """
    Format a metrics dictionary as a neatly aligned plain-text
    table suitable for logging and classification report files.

    Parameters
    ----------
    metrics : dict
        ``{metric_name: value}`` mapping.

    Returns
    -------
    str
        Multi-line string table.

    Examples
    --------
    >>> print(format_metrics_table({"Accuracy": 0.9812, "F1": 0.9734}))
    Metric                              Value
    ------------------------------------------
    Accuracy                            0.9812
    F1                                  0.9734
    """
    lines = [f"{'Metric':<35} {'Value':>10}", "-" * 47]
    for name, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{name:<35} {value:>10.4f}")
        else:
            lines.append(f"{name:<35} {str(value):>10}")
    return "\n".join(lines)


def get_timestamp() -> str:
    """
    Return a filesystem-safe timestamp string for the current
    moment, used to version output files and log entries.

    Returns
    -------
    str
        Timestamp in format ``YYYYMMDD_HHMMSS``.

    Examples
    --------
    >>> get_timestamp()
    '20250512_143022'
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_banner(title: str) -> None:
    """
    Print a decorated banner to stdout at pipeline startup.

    Used by ``run_pipeline.py`` to produce a clear, visually
    distinctive header in terminal output and pipeline log
    screenshots.

    Parameters
    ----------
    title : str
        Title text to display inside the banner.
    """
    width = 64
    border = "=" * width
    padding = " " * ((width - len(title)) // 2)
    print(f"\n{border}")
    print(f"{padding}{title}")
    print(f"{border}\n")


# Validation Helpers

def validate_array_shapes(
    arrays: Dict[str, np.ndarray],
    expected_ndim: Optional[Dict[str, int]] = None,
) -> None:
    """
    Validate that the provided arrays are non-empty and
    optionally match expected dimensionality.

    Parameters
    ----------
    arrays : dict
        ``{name: array}`` pairs to validate.
    expected_ndim : dict, optional
        ``{name: ndim}`` pairs specifying expected dimensions.

    Raises
    ------
    ValueError
        If any array is empty or has the wrong number of dims.
    """
    for name, arr in arrays.items():
        if arr is None or arr.size == 0:
            raise ValueError(f"Array '{name}' is empty or None.")
        if expected_ndim and name in expected_ndim:
            if arr.ndim != expected_ndim[name]:
                raise ValueError(
                    f"Array '{name}' has {arr.ndim} dimensions, "
                    f"expected {expected_ndim[name]}. "
                    f"Shape: {arr.shape}"
                )
    logger.debug(
        "Array validation passed for: %s",
        {n: str(a.shape) for n, a in arrays.items()},
    )


def check_label_consistency(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """
    Verify that the training, validation, and test sets share
    the same set of class labels.

    A mismatch indicates a stratification error in the split
    step and would cause incorrect model output dimensions.

    Parameters
    ----------
    y_train : np.ndarray
        Training labels.
    y_val : np.ndarray
        Validation labels.
    y_test : np.ndarray
        Test labels.

    Raises
    ------
    ValueError
        If the sets of unique labels do not match.
    """
    train_classes = set(np.unique(y_train).tolist())
    val_classes = set(np.unique(y_val).tolist())
    test_classes = set(np.unique(y_test).tolist())

    if not (train_classes == val_classes == test_classes):
        raise ValueError(
            f"Label mismatch across splits.\n"
            f"  Train classes : {sorted(train_classes)}\n"
            f"  Val classes   : {sorted(val_classes)}\n"
            f"  Test classes  : {sorted(test_classes)}\n"
            "Ensure stratified splitting was applied correctly."
        )
    logger.info(
        "Label consistency check passed — %d classes in all splits.",
        len(train_classes),
    )