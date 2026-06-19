
# src/models/baseline_models.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Implement the classical ML baseline models used
#          for comparison with the LSTM in Chapter 4.
#
#          Baselines (Chapter 3, Section 3.5 — Baseline
#          Models for Comparison):
#            1. Random Forest
#            2. Support Vector Machine (SVM)
#            3. Logistic Regression
#
#          Each baseline receives the same preprocessed
#          feature data as the LSTM, but in flattened 2-D
#          form (samples, features) since these models do
#          not accept sequential 3-D input.
#
#          Results feed directly into the Model Comparison
#          table and bar chart generated in Chapter 4.

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger
from src.utils.helpers import Timer, flatten_sequences

logger = get_logger(__name__)


# Model Factories

def build_random_forest(
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    class_weight: str = "balanced",
    random_state: int = 42,
    n_jobs: int = -1,
) -> Any:
    """
    Build a Random Forest classifier.

    Parameters
    ----------
    n_estimators : int
        Number of trees.  Default: 100.
    max_depth : int, optional
        Maximum tree depth.  None = unlimited.
    class_weight : str
        ``"balanced"`` applies inverse-frequency weights
        matching the LSTM class weighting strategy.
    random_state : int
    n_jobs : int
        Parallel jobs.  -1 uses all CPU cores.

    Returns
    -------
    sklearn.ensemble.RandomForestClassifier
    """
    from sklearn.ensemble import RandomForestClassifier  # type: ignore

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    logger.info(
        "Random Forest built — estimators: %d, max_depth: %s, "
        "class_weight: %s.",
        n_estimators, max_depth, class_weight,
    )
    return model


def build_svm(
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str = "scale",
    max_iter: int = 1000,
    class_weight: str = "balanced",
    random_state: int = 42,
) -> Any:
    """
    Build a Support Vector Machine classifier.

    Parameters
    ----------
    kernel : str
        SVM kernel.  Default: ``"rbf"``.
    C : float
        Regularisation parameter.
    gamma : str or float
        Kernel coefficient.  ``"scale"`` = 1/(n_features × X.var()).
    max_iter : int
        Maximum solver iterations.
    class_weight : str
    random_state : int

    Returns
    -------
    sklearn.svm.SVC
    """
    from sklearn.svm import SVC  # type: ignore

    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
        probability=True,     # Needed for ROC-AUC computation
    )
    logger.info(
        "SVM built — kernel: %s, C: %.2f, gamma: %s.",
        kernel, C, gamma,
    )
    return model


def build_logistic_regression(
    max_iter: int = 1000,
    class_weight: str = "balanced",
    solver: str = "lbfgs",
    random_state: int = 42,
) -> Any:
    """
    Build a Logistic Regression classifier.

    Parameters
    ----------
    max_iter : int
    class_weight : str
    solver : str
        Optimisation algorithm.  ``"lbfgs"`` supports
        multinomial multiclass natively (the only supported
        mode in scikit-learn >= 1.5; ``multi_class`` was
        removed in that release).
    random_state : int

    Returns
    -------
    sklearn.linear_model.LogisticRegression
    """
    from sklearn.linear_model import LogisticRegression  # type: ignore

    model = LogisticRegression(
        max_iter=max_iter,
        class_weight=class_weight,
        solver=solver,
        random_state=random_state,
        n_jobs=-1,
    )
    logger.info(
        "Logistic Regression built — solver: %s, max_iter: %d.",
        solver, max_iter,
    )
    return model


# Training

def train_baseline(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = "baseline",
) -> Any:
    """
    Fit a baseline model on training data.

    Accepts 3-D LSTM-format input and automatically flattens
    it to 2-D by taking the last timestep slice, which is
    the most representative record in each window.

    Parameters
    ----------
    model : sklearn estimator
    X_train : np.ndarray
        Training array — 3-D (samples, window, features) or
        2-D (samples, features).
    y_train : np.ndarray
        1-D training labels.
    model_name : str
        Display name for logging.

    Returns
    -------
    fitted estimator
    """
    X_flat = _ensure_2d(X_train, model_name)

    logger.info(
        "Training %s — X: %s, classes: %s ...",
        model_name, X_flat.shape,
        sorted(np.unique(y_train).tolist()),
    )

    with Timer(f"{model_name} training"):
        model.fit(X_flat, y_train)

    logger.info("%s training complete.", model_name)
    return model


def train_all_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Build and train all three baseline models.

    Parameters
    ----------
    X_train : np.ndarray
        Training data (3-D or 2-D).
    y_train : np.ndarray
        Training labels.
    config : dict, optional
        Override default hyperparameters.  Keys:
        ``random_forest``, ``svm``, ``logistic_regression``.

    Returns
    -------
    dict
        ``{model_name: fitted_model}``
    """
    cfg = config or {}

    rf_cfg  = cfg.get("random_forest",       {})
    svm_cfg = cfg.get("svm",                 {})
    lr_cfg  = cfg.get("logistic_regression", {})

    models = {
        "random_forest":       build_random_forest(**rf_cfg),
        "svm":                 build_svm(**svm_cfg),
        "logistic_regression": build_logistic_regression(**lr_cfg),
    }

    fitted: Dict[str, Any] = {}
    for name, model in models.items():
        fitted[name] = train_baseline(model, X_train, y_train, name)

    return fitted


# Prediction

def predict_baseline(
    model: Any,
    X: np.ndarray,
    model_name: str = "baseline",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Generate class predictions and probability scores for a
    fitted baseline model.

    Parameters
    ----------
    model : fitted sklearn estimator
    X : np.ndarray
        Feature array (3-D or 2-D).
    model_name : str

    Returns
    -------
    tuple of (y_pred, y_prob)
        y_pred : 1-D integer class predictions.
        y_prob : 2-D probability matrix (n_samples, n_classes),
                 or None if the model lacks ``predict_proba``.
    """
    X_flat = _ensure_2d(X, model_name)
    y_pred = model.predict(X_flat)

    y_prob: Optional[np.ndarray] = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_flat)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "predict_proba failed for %s: %s", model_name, exc
            )

    return y_pred.astype(np.int64), y_prob


# Save / Load

def save_all_baselines(
    fitted_models: Dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Save all fitted baseline models to *output_dir*.

    Parameters
    ----------
    fitted_models : dict
        ``{model_name: fitted_model}``
    output_dir : Path
        Destination directory (``models/baselines/``).
    """
    from src.utils.serialization import save_baseline_model

    for name, model in fitted_models.items():
        save_baseline_model(model, name, output_dir)
    logger.info(
        "All %d baseline models saved to: %s",
        len(fitted_models), output_dir,
    )


def load_all_baselines(input_dir: Path) -> Dict[str, Any]:
    """
    Load all baseline models from *input_dir*.

    Parameters
    ----------
    input_dir : Path

    Returns
    -------
    dict
        ``{model_name: fitted_model}``
    """
    from src.utils.serialization import load_baseline_model
    from src.utils.constants import (
        RANDOM_FOREST_PKL, SVM_PKL, LOGISTIC_REGRESSION_PKL,
    )

    names = ["random_forest", "svm", "logistic_regression"]
    loaded: Dict[str, Any] = {}
    for name in names:
        try:
            loaded[name] = load_baseline_model(name, input_dir)
        except FileNotFoundError:
            logger.warning(
                "Baseline model not found: %s — skipping.", name
            )
    return loaded


# Internal Helpers

def _ensure_2d(X: np.ndarray, name: str = "") -> np.ndarray:
    """
    Flatten 3-D input to 2-D for sklearn baseline models.

    Uses the last timestep of each window as the feature
    vector — this is the same record whose label is assigned
    to the sequence (label_position='last'), making it the
    most semantically consistent choice for flat classifiers.

    Parameters
    ----------
    X : np.ndarray
        2-D or 3-D input array.
    name : str

    Returns
    -------
    np.ndarray
        2-D array (n_samples, n_features).
    """
    if X.ndim == 3:
        # Use last timestep: (n_samples, window, features)
        # → (n_samples, features)
        X_2d = X[:, -1, :]
        logger.debug(
            "%s: 3-D input %s → last-timestep 2-D %s.",
            name, X.shape, X_2d.shape,
        )
        return X_2d
    elif X.ndim == 2:
        return X
    else:
        raise ValueError(
            f"Expected 2-D or 3-D input for {name}, "
            f"got {X.ndim}-D shape {X.shape}."
        )