# src/tuning/tuner.py

"""Thin wrapper that delegates to the hyperparameter tuning module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.training.hyperparameter_tuning import (
    run_grid_search,
    run_random_search,
)

logger = logging.getLogger(__name__)


def run_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any],
    dataset: str,
    n_classes: int,
    output_dir: str = ".",
    seed: int = 42,
) -> Optional[Dict]:
    """
    Run hyperparameter tuning and return the best configuration.

    Delegates to ``run_grid_search`` from the training module.

    Parameters
    ----------
    X_train, y_train, X_val, y_val : np.ndarray
    config : dict
    dataset : str
    n_classes : int
    output_dir : str
    seed : int

    Returns
    -------
    dict or None
        Best hyperparameter configuration, or None if no improvement.
    """
    max_epochs = config.get("training", {}).get("epochs", 100)
    patience = config.get("training", {}).get("patience", 10)

    logger.info(
        "Starting hyperparameter tuning for %s (max_epochs=%d, patience=%d)",
        dataset, max_epochs, patience,
    )

    try:
        best_params, all_results = run_grid_search(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            n_classes=n_classes,
            max_epochs=max_epochs,
            patience=patience,
            config=config,
            save_results=True,
        )
        return best_params
    except Exception as e:
        logger.warning("Grid search failed: %s — falling back to random search", e)
        best_params, all_results = run_random_search(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            n_classes=n_classes,
            n_trials=10,
            max_epochs=max_epochs,
            patience=patience,
            config=config,
            save_results=True,
        )
        return best_params
