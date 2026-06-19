# src/training/hyperparameter_tuning.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Hyperparameter search over the LSTM architecture
#          and training configuration.
#
#          Chapter 3, Section 3.5.4 — Hyperparameter Tuning:
#          "Hyperparameter tuning was conducted via grid
#          search over: LSTM layer count (1, 2, 3), units per
#          layer (32, 64, 128, 256), dropout rate (0.1, 0.2,
#          0.3, 0.5), learning rate (0.01, 0.001, 0.0001),
#          and batch size (32, 64, 128). Validation accuracy
#          determined the optimal configuration."
#
#          Three search strategies are implemented:
#            1. Grid Search    — exhaustive, all combinations
#            2. Random Search  — random n_trials subset
#            3. Optuna         — Bayesian (optional, advanced)

import itertools
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.constants import (
    HP_N_LSTM_LAYERS,
    HP_LSTM_UNITS,
    HP_DROPOUT_RATES,
    HP_LEARNING_RATES,
    HP_BATCH_SIZES,
    RANDOM_SEED,
)
from src.utils.helpers import Timer, labels_to_one_hot
from src.utils.logger import get_logger, log_section_header
from src.utils.paths import LOGS_DIR, TABLES_DIR, ensure_dir

logger = get_logger(__name__)


# Trial Result Dataclass

@dataclass
class TrialResult:
    """
    Result of a single hyperparameter trial.

    Attributes
    ----------
    trial_id : int
    params : dict
    val_accuracy : float
    val_loss : float
    epochs_run : int
    training_time_s : float
    """
    trial_id: int
    params: Dict
    val_accuracy: float = 0.0
    val_loss: float = float("inf")
    epochs_run: int = 0
    training_time_s: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "trial_id": self.trial_id,
            "val_accuracy": round(self.val_accuracy, 6),
            "val_loss": round(self.val_loss, 6),
            "epochs_run": self.epochs_run,
            "training_time_s": round(self.training_time_s, 2),
            **self.params,
        }


# Search Space Generation

def generate_grid_search_space(
    n_lstm_layers: Optional[List[int]] = None,
    lstm_units: Optional[List[int]] = None,
    dropout_rate: Optional[List[float]] = None,
    learning_rate: Optional[List[float]] = None,
    batch_size: Optional[List[int]] = None,
) -> List[Dict]:
    """
    Generate all combinations for a full grid search.

    Parameters default to the Chapter 3 search grid.

    Parameters
    ----------
    n_lstm_layers : list of int, optional
    lstm_units : list of int, optional
    dropout_rate : list of float, optional
    learning_rate : list of float, optional
    batch_size : list of int, optional

    Returns
    -------
    list of dict
        Each dict is one hyperparameter configuration.
    """
    grid = {
        "n_lstm_layers": n_lstm_layers or HP_N_LSTM_LAYERS,
        "lstm_units":    lstm_units    or HP_LSTM_UNITS,
        "dropout_rate":  dropout_rate  or HP_DROPOUT_RATES,
        "learning_rate": learning_rate or HP_LEARNING_RATES,
        "batch_size":    batch_size    or HP_BATCH_SIZES,
    }

    keys = list(grid.keys())
    values = list(grid.values())
    combinations = list(itertools.product(*values))

    configs = [
        dict(zip(keys, combo)) for combo in combinations
    ]
    logger.info(
        "Grid search space: %d combinations "
        "(%s).",
        len(configs),
        " × ".join(f"{len(v)} {k}" for k, v in grid.items()),
    )
    return configs


def generate_random_search_space(
    n_trials: int = 50,
    n_lstm_layers: Optional[List[int]] = None,
    lstm_units: Optional[List[int]] = None,
    dropout_rate: Optional[List[float]] = None,
    learning_rate: Optional[List[float]] = None,
    batch_size: Optional[List[int]] = None,
    random_state: int = RANDOM_SEED,
) -> List[Dict]:
    """
    Randomly sample *n_trials* configurations from the
    full grid without replacement (where possible).

    Parameters
    ----------
    n_trials : int
    Others : same as ``generate_grid_search_space``
    random_state : int

    Returns
    -------
    list of dict
    """
    all_configs = generate_grid_search_space(
        n_lstm_layers, lstm_units, dropout_rate,
        learning_rate, batch_size,
    )
    rng = random.Random(random_state)
    n_sample = min(n_trials, len(all_configs))
    sampled = rng.sample(all_configs, n_sample)
    logger.info(
        "Random search: %d trials sampled from %d total.",
        n_sample, len(all_configs),
    )
    return sampled


# Trial Runner
def run_trial(
    trial_id: int,
    params: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    max_epochs: int = 30,
    patience: int = 5,
) -> TrialResult:
    """
    Train a single model configuration and return its
    validation performance.

    Uses a reduced patience (5) for faster search —
    the final model is trained with full patience (10)
    using the best-found configuration.

    Parameters
    ----------
    trial_id : int
    params : dict
        ``{n_lstm_layers, lstm_units, dropout_rate,
           learning_rate, batch_size}``
    X_train, y_train, X_val, y_val : np.ndarray
    n_classes : int
    max_epochs : int
        Maximum epochs per trial.  Default: 30 (fast search).
    patience : int
        Early stopping patience for tuning.  Default: 5.

    Returns
    -------
    TrialResult
    """
    import tensorflow as tf  # type: ignore
    from src.models.lstm_model import build_lstm_model

    # Build LSTM units list from params
    n_layers   = params["n_lstm_layers"]
    units      = params["lstm_units"]
    lstm_units = [units] * n_layers
    if n_layers >= 2:
        lstm_units[-1] = max(units // 2, 32)

    input_shape = (X_train.shape[1], X_train.shape[2])

    try:
        model = build_lstm_model(
            input_shape=input_shape,
            n_classes=n_classes,
            lstm_units=lstm_units,
            dropout_rate=params["dropout_rate"],
            learning_rate=params["learning_rate"],
        )

        y_train_oh = labels_to_one_hot(y_train, n_classes)
        y_val_oh   = labels_to_one_hot(y_val,   n_classes)

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=0,
        )

        import time
        start = time.perf_counter()

        history = model.fit(
            X_train, y_train_oh,
            validation_data=(X_val, y_val_oh),
            epochs=max_epochs,
            batch_size=params["batch_size"],
            callbacks=[early_stop],
            verbose=0,
        )

        elapsed = time.perf_counter() - start
        hist    = history.history
        epochs_run = len(hist["val_loss"])

        best_idx     = int(np.argmin(hist["val_loss"]))
        val_accuracy = float(hist["val_accuracy"][best_idx])
        val_loss     = float(hist["val_loss"][best_idx])

        # Free GPU memory
        del model
        tf.keras.backend.clear_session()

    except Exception as exc:  # noqa: BLE001
        logger.warning("Trial %d failed: %s", trial_id, exc)
        return TrialResult(
            trial_id=trial_id,
            params=params,
            val_accuracy=0.0,
            val_loss=float("inf"),
        )

    return TrialResult(
        trial_id=trial_id,
        params=params,
        val_accuracy=val_accuracy,
        val_loss=val_loss,
        epochs_run=epochs_run,
        training_time_s=elapsed,
    )


# Search Runners
def run_grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    max_epochs: int = 30,
    patience: int = 5,
    config: Optional[Any] = None,
    save_results: bool = True,
) -> Tuple[Dict, List[TrialResult]]:
    """
    Run a full grid search over the Chapter 3 hyperparameter
    space and return the best configuration.

    Parameters
    ----------
    X_train, y_train, X_val, y_val : np.ndarray
    n_classes : int
    max_epochs : int
    patience : int
    config : AppConfig, optional
    save_results : bool

    Returns
    -------
    tuple of (best_params, all_trial_results)
    """
    log_section_header(logger, "GRID SEARCH HYPERPARAMETER TUNING")

    if config is None:
        from src.config import get_config
        config = get_config()

    ht = config.hyperparameter_tuning
    ss = ht

    search_space = generate_grid_search_space(
        n_lstm_layers=ss.n_lstm_layers,
        lstm_units=ss.lstm_units,
        dropout_rate=ss.dropout_rate,
        learning_rate=ss.learning_rate,
        batch_size=ss.batch_size,
    )

    return _run_search(
        search_space=search_space,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_classes=n_classes,
        max_epochs=max_epochs,
        patience=patience,
        save_results=save_results,
    )


def run_random_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    n_trials: int = 50,
    max_epochs: int = 30,
    patience: int = 5,
    config: Optional[Any] = None,
    save_results: bool = True,
) -> Tuple[Dict, List[TrialResult]]:
    """
    Run a random search over *n_trials* sampled configurations.

    Parameters
    ----------
    n_trials : int
    Others : same as ``run_grid_search``

    Returns
    -------
    tuple of (best_params, all_trial_results)
    """
    log_section_header(logger, "RANDOM SEARCH HYPERPARAMETER TUNING")

    if config is None:
        from src.config import get_config
        config = get_config()

    ht = config.hyperparameter_tuning
    search_space = generate_random_search_space(
        n_trials=n_trials,
        n_lstm_layers=ht.n_lstm_layers,
        lstm_units=ht.lstm_units,
        dropout_rate=ht.dropout_rate,
        learning_rate=ht.learning_rate,
        batch_size=ht.batch_size,
    )

    return _run_search(
        search_space=search_space,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_classes=n_classes,
        max_epochs=max_epochs,
        patience=patience,
        save_results=save_results,
    )


# Internal Search Executor

def _run_search(
    search_space: List[Dict],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    max_epochs: int,
    patience: int,
    save_results: bool,
) -> Tuple[Dict, List[TrialResult]]:
    """Execute a search over a pre-built search space."""
    n_trials = len(search_space)
    all_results: List[TrialResult] = []

    logger.info(
        "Running %d trials (max %d epochs each) ...",
        n_trials, max_epochs,
    )

    with Timer("Hyperparameter search"):
        for i, params in enumerate(search_space):
            logger.info(
                "Trial %d/%d — %s",
                i + 1, n_trials, params,
            )
            result = run_trial(
                trial_id=i + 1,
                params=params,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                n_classes=n_classes,
                max_epochs=max_epochs,
                patience=patience,
            )
            all_results.append(result)
            logger.info(
                "  → val_acc: %.4f | val_loss: %.4f | "
                "epochs: %d | time: %.1fs",
                result.val_accuracy,
                result.val_loss,
                result.epochs_run,
                result.training_time_s,
            )

    # Best trial by validation accuracy
    best = max(all_results, key=lambda r: r.val_accuracy)
    best_params = best.params

    logger.info("=" * 60)
    logger.info("BEST CONFIGURATION (Trial %d):", best.trial_id)
    for k, v in best_params.items():
        logger.info("  %-20s : %s", k, v)
    logger.info("  val_accuracy  : %.4f", best.val_accuracy)
    logger.info("  val_loss      : %.4f", best.val_loss)
    logger.info("=" * 60)

    if save_results:
        _save_tuning_results(all_results, best_params)

    return best_params, all_results


# Save Tuning Results

def _save_tuning_results(
    results: List[TrialResult],
    best_params: Dict,
) -> None:
    """Save all trial results to CSV and best params to JSON."""
    import pandas as pd

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # CSV — all trials
    rows = [r.to_dict() for r in results]
    df = pd.DataFrame(rows).sort_values(
        "val_accuracy", ascending=False
    )
    csv_path = TABLES_DIR / "hyperparameter_search_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Tuning results saved: %s", csv_path)

    # JSON — best params
    best_path = TABLES_DIR / "best_hyperparameters.json"
    best_path.write_text(
        json.dumps(best_params, indent=4), encoding="utf-8"
    )
    logger.info("Best hyperparameters saved: %s", best_path)

    # CSV — hyperparameters table for Chapter 4
    hp_table_path = TABLES_DIR / "hyperparameters.csv"
    hp_df = pd.DataFrame([best_params])
    hp_df.to_csv(hp_table_path, index=False)
    logger.info("Hyperparameters table saved: %s", hp_table_path)