# src/training/callbacks.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Build and return the full set of Keras callbacks
#          used during LSTM training, as specified in
#          Chapter 3, Section 3.5.4 — Model Training and
#          Optimisation.
#
#          Callbacks configured:
#            1. EarlyStopping      — patience=10, val_loss
#            2. ModelCheckpoint    — save best model (.keras)
#            3. ReduceLROnPlateau  — halve LR on plateau
#            4. CSVLogger          — write per-epoch metrics
#            5. TensorBoard        — visualise training curves
#            6. EpochProgressLogger — human-readable log lines

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.constants import (
    EARLY_STOPPING_PATIENCE,
    REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR,
    MIN_LEARNING_RATE,
    BEST_MODEL_KERAS,
)
from src.utils.logger import get_logger
from src.utils.paths import (
    CHECKPOINTS_DIR,
    LOGS_DIR,
    TENSORBOARD_LOG_DIR,
    ensure_dir,
)

logger = get_logger(__name__)


import tensorflow as tf  # type: ignore


# Custom Callback — Epoch Progress Logger
class EpochProgressLogger(tf.keras.callbacks.Callback):
    """
    Keras-compatible callback that logs a clean one-line
    summary at the end of every epoch using the project
    logger (instead of Keras' default stdout print).

    The log line format is:
        Epoch 05/100 | loss: 0.1234 | acc: 0.9612 |
        val_loss: 0.1456 | val_acc: 0.9531 | lr: 1.00e-03

    This format is readable in both terminal output and the
    ``reports/logs/training.log`` file, making it ideal for
    Chapter 4 training progress screenshots.
    """

    def __init__(self) -> None:
        super().__init__()
        self._epoch_start: float = 0.0
        self.history: List[Dict[str, float]] = []

    def on_train_begin(self, logs: Optional[Dict] = None) -> None:
        logger.info(
            "Training started — %d epochs × batch size from config.",
            self.params.get("epochs", "?"),
        )

    def on_epoch_begin(
        self, epoch: int, logs: Optional[Dict] = None
    ) -> None:
        self._epoch_start = time.perf_counter()

    def on_epoch_end(
        self, epoch: int, logs: Optional[Dict] = None
    ) -> None:
        logs = logs or {}
        elapsed = time.perf_counter() - self._epoch_start
        total_epochs = self.params.get("epochs", "?")

        parts = [f"Epoch {epoch + 1:03d}/{total_epochs}"]
        for key in ["loss", "accuracy", "val_loss", "val_accuracy"]:
            if key in logs:
                parts.append(f"{key}: {logs[key]:.4f}")

        # Learning rate
        try:
            import tensorflow as tf  # type: ignore
            lr = float(self.model.optimizer.learning_rate)
            parts.append(f"lr: {lr:.2e}")
        except Exception:  # noqa: BLE001
            pass

        parts.append(f"({elapsed:.1f}s)")
        logger.info(" | ".join(parts))
        self.history.append({**logs, "epoch": epoch + 1})

    def on_train_end(self, logs: Optional[Dict] = None) -> None:
        logger.info(
            "Training finished — %d epochs recorded.",
            len(self.history),
        )


# Callback Builder
def build_callbacks(
    checkpoint_path: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    tensorboard_log_dir: Optional[Path] = None,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    early_stopping_monitor: str = "val_loss",
    restore_best_weights: bool = True,
    min_delta: float = 0.0001,
    reduce_lr_patience: int = REDUCE_LR_PATIENCE,
    reduce_lr_factor: float = REDUCE_LR_FACTOR,
    min_lr: float = MIN_LEARNING_RATE,
    enable_tensorboard: bool = True,
    enable_csv_logger: bool = True,
    enable_epoch_logger: bool = True,
    resume: bool = False,
) -> List[Any]:
    """
    Build and return the complete list of Keras callbacks for
    LSTM training.

    All callbacks are configured from the parameters defined
    in Chapter 3, Section 3.5.4 with sensible defaults that
    match ``config.yaml``.

    Parameters
    ----------
    checkpoint_path : Path, optional
        Path for the best-model checkpoint file.
        Defaults to ``models/checkpoints/best_model.keras``.
    log_dir : Path, optional
        Directory for the CSVLogger output.
        Defaults to ``reports/logs/``.
    tensorboard_log_dir : Path, optional
        Directory for TensorBoard event files.
        Defaults to ``reports/logs/tensorboard/``.
    early_stopping_patience : int
        Number of epochs with no improvement before stopping.
        Default: 10 (Chapter 3, Section 3.5.4).
    early_stopping_monitor : str
        Metric to monitor.  Default: ``"val_loss"``.
    restore_best_weights : bool
        Restore weights from the best epoch on stop.
    min_delta : float
        Minimum change to qualify as improvement.
    reduce_lr_patience : int
        Epochs with no improvement before LR reduction.
    reduce_lr_factor : float
        Factor to multiply LR by on reduction.  Default: 0.5.
    min_lr : float
        Floor for the learning rate.  Default: 1e-6.
    enable_tensorboard : bool
        Include TensorBoard callback.
    enable_csv_logger : bool
        Include CSVLogger callback.
    enable_epoch_logger : bool
        Include the custom EpochProgressLogger callback.
    resume : bool
        When True, CSVLogger appends to the history file instead of overwriting.

    Returns
    -------
    list
        List of configured Keras callback instances.
    """
    import tensorflow as tf  # type: ignore

    callbacks: List[Any] = []

    # 1. EarlyStopping
    # Chapter 3: "Training ran for up to 100 epochs ...
    # subject to early stopping with a patience of 10 epochs
    # monitoring validation loss."
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor=early_stopping_monitor,
        patience=early_stopping_patience,
        min_delta=min_delta,
        restore_best_weights=restore_best_weights,
        verbose=0,
    )
    callbacks.append(early_stop)
    logger.info(
        "EarlyStopping — monitor: %s, patience: %d, "
        "restore_best: %s.",
        early_stopping_monitor,
        early_stopping_patience,
        restore_best_weights,
    )

    # 2. ModelCheckpoint
    # Saves the best-generalising model (not the last epoch).
    ckpt_path = checkpoint_path or (CHECKPOINTS_DIR / BEST_MODEL_KERAS)
    ensure_dir(ckpt_path)

    model_ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(ckpt_path),
        monitor=early_stopping_monitor,
        save_best_only=True,
        save_weights_only=False,
        verbose=0,
    )
    callbacks.append(model_ckpt)
    logger.info(
        "ModelCheckpoint — path: %s, monitor: %s.",
        ckpt_path, early_stopping_monitor,
    )

    # 3. ReduceLROnPlateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=early_stopping_monitor,
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        min_lr=min_lr,
        verbose=0,
    )
    callbacks.append(reduce_lr)
    logger.info(
        "ReduceLROnPlateau — factor: %.2f, patience: %d, "
        "min_lr: %s.",
        reduce_lr_factor, reduce_lr_patience, min_lr,
    )

    # 4. CSVLogger
    if enable_csv_logger:
        csv_log_dir = log_dir or LOGS_DIR
        ensure_dir(csv_log_dir / "training_history.csv")
        csv_path = csv_log_dir / "training_history.csv"

        csv_logger = tf.keras.callbacks.CSVLogger(
            filename=str(csv_path),
            separator=",",
            append=resume,
        )
        callbacks.append(csv_logger)
        logger.info("CSVLogger — path: %s (append: %s).", csv_path, resume)

    # 5. TensorBoard
    if enable_tensorboard:
        tb_dir = tensorboard_log_dir or TENSORBOARD_LOG_DIR
        tb_dir.mkdir(parents=True, exist_ok=True)

        try:
            tensorboard = tf.keras.callbacks.TensorBoard(
                log_dir=str(tb_dir),
                histogram_freq=0,
                write_graph=True,
                update_freq="epoch",
            )
            # Eagerly verify that tensorflow.summary is importable,
            # since the callback only imports it lazily at train time.
            import importlib
            importlib.import_module("tensorflow.summary")

            callbacks.append(tensorboard)
            logger.info(
                "TensorBoard — log dir: %s. "
                "Launch with: tensorboard --logdir %s",
                tb_dir, tb_dir,
            )
        except (ImportError, ModuleNotFoundError):
            logger.warning(
                "TensorBoard callback skipped — 'tensorflow.summary' "
                "is not available (expected when using Keras 3 on the "
                "PyTorch backend). Training will continue without "
                "TensorBoard logging."
            )

    # 6. Custom Epoch Progress Logger 
    if enable_epoch_logger:
        callbacks.append(EpochProgressLogger())
        logger.info("EpochProgressLogger attached.")

    logger.info(
        "Callbacks built — %d total: %s.",
        len(callbacks),
        [type(c).__name__ for c in callbacks],
    )
    return callbacks


# Config-Driven Builder
def build_callbacks_from_config(
    config: Optional[Any] = None,
    resume: bool = False,
) -> List[Any]:
    """
    Build callbacks by reading all parameters from the
    ``AppConfig`` singleton.

    This is the entry point used by ``trainer.py`` — no
    hardcoded values anywhere in the training script.

    Parameters
    ----------
    config : AppConfig, optional
        Loaded automatically when not provided.
    resume : bool
        Passed to build_callbacks to determine append mode for CSVLogger.

    Returns
    -------
    list of Keras callbacks
    """
    if config is None:
        from src.config import get_config
        config = get_config()

    tr = config.training
    es = tr.early_stopping
    mc = tr.model_checkpoint
    rl = tr.reduce_lr

    ckpt_path = Path(mc.filepath)
    ensure_dir(ckpt_path)

    return build_callbacks(
        checkpoint_path=ckpt_path,
        log_dir=LOGS_DIR,
        tensorboard_log_dir=TENSORBOARD_LOG_DIR,
        early_stopping_patience=es.patience,
        early_stopping_monitor=es.monitor,
        restore_best_weights=es.restore_best_weights,
        min_delta=es.min_delta,
        reduce_lr_patience=rl.patience,
        reduce_lr_factor=rl.factor,
        min_lr=rl.min_lr,
        enable_tensorboard=True,
        enable_csv_logger=True,
        enable_epoch_logger=True,
        resume=resume,
    )