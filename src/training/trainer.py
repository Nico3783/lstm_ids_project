
# src/training/trainer.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Full training orchestration for both the LSTM
#          model and all baseline classifiers.
#
#          Responsibilities:
#            - One-hot encode y_train/y_val for Keras
#            - Compute and apply class weights
#            - Build callbacks via callbacks.py
#            - Call model.fit with all configured settings
#            - Save the trained model and all artifacts
#            - Log training history and elapsed time
#            - Return a TrainingResult dataclass for the
#              evaluation pipeline to consume
#
#          Aligned with Chapter 3, Section 3.5.4 —
#          Model Training and Optimisation.

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger, log_section_header
from src.utils.helpers import Timer, get_timestamp, labels_to_one_hot
from src.utils.paths import (
    FINAL_MODEL_DIR,
    CHECKPOINTS_DIR,
    BASELINES_DIR,
    LOGS_DIR,
    ensure_dir,
)
from src.utils.serialization import (
    save_keras_model,
    save_preprocessing_artifacts,
    save_baseline_results,
)
from src.training.callbacks import build_callbacks_from_config
from src.training.class_weights import get_class_weights

logger = get_logger(__name__)


# Result Dataclass

@dataclass
class TrainingResult:
    """
    Container for everything produced by a single training run.

    Passed from the trainer to the evaluation module so that
    results can be reported without re-loading artifacts.

    Attributes
    ----------
    model_type : str
    model : fitted model
    history : dict
        Keras training history (loss, accuracy per epoch).
    best_epoch : int
        Epoch at which the best val_loss was achieved.
    best_val_loss : float
    best_val_accuracy : float
    training_time_s : float
        Wall-clock training duration in seconds.
    model_path : Path
        Saved model file path.
    metadata : dict
    """
    model_type: str
    model: Any
    history: Dict[str, List[float]] = field(default_factory=dict)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    best_val_accuracy: float = 0.0
    training_time_s: float = 0.0
    model_path: Optional[Path] = None
    metadata: Dict = field(default_factory=dict)


# LSTM Trainer

def train_lstm(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    epochs: int = 100,
    batch_size: int = 64,
    use_class_weights: bool = True,
    config: Optional[Any] = None,
    save_dir: Optional[Path] = None,
    dataset: str = "nsl_kdd",
    resume: bool = False,
    checkpoint_dir: Optional[Path] = None,
) -> TrainingResult:
    """
    Train the LSTM model on sequence data.

    Parameters
    ----------
    model : tf.keras.Model
        Compiled LSTM model from ``build_lstm_model()``.
    X_train : np.ndarray
        3-D training array (samples, window, features).
    y_train : np.ndarray
        1-D integer training labels.
    X_val : np.ndarray
        3-D validation array.
    y_val : np.ndarray
        1-D integer validation labels.
    n_classes : int
        Number of output classes.
    epochs : int
        Maximum training epochs.  Default: 100.
    batch_size : int
        Mini-batch size.  Default: 64.
    use_class_weights : bool
        Apply inverse-frequency class weights.
    config : AppConfig, optional
    save_dir : Path, optional
        Directory for saving the final model.
    dataset : str
    resume : bool
        If True, attempt to resume training from a checkpoint.
    checkpoint_dir : Path, optional
        Dataset-specific directory for checkpoints and logs.

    Returns
    -------
    TrainingResult
    """
    log_section_header(logger, "LSTM MODEL TRAINING")

    if config is None:
        from src.config import get_config
        config = get_config()

    # ---- Check checkpoints for resume capability ----
    initial_epoch = 0
    checkpoint_loaded = False
    
    if resume:
        ckpt_path = Path(config.training.model_checkpoint.filepath)
        if not ckpt_path.is_absolute():
            from src.utils.paths import PROJECT_ROOT
            ckpt_path = PROJECT_ROOT / ckpt_path

        if ckpt_path.exists():
            logger.info("Checkpoint found at %s. Attempting to resume training...", ckpt_path)
            try:
                from src.utils.serialization import load_keras_model
                model = load_keras_model(ckpt_path)
                checkpoint_loaded = True

                # Determine initial epoch from history CSV
                csv_path = Path(config.training.csv_logger_filepath)
                if not csv_path.is_absolute():
                    from src.utils.paths import PROJECT_ROOT
                    csv_path = PROJECT_ROOT / csv_path

                if csv_path.exists() and csv_path.stat().st_size > 0:
                    with open(csv_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    if len(lines) > 1:
                        initial_epoch = len(lines) - 1
                        logger.info("Resuming training from epoch %d", initial_epoch + 1)
            except Exception as e:
                logger.warning("Could not load checkpoint or determine initial epoch: %s. Starting from scratch.", e)
        else:
            logger.info("No checkpoint found at %s. Starting training from scratch.", ckpt_path)

    # ---- One-hot encode labels for categorical_crossentropy ----
    y_train_oh = labels_to_one_hot(y_train, n_classes=n_classes)
    y_val_oh   = labels_to_one_hot(y_val,   n_classes=n_classes)
    logger.info(
        "Labels one-hot encoded — train: %s, val: %s.",
        y_train_oh.shape, y_val_oh.shape,
    )

    # ---- Class weights ----
    class_weight_dict: Optional[Dict[int, float]] = None
    if use_class_weights:
        class_weight_dict = get_class_weights(y_train)

    # ---- Callbacks ----
    callbacks = build_callbacks_from_config(
        config,
        resume=(resume and checkpoint_loaded and initial_epoch > 0),
        output_dir=checkpoint_dir,
    )

    # ---- Log training configuration ----
    logger.info("Training configuration:")
    logger.info("  Dataset      : %s", dataset)
    logger.info("  Epochs       : %d (max)", epochs)
    logger.info("  Batch size   : %d", batch_size)
    logger.info("  X_train      : %s", X_train.shape)
    logger.info("  X_val        : %s", X_val.shape)
    logger.info("  n_classes    : %d", n_classes)
    logger.info(
        "  Class weights: %s",
        "enabled" if use_class_weights else "disabled",
    )
    if resume and checkpoint_loaded:
        logger.info("  Resume epoch : %d", initial_epoch + 1)

    # ---- Train ----
    start_time = time.perf_counter()

    hist_dict = {}
    if initial_epoch >= epochs:
        logger.info("Training already completed %d epochs (max epochs: %d). Skipping fit.", initial_epoch, epochs)
    else:
        history = model.fit(
            X_train,
            y_train_oh,
            validation_data=(X_val, y_val_oh),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=0,   # Silent — our EpochProgressLogger handles output
            shuffle=True,
            initial_epoch=initial_epoch,
        )
        hist_dict = history.history

    elapsed = time.perf_counter() - start_time

    # ---- Load full history from CSV logger to get accurate best epoch and metrics ----
    import csv
    full_history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    try:
        csv_path = Path(config.training.csv_logger_filepath)
        if not csv_path.is_absolute():
            from src.utils.paths import PROJECT_ROOT
            csv_path = PROJECT_ROOT / csv_path
        
        if csv_path.exists():
            with open(csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for key in full_history.keys():
                        if key in row:
                            full_history[key].append(float(row[key]))
    except Exception as e:
        logger.warning("Could not read complete training history from CSV: %s", e)

    # Fallback to current run's history if CSV reading fails or is empty
    if not full_history["val_loss"]:
        full_history = hist_dict

    val_losses = full_history.get("val_loss", [float("inf")])
    best_epoch = int(np.argmin(val_losses)) + 1
    best_val_loss = float(np.min(val_losses))
    best_val_acc = float(
        full_history.get("val_accuracy", [0.0])[best_epoch - 1]
    )

    logger.info(
        "Training complete — elapsed: %.1f s | "
        "best epoch: %d | best val_loss: %.4f | "
        "best val_accuracy: %.4f.",
        elapsed, best_epoch, best_val_loss, best_val_acc,
    )

    # ---- Save final model ----
    out_dir = save_dir or FINAL_MODEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    from src.utils.constants import FINAL_MODEL_KERAS
    model_path = out_dir / FINAL_MODEL_KERAS
    save_keras_model(model, model_path, save_h5_copy=True)

    # ---- Build and save metadata ----
    from src.models.model_utils import (
        build_model_metadata,
        save_model_metadata,
    )
    metadata = build_model_metadata(
        model=model,
        dataset=dataset,
        n_classes=n_classes,
        input_shape=tuple(X_train.shape[1:]),
        window_size=int(X_train.shape[1]),
        n_features=int(X_train.shape[2]),
        training_params={
            "epochs_run": len(val_losses),
            "max_epochs": epochs,
            "batch_size": batch_size,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_accuracy": best_val_acc,
            "training_time_s": round(elapsed, 2),
            "class_weights_used": use_class_weights,
        },
    )
    save_model_metadata(metadata, out_dir / "model_metadata.json")

    return TrainingResult(
        model_type="lstm",
        model=model,
        history=full_history,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_val_accuracy=best_val_acc,
        training_time_s=elapsed,
        model_path=model_path,
        metadata=metadata,
    )


# Baseline Trainer

def train_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Optional[Any] = None,
    save_dir: Optional[Path] = None,
) -> Dict[str, TrainingResult]:
    """
    Build, train, and save all baseline classifiers.

    Parameters
    ----------
    X_train : np.ndarray
        3-D or 2-D training array.
    y_train : np.ndarray
        1-D integer training labels.
    config : AppConfig, optional
    save_dir : Path, optional
        Directory to save baseline models.

    Returns
    -------
    dict
        ``{model_name: TrainingResult}``
    """
    log_section_header(logger, "BASELINE MODEL TRAINING")

    if config is None:
        from src.config import get_config
        config = get_config()

    out_dir = save_dir or BASELINES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    from src.models.baseline_models import (
        build_random_forest,
        build_svm,
        build_logistic_regression,
        train_baseline,
        save_all_baselines,
    )

    raw_cfg = config.raw.get("baselines", {})

    results: Dict[str, TrainingResult] = {}

    baseline_defs = {
        "random_forest": build_random_forest(
            **{
                k: v for k, v in raw_cfg.get(
                    "random_forest", {}
                ).items()
                if k != "enabled"
            }
        ),
        "svm": build_svm(
            **{
                k: v for k, v in raw_cfg.get("svm", {}).items()
                if k != "enabled"
            }
        ),
        "logistic_regression": build_logistic_regression(
            **{
                k: v for k, v in raw_cfg.get(
                    "logistic_regression", {}
                ).items()
                if k != "enabled"
            }
        ),
    }

    fitted: Dict[str, Any] = {}
    for name, model in baseline_defs.items():
        logger.info("Training baseline: %s ...", name)
        start = time.perf_counter()
        trained = train_baseline(model, X_train, y_train, name)
        elapsed = time.perf_counter() - start
        fitted[name] = trained

        results[name] = TrainingResult(
            model_type=name,
            model=trained,
            training_time_s=elapsed,
            metadata={"model_type": name, "training_time_s": elapsed},
        )
        logger.info(
            "%s trained in %.1f s.", name, elapsed
        )

    save_all_baselines(fitted, out_dir)
    logger.info(
        "All baselines trained and saved to: %s", out_dir
    )
    return results


# Combined Pipeline

def run_full_training(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    dataset: str = "nsl_kdd",
    config: Optional[Any] = None,
    resume: bool = False,
    output_dir: Optional[Path] = None,
) -> Dict[str, TrainingResult]:
    """
    Run complete training for both the LSTM and all baseline
    models in a single call.

    Called by ``run_pipeline.py`` after the preprocessing and
    splitting stages are complete.

    Parameters
    ----------
    X_train, X_val, X_test : np.ndarray
    y_train, y_val, y_test : np.ndarray
    n_classes : int
    dataset : str
    config : AppConfig, optional
    resume : bool
        If True, attempt to resume training from a checkpoint.
    output_dir : Path, optional
        Dataset-specific output root.  When provided, models
        are saved under ``output_dir/models/`` instead of the
        global ``models/`` directory.

    Returns
    -------
    dict
        ``{model_name: TrainingResult}`` for all models.
    """
    if config is None:
        from src.config import get_config
        config = get_config()

    # Resolve dataset-specific output paths
    if output_dir is not None:
        baselines_out = output_dir / "models" / "baselines"
        final_out = output_dir / "models" / "final"
        ckpts_out = output_dir / "models" / "checkpoints"
    else:
        baselines_out = None
        final_out = None
        ckpts_out = None

    results: Dict[str, TrainingResult] = {}

    # ---- Train baselines ----
    skip_baselines = False
    if resume:
        # Check if baseline models exist
        bl_check = baselines_out or BASELINES_DIR
        rf_path = bl_check / "random_forest.pkl"
        svm_path = bl_check / "svm.pkl"
        lr_path = bl_check / "logistic_regression.pkl"
        if rf_path.exists() and svm_path.exists() and lr_path.exists():
            logger.info("Baseline models found on disk. Loading baseline models.")
            skip_baselines = True
            from src.utils.serialization import load_object
            try:
                results["random_forest"] = TrainingResult(
                    model_type="random_forest",
                    model=load_object(rf_path),
                    metadata={"model_type": "random_forest"}
                )
                results["svm"] = TrainingResult(
                    model_type="svm",
                    model=load_object(svm_path),
                    metadata={"model_type": "svm"}
                )
                results["logistic_regression"] = TrainingResult(
                    model_type="logistic_regression",
                    model=load_object(lr_path),
                    metadata={"model_type": "logistic_regression"}
                )
            except Exception as e:
                logger.warning("Failed to load baseline models: %s. Re-training baselines.", e)
                skip_baselines = False

    if not skip_baselines:
        baseline_results = train_baselines(
            X_train, y_train, config=config,
            save_dir=baselines_out,
        )
        results.update(baseline_results)

    # ---- Build and train LSTM ----
    from src.models.model_factory import create_model

    n_features = X_train.shape[2]
    window_size = X_train.shape[1]

    lstm_model = create_model(
        model_type="lstm",
        input_shape=(window_size, n_features),
        n_classes=n_classes,
        config=config,
    )

    from src.models.model_utils import log_model_summary
    log_model_summary(lstm_model)

    lstm_result = train_lstm(
        model=lstm_model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_classes=n_classes,
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        use_class_weights=config.preprocessing.use_class_weights,
        config=config,
        save_dir=final_out,
        dataset=dataset,
        resume=resume,
        checkpoint_dir=ckpts_out,
    )
    results["lstm"] = lstm_result

    # ---- Summary ----
    log_section_header(logger, "TRAINING SUMMARY")
    for name, res in results.items():
        logger.info(
            "  %-22s  time: %6.1f s  |  best_val_loss: %s  "
            "|  best_val_acc: %s",
            name,
            res.training_time_s,
            f"{res.best_val_loss:.4f}" if res.best_val_loss < float("inf")
            else "N/A",
            f"{res.best_val_accuracy:.4f}"
            if res.best_val_accuracy > 0
            else "N/A",
        )

    return results