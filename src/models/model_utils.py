
# src/models/model_utils.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Model inspection utilities — parameter counting,
#          architecture diagram generation, checkpoint
#          management, and model metadata serialisation.
#          These helpers support Chapter 4 documentation by
#          generating the model summary screenshot and the
#          LSTM architecture figure.

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.logger import get_logger
from src.utils.paths import (
    FIGURES_DIR,
    METRICS_DIR,
    CHECKPOINTS_DIR,
    ensure_dir,
)
from src.utils.constants import (
    FIGURE_DPI,
    FIGURE_SIZE,
    FONT_SIZE,
    TITLE_FONT_SIZE,
    FIG_LSTM_ARCHITECTURE,
)

logger = get_logger(__name__)


# Parameter Counting

def count_parameters(model: Any) -> Dict[str, int]:
    """
    Count total, trainable, and non-trainable parameters.

    Parameters
    ----------
    model : tf.keras.Model

    Returns
    -------
    dict
        Keys: ``total``, ``trainable``, ``non_trainable``.
    """
    total       = int(model.count_params())
    trainable   = int(
        sum(np.prod(w.shape) for w in model.trainable_weights)
    )
    non_trainable = total - trainable

    logger.info(
        "Parameters — total: {:,} | trainable: {:,} | "
        "non-trainable: {:,}".format(total, trainable, non_trainable)
    )
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": non_trainable,
    }


# Model Summary Logging

def log_model_summary(model: Any) -> None:
    """
    Log the Keras model summary through the module logger.

    Captures the output of ``model.summary()`` into a string
    buffer and emits each line at INFO level so that it appears
    in the project log file alongside the rest of training output.

    Parameters
    ----------
    model : tf.keras.Model
    """
    import io
    buf = io.StringIO()
    model.summary(print_fn=lambda line: buf.write(line + "\n"))
    summary_text = buf.getvalue()
    logger.info("Model summary:\n%s", summary_text)


# Model Metadata

def build_model_metadata(
    model: Any,
    dataset: str,
    n_classes: int,
    input_shape: tuple,
    window_size: int,
    n_features: int,
    training_params: Optional[Dict] = None,
) -> Dict:
    """
    Build a comprehensive metadata dictionary that is saved
    alongside the model in ``models/final/model_metadata.json``.

    This metadata allows the inference pipeline to reconstruct
    the exact input specification without re-running training.

    Parameters
    ----------
    model : tf.keras.Model
    dataset : str
    n_classes : int
    input_shape : tuple
    window_size : int
    n_features : int
    training_params : dict, optional

    Returns
    -------
    dict
    """
    from src.models.lstm_model import get_model_config_dict
    from src.utils.helpers import get_timestamp

    param_counts = count_parameters(model)
    model_cfg    = get_model_config_dict(model)

    metadata = {
        "model_name": model.name,
        "saved_at": get_timestamp(),
        "dataset": dataset,
        "n_classes": n_classes,
        "input_shape": list(input_shape),
        "window_size": window_size,
        "n_features": n_features,
        "parameters": param_counts,
        "architecture": model_cfg,
        "training": training_params or {},
    }
    return metadata


def save_model_metadata(
    metadata: Dict,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Save model metadata as a formatted JSON file.

    Parameters
    ----------
    metadata : dict
    output_path : Path, optional
        Defaults to ``models/final/model_metadata.json``.

    Returns
    -------
    Path
    """
    from src.utils.serialization import save_metadata
    from src.utils.paths import FINAL_MODEL_DIR
    from src.utils.constants import MODEL_METADATA_JSON

    path = output_path or (FINAL_MODEL_DIR / MODEL_METADATA_JSON)
    save_metadata(metadata, path)
    return path


# Architecture Diagram

def plot_lstm_architecture(
    input_shape: tuple = (10, 41),
    n_classes: int = 5,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Generate a clean schematic diagram of the LSTM architecture
    described in Chapter 3, Section 3.5.3.

    The diagram shows each layer as a labelled box with
    output shape and key hyperparameters, connected by
    directional arrows — suitable for the Chapter 3 /
    Chapter 4 methodology figures.

    Parameters
    ----------
    input_shape : tuple
        (window_size, n_features).
    n_classes : int
    output_path : Path, optional

    Returns
    -------
    Path
        Saved figure path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch

    window_size, n_features = input_shape

    # Layer definitions: (label, sublabel, color)
    layers_info = [
        (
            "Input Layer",
            f"Shape: ({window_size}, {n_features})\n"
            f"{window_size} timesteps × {n_features} features",
            "#AED6F1",
        ),
        (
            "LSTM Layer 1",
            "128 units | return_sequences=True\n"
            "tanh / sigmoid | Dropout(0.2)",
            "#A9DFBF",
        ),
        (
            "LSTM Layer 2",
            "64 units | return_sequences=False\n"
            "tanh / sigmoid | Dropout(0.2)",
            "#A9DFBF",
        ),
        (
            "Dense Layer",
            "32 units | ReLU\n"
            "L2 regularisation (λ=0.001)",
            "#F9E79F",
        ),
        (
            "Batch Normalisation",
            "Normalise activations\n"
            "Stabilise training",
            "#FAD7A0",
        ),
        (
            "Output Layer",
            f"{n_classes} units | Softmax\n"
            "Class probabilities",
            "#F1948A",
        ),
    ]

    n_layers = len(layers_info)
    box_w, box_h = 3.4, 0.8
    gap = 0.55
    fig_h = n_layers * (box_h + gap) + 1.0
    fig, ax = plt.subplots(figsize=(6.5, fig_h))
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.4, fig_h - 0.2)
    ax.axis("off")

    y_positions = []
    cx = 2.5   # centre x

    for i, (title, subtitle, color) in enumerate(layers_info):
        y = fig_h - 1.0 - i * (box_h + gap)
        y_positions.append(y)

        rect = mpatches.FancyBboxPatch(
            (cx - box_w / 2, y - box_h / 2),
            box_w, box_h,
            boxstyle="round,pad=0.05",
            linewidth=1.2,
            edgecolor="#2C3E50",
            facecolor=color,
            zorder=3,
        )
        ax.add_patch(rect)

        ax.text(
            cx, y + 0.14,
            title,
            ha="center", va="center",
            fontsize=9, fontweight="bold",
            color="#1A252F", zorder=4,
        )
        ax.text(
            cx, y - 0.18,
            subtitle,
            ha="center", va="center",
            fontsize=6.5,
            color="#2C3E50",
            zorder=4,
        )

    # Arrows between layers
    for i in range(n_layers - 1):
        y_top    = y_positions[i]    - box_h / 2
        y_bottom = y_positions[i + 1] + box_h / 2
        ax.annotate(
            "",
            xy=(cx, y_bottom + 0.02),
            xytext=(cx, y_top - 0.02),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#2C3E50",
                lw=1.5,
            ),
            zorder=2,
        )

    ax.set_title(
        "LSTM-IDS Model Architecture",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        pad=10,
        color="#1A252F",
    )

    out_path = output_path or (FIGURES_DIR / FIG_LSTM_ARCHITECTURE)
    ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("LSTM architecture diagram saved: %s", out_path)
    return out_path


# Checkpoint Management

def list_checkpoints(
    checkpoint_dir: Optional[Path] = None,
) -> List[Path]:
    """
    Return a sorted list of checkpoint files in
    *checkpoint_dir*.

    Parameters
    ----------
    checkpoint_dir : Path, optional
        Defaults to ``models/checkpoints/``.

    Returns
    -------
    list of Path
    """
    ckpt_dir = checkpoint_dir or CHECKPOINTS_DIR
    if not ckpt_dir.exists():
        return []

    ckpt_files = sorted(
        list(ckpt_dir.glob("*.keras")) + list(ckpt_dir.glob("*.h5"))
    )
    logger.info(
        "Found %d checkpoint(s) in: %s", len(ckpt_files), ckpt_dir
    )
    return ckpt_files


def get_best_checkpoint(
    checkpoint_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Return the path to the best model checkpoint
    (``best_model.keras``), or None if it does not exist.

    Parameters
    ----------
    checkpoint_dir : Path, optional

    Returns
    -------
    Path or None
    """
    from src.utils.constants import BEST_MODEL_KERAS

    ckpt_dir = checkpoint_dir or CHECKPOINTS_DIR
    best = ckpt_dir / BEST_MODEL_KERAS
    if best.exists():
        logger.info("Best checkpoint found: %s", best)
        return best
    logger.warning("No best checkpoint found in: %s", ckpt_dir)
    return None


def cleanup_old_checkpoints(
    checkpoint_dir: Optional[Path] = None,
    keep_best: bool = True,
    keep_last_n: int = 3,
) -> None:
    """
    Delete old epoch checkpoints keeping only the best model
    and the last *keep_last_n* epoch checkpoints.

    Parameters
    ----------
    checkpoint_dir : Path, optional
    keep_best : bool
        Always keep ``best_model.keras``.
    keep_last_n : int
        Number of most-recent epoch checkpoints to retain.
    """
    from src.utils.constants import BEST_MODEL_KERAS, BEST_MODEL_H5

    ckpt_dir = checkpoint_dir or CHECKPOINTS_DIR
    if not ckpt_dir.exists():
        return

    protected = {BEST_MODEL_KERAS, BEST_MODEL_H5}
    epoch_ckpts = sorted(
        ckpt_dir.glob("checkpoint_epoch_*.keras"),
        key=lambda p: p.stat().st_mtime,
    )

    to_delete = epoch_ckpts[:-keep_last_n] if keep_last_n > 0 else epoch_ckpts

    for ckpt in to_delete:
        if keep_best and ckpt.name in protected:
            continue
        ckpt.unlink()
        logger.debug("Deleted old checkpoint: %s", ckpt.name)

    if to_delete:
        logger.info(
            "Cleaned up %d old checkpoint(s).", len(to_delete)
        )