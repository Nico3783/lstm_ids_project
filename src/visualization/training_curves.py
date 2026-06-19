
# src/visualization/training_curves.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Plot training accuracy and loss curves from the
#          Keras history object.  Chapter 4 figure.

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.constants import (
    FIGURE_DPI, FIGURE_SIZE_WIDE, FIGURE_SIZE,
    FONT_SIZE, TITLE_FONT_SIZE,
    FIG_TRAINING_ACCURACY, FIG_TRAINING_LOSS,
)
from src.utils.logger import get_logger
from src.utils.paths import FIGURES_DIR, ensure_dir

logger = get_logger(__name__)
warnings.filterwarnings("ignore")


def plot_training_curves(
    history: Dict[str, List[float]],
    model_name: str = "LSTM",
    dataset: str = "nsl_kdd",
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Plot training and validation accuracy + loss curves and
    save as separate high-resolution PNG files.

    Parameters
    ----------
    history : dict
        Keras ``history.history`` dict with keys:
        ``loss``, ``accuracy``, ``val_loss``, ``val_accuracy``.
    model_name : str
    dataset : str
    output_dir : Path, optional

    Returns
    -------
    dict
        ``{"accuracy": path, "loss": path}``
    """
    out_dir = output_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, Path] = {}

    epochs = range(1, len(history.get("loss", [])) + 1)

    # ---- Accuracy curve ----
    if "accuracy" in history:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        ax.plot(
            epochs, history["accuracy"],
            color="#2ecc71", lw=2, label="Train Accuracy",
        )
        if "val_accuracy" in history:
            ax.plot(
                epochs, history["val_accuracy"],
                color="#e74c3c", lw=2, linestyle="--",
                label="Validation Accuracy",
            )
        _style_curve_ax(
            ax,
            title=f"Training Accuracy — {model_name} on "
                  f"{dataset.upper().replace('_', '-')}",
            xlabel="Epoch",
            ylabel="Accuracy",
        )
        acc_path = out_dir / FIG_TRAINING_ACCURACY
        _save(fig, acc_path)
        saved["accuracy"] = acc_path

    # ---- Loss curve ----
    if "loss" in history:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        ax.plot(
            epochs, history["loss"],
            color="#3498db", lw=2, label="Train Loss",
        )
        if "val_loss" in history:
            ax.plot(
                epochs, history["val_loss"],
                color="#e67e22", lw=2, linestyle="--",
                label="Validation Loss",
            )
        _style_curve_ax(
            ax,
            title=f"Training Loss — {model_name} on "
                  f"{dataset.upper().replace('_', '-')}",
            xlabel="Epoch",
            ylabel="Categorical Cross-Entropy Loss",
        )
        loss_path = out_dir / FIG_TRAINING_LOSS
        _save(fig, loss_path)
        saved["loss"] = loss_path

    logger.info(
        "Training curves saved: %s",
        {k: str(v) for k, v in saved.items()},
    )
    return saved


def plot_combined_training_curves(
    history: Dict[str, List[float]],
    model_name: str = "LSTM",
    dataset: str = "nsl_kdd",
    output_path: Optional[Path] = None,
) -> Path:
    """
    Plot accuracy and loss side-by-side in a single figure.

    Parameters
    ----------
    history : dict
    model_name : str
    dataset : str
    output_path : Path, optional

    Returns
    -------
    Path
    """
    epochs = range(1, len(history.get("loss", [])) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE_WIDE)

    # Accuracy
    if "accuracy" in history:
        ax1.plot(epochs, history["accuracy"],
                 color="#2ecc71", lw=2, label="Train")
        if "val_accuracy" in history:
            ax1.plot(epochs, history["val_accuracy"],
                     color="#e74c3c", lw=2, ls="--", label="Validation")
        _style_curve_ax(ax1, "Accuracy", "Epoch", "Accuracy")

    # Loss
    if "loss" in history:
        ax2.plot(epochs, history["loss"],
                 color="#3498db", lw=2, label="Train")
        if "val_loss" in history:
            ax2.plot(epochs, history["val_loss"],
                     color="#e67e22", lw=2, ls="--", label="Validation")
        _style_curve_ax(ax2, "Loss", "Epoch",
                        "Categorical Cross-Entropy")

    fig.suptitle(
        f"Training History — {model_name} on "
        f"{dataset.upper().replace('_', '-')}",
        fontsize=TITLE_FONT_SIZE, fontweight="bold", y=1.02,
    )

    out = output_path or (
        FIGURES_DIR / f"training_curves_combined_{dataset}.png"
    )
    ensure_dir(out)
    fig.tight_layout()
    fig.savefig(str(out), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Combined training curves saved: %s", out)
    return out


def save_training_history_csv(
    history: Dict[str, List[float]],
    output_path: Optional[Path] = None,
) -> Path:
    """Save Keras training history to CSV."""
    from src.utils.paths import LOGS_DIR
    out = output_path or (LOGS_DIR / "training_history.csv")
    ensure_dir(out)
    pd.DataFrame(history).to_csv(out, index_label="epoch")
    logger.info("Training history CSV saved: %s", out)
    return out


def _style_curve_ax(
    ax: plt.Axes, title: str, xlabel: str, ylabel: str
) -> None:
    ax.set_title(title, fontsize=TITLE_FONT_SIZE,
                 fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.legend(fontsize=FONT_SIZE - 1)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=FONT_SIZE - 1)


def _save(fig: plt.Figure, path: Path) -> None:
    ensure_dir(path)
    fig.tight_layout()
    fig.savefig(str(path), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved: %s", path)