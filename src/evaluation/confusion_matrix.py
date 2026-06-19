
# src/evaluation/confusion_matrix.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Generate and save publication-quality confusion
#          matrix heatmaps for Chapter 4.

import warnings
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix  # type: ignore

from src.utils.constants import (
    NSL_KDD_CLASS_NAMES,
    FIGURE_DPI,
    FIGURE_SIZE_SQUARE,
    FONT_SIZE,
    TITLE_FONT_SIZE,
    FIG_CONFUSION_MATRIX,
)
from src.utils.logger import get_logger
from src.utils.paths import FIGURES_DIR, ensure_dir

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    dataset: str = "nsl_kdd",
    model_name: str = "LSTM",
    normalize: bool = True,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Generate and save a confusion matrix heatmap.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    class_names : list of str, optional
    dataset : str
    model_name : str
    normalize : bool
        Normalise by true label counts (show percentages).
    output_path : Path, optional

    Returns
    -------
    Path
    """
    names = class_names or (
        NSL_KDD_CLASS_NAMES if dataset == "nsl_kdd" else None
    )
    labels = sorted(np.unique(
        np.concatenate([y_true, y_pred])
    ).tolist())
    display_names = (
        [names[i] for i in labels if names and i < len(names)]
        if names else [str(i) for i in labels]
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        cm_plot = cm.astype(float) / (
            cm.sum(axis=1, keepdims=True) + 1e-9
        )
        fmt, vmax = ".2%", 1.0
    else:
        cm_plot = cm.astype(float)
        fmt, vmax = "d", None

    n = len(labels)
    fig_size = max(FIGURE_SIZE_SQUARE[0], n * 1.2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))

    sns.heatmap(
        cm_plot,
        ax=ax,
        annot=True,
        fmt=fmt if not normalize else ".1%",
        cmap="Blues",
        xticklabels=display_names,
        yticklabels=display_names,
        linewidths=0.5,
        linecolor="white",
        vmin=0,
        vmax=vmax,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
        annot_kws={"size": max(8, FONT_SIZE - 2)},
    )

    ax.set_xlabel("Predicted Label", fontsize=FONT_SIZE, labelpad=10)
    ax.set_ylabel("True Label",      fontsize=FONT_SIZE, labelpad=10)
    ax.set_title(
        f"Confusion Matrix — {model_name} on "
        f"{dataset.upper().replace('_', '-')}",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        pad=14,
    )
    ax.tick_params(axis="both", labelsize=FONT_SIZE - 1)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)

    out_path = output_path or (FIGURES_DIR / FIG_CONFUSION_MATRIX)
    ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved: %s", out_path)
    return out_path


def plot_confusion_matrix_comparison(
    results: dict,
    dataset: str = "nsl_kdd",
    class_names: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Plot confusion matrices for multiple models side by side.

    Parameters
    ----------
    results : dict
        ``{model_name: {"y_true": ..., "y_pred": ...}}``
    dataset : str
    class_names : list of str, optional
    output_path : Path, optional

    Returns
    -------
    Path
    """
    names = class_names or (
        NSL_KDD_CLASS_NAMES if dataset == "nsl_kdd" else None
    )
    n_models = len(results)
    fig, axes = plt.subplots(
        1, n_models,
        figsize=(6 * n_models, 6),
    )
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, data) in zip(axes, results.items()):
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        labels = sorted(np.unique(
            np.concatenate([y_true, y_pred])
        ).tolist())
        display = (
            [names[i] for i in labels if names and i < len(names)]
            if names else [str(i) for i in labels]
        )
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_norm = cm.astype(float) / (
            cm.sum(axis=1, keepdims=True) + 1e-9
        )

        sns.heatmap(
            cm_norm, ax=ax,
            annot=True, fmt=".1%",
            cmap="Blues",
            xticklabels=display,
            yticklabels=display,
            linewidths=0.3,
            cbar=False,
            annot_kws={"size": 7},
        )
        ax.set_title(
            model_name, fontsize=FONT_SIZE, fontweight="bold"
        )
        ax.set_xlabel("Predicted", fontsize=FONT_SIZE - 1)
        ax.set_ylabel("True",      fontsize=FONT_SIZE - 1)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
                 fontsize=FONT_SIZE - 2)
        plt.setp(ax.get_yticklabels(), rotation=0,
                 fontsize=FONT_SIZE - 2)

    fig.suptitle(
        f"Confusion Matrix Comparison — "
        f"{dataset.upper().replace('_', '-')}",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        y=1.02,
    )

    out_path = output_path or (
        FIGURES_DIR / f"confusion_matrix_comparison_{dataset}.png"
    )
    ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Comparison confusion matrix saved: %s", out_path)
    return out_path