# src/visualization/plots.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Miscellaneous publication-quality plots used
#          across Chapter 3 and Chapter 4 of the report —
#          preprocessing pipeline diagram, precision-recall
#          curves, and class balance charts.

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score  # type: ignore

from src.utils.constants import (
    FIGURE_DPI, FIGURE_SIZE, FIGURE_SIZE_WIDE,
    FONT_SIZE, TITLE_FONT_SIZE, COLOR_PALETTE,
    CLASS_COLORS, NSL_KDD_CLASS_NAMES,
    FIG_PREPROCESSING_PIPELINE,
)
from src.utils.logger import get_logger
from src.utils.paths import FIGURES_DIR, ensure_dir

logger = get_logger(__name__)
warnings.filterwarnings("ignore")


def plot_preprocessing_pipeline(
    output_path: Optional[Path] = None,
) -> Path:
    """
    Generate a flowchart diagram of the preprocessing pipeline
    described in Chapter 3, Section 3.5.2.

    Steps shown:
      Raw Data → Drop Irrelevant Columns → Handle Missing/Inf
      → Remove Duplicates → Map Labels → One-Hot Encode
      → Min-Max Scale → Sliding Window → Train/Val/Test Split

    Returns
    -------
    Path
    """
    steps = [
        ("Raw Dataset\n(NSL-KDD / CICIDS2017 / UNSW-NB15)",
         "#AED6F1"),
        ("Drop Irrelevant Columns\n(difficulty, _split)",
         "#D5F5E3"),
        ("Handle Missing & Infinite Values\n"
         "(mean imputation / mode imputation)",
         "#D5F5E3"),
        ("Remove Duplicate Rows",
         "#D5F5E3"),
        ("Map Labels to Integer Classes\n"
         "(normal=0, DoS=1, Probe=2, R2L=3, U2R=4)",
         "#FDEBD0"),
        ("One-Hot Encode Categorical Features\n"
         "(protocol_type, service, flag)",
         "#FDEBD0"),
        ("Min-Max Scaling [0, 1]\n"
         "(fitted on training set only)",
         "#FDEBD0"),
        ("Sliding Window Sequence Construction\n"
         "(window=10, step=1, label=last timestep)",
         "#F9E79F"),
        ("Stratified Train / Val / Test Split\n"
         "(70% / 15% / 15%)",
         "#F1948A"),
    ]

    n = len(steps)
    box_w, box_h = 4.2, 0.7
    gap = 0.45
    fig_h = n * (box_h + gap) + 1.0
    fig, ax = plt.subplots(figsize=(7, fig_h))
    ax.set_xlim(0, 6)
    ax.set_ylim(-0.3, fig_h)
    ax.axis("off")

    cx = 3.0
    y_positions = []
    for i, (text, color) in enumerate(steps):
        y = fig_h - 0.8 - i * (box_h + gap)
        y_positions.append(y)
        rect = mpatches.FancyBboxPatch(
            (cx - box_w / 2, y - box_h / 2),
            box_w, box_h,
            boxstyle="round,pad=0.06",
            linewidth=1.1,
            edgecolor="#2C3E50",
            facecolor=color,
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(
            cx, y, text,
            ha="center", va="center",
            fontsize=7.5, color="#1A252F",
            zorder=4,
        )

    for i in range(n - 1):
        y_top    = y_positions[i]    - box_h / 2
        y_bottom = y_positions[i + 1] + box_h / 2
        ax.annotate(
            "",
            xy=(cx, y_bottom + 0.02),
            xytext=(cx, y_top - 0.02),
            arrowprops=dict(
                arrowstyle="-|>", color="#2C3E50", lw=1.4
            ),
        )

    ax.set_title(
        "Data Preprocessing Pipeline",
        fontsize=TITLE_FONT_SIZE, fontweight="bold",
        pad=10, color="#1A252F",
    )

    out = output_path or (FIGURES_DIR / FIG_PREPROCESSING_PIPELINE)
    ensure_dir(out)
    fig.tight_layout()
    fig.savefig(str(out), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Preprocessing pipeline diagram saved: %s", out)
    return out


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
    dataset: str = "nsl_kdd",
    model_name: str = "LSTM",
    output_path: Optional[Path] = None,
) -> Path:
    """
    Plot per-class precision-recall curves.

    Parameters
    ----------
    y_true : np.ndarray
    y_prob : np.ndarray
    class_names : list of str, optional
    dataset : str
    model_name : str
    output_path : Path, optional

    Returns
    -------
    Path
    """
    from sklearn.preprocessing import label_binarize  # type: ignore

    names = class_names or (
        NSL_KDD_CLASS_NAMES if dataset == "nsl_kdd" else None
    )
    n_classes = y_prob.shape[1]
    labels    = list(range(n_classes))
    y_bin     = label_binarize(y_true, classes=labels)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    colors = CLASS_COLORS if len(CLASS_COLORS) >= n_classes \
        else sns.color_palette(COLOR_PALETTE, n_classes)

    for i in labels:
        if y_bin[:, i].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(
            y_bin[:, i], y_prob[:, i]
        )
        ap = average_precision_score(y_bin[:, i], y_prob[:, i])
        name = (
            names[i] if names and i < len(names) else str(i)
        )
        ax.plot(
            recall, precision,
            color=colors[i], lw=1.8,
            label=f"{name} (AP = {ap:.4f})",
        )

    ax.set_xlabel("Recall",    fontsize=FONT_SIZE)
    ax.set_ylabel("Precision", fontsize=FONT_SIZE)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_title(
        f"Precision-Recall Curves — {model_name} on "
        f"{dataset.upper().replace('_', '-')}",
        fontsize=TITLE_FONT_SIZE, fontweight="bold", pad=12,
    )
    ax.legend(loc="lower left", fontsize=FONT_SIZE - 2)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = output_path or (
        FIGURES_DIR / f"precision_recall_curve_{dataset}.png"
    )
    ensure_dir(out)
    fig.tight_layout()
    fig.savefig(str(out), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Precision-Recall curves saved: %s", out)
    return out