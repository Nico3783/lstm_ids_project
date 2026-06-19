# src/visualization/architecture_diagrams.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Generate system architecture and data flow
#          diagrams for Chapter 3 and Chapter 4.

import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.utils.constants import FIGURE_DPI, TITLE_FONT_SIZE
from src.utils.logger import get_logger
from src.utils.paths import FIGURES_DIR, ensure_dir

logger = get_logger(__name__)
warnings.filterwarnings("ignore")


def plot_system_architecture(
    output_path: Optional[Path] = None,
) -> Path:
    """
    Generate a high-level system architecture diagram showing
    the end-to-end IDS pipeline from raw data to alert.

    Returns
    -------
    Path
    """
    components = [
        # (label, x, y, w, h, color)
        ("Network Traffic\n(Raw Packets)", 0.05, 0.75, 0.18, 0.14, "#AED6F1"),
        ("Dataset\n(NSL-KDD /\nCICIDS2017 /\nUNSW-NB15)",
         0.05, 0.48, 0.18, 0.18, "#AED6F1"),
        ("Preprocessing\nPipeline",       0.32, 0.62, 0.18, 0.14, "#A9DFBF"),
        ("Sequence\nBuilder\n(window=10)", 0.56, 0.62, 0.16, 0.14, "#F9E79F"),
        ("LSTM Model\n(128→64 units)",    0.56, 0.30, 0.16, 0.14, "#FAD7A0"),
        ("Train / Val\nSplit (70/15/15)", 0.79, 0.62, 0.16, 0.14, "#F9E79F"),
        ("Evaluation\n& Reports",         0.79, 0.30, 0.16, 0.14, "#F1948A"),
        ("IDS Alert /\nPrediction",       0.56, 0.05, 0.16, 0.10, "#E8DAEF"),
    ]

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for label, x, y, w, h, color in components:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02",
            linewidth=1.3,
            edgecolor="#2C3E50",
            facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2, y + h / 2, label,
            ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#1A252F",
        )

    # Arrows (src_xy → dst_xy)
    arrows = [
        ((0.23, 0.82), (0.32, 0.72)),
        ((0.23, 0.57), (0.32, 0.69)),
        ((0.50, 0.69), (0.56, 0.69)),
        ((0.72, 0.69), (0.79, 0.69)),
        ((0.87, 0.62), (0.87, 0.44)),
        ((0.64, 0.62), (0.64, 0.44)),
        ((0.64, 0.30), (0.64, 0.15)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate(
            "",
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>", color="#2C3E50", lw=1.5
            ),
        )

    ax.set_title(
        "Deep Learning IDS — System Architecture",
        fontsize=TITLE_FONT_SIZE + 1,
        fontweight="bold", pad=12,
    )

    out = output_path or (
        FIGURES_DIR / "system_architecture.png"
    )
    ensure_dir(out)
    fig.tight_layout()
    fig.savefig(str(out), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("System architecture diagram saved: %s", out)
    return out


def plot_data_flow_diagram(
    output_path: Optional[Path] = None,
) -> Path:
    """
    Generate a data flow diagram showing how data moves
    through each pipeline stage.

    Returns
    -------
    Path
    """
    stages = [
        ("Raw Files\n(CSV / TXT)",       "#AED6F1"),
        ("Data Loader",                   "#D5F5E3"),
        ("EDA & Validation",              "#D5F5E3"),
        ("Preprocessing\n(clean / encode / scale)", "#FDEBD0"),
        ("Sequence Builder\n(3-D tensors)", "#F9E79F"),
        ("Train / Val / Test Split",      "#F9E79F"),
        ("LSTM Training",                 "#FAD7A0"),
        ("Model Evaluation",              "#F1948A"),
        ("Reports & Figures",             "#E8DAEF"),
    ]

    n = len(stages)
    box_w, box_h = 3.6, 0.65
    gap  = 0.38
    fig_h = n * (box_h + gap) + 1.0
    fig, ax = plt.subplots(figsize=(6.5, fig_h))
    ax.set_xlim(0, 5.5)
    ax.set_ylim(-0.3, fig_h)
    ax.axis("off")

    cx = 2.75
    ys = []
    for i, (text, color) in enumerate(stages):
        y = fig_h - 0.7 - i * (box_h + gap)
        ys.append(y)
        rect = mpatches.FancyBboxPatch(
            (cx - box_w / 2, y - box_h / 2),
            box_w, box_h,
            boxstyle="round,pad=0.05",
            linewidth=1.1,
            edgecolor="#2C3E50",
            facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(cx, y, text,
                ha="center", va="center",
                fontsize=8, color="#1A252F")

    for i in range(n - 1):
        ax.annotate(
            "",
            xy=(cx, ys[i + 1] + box_h / 2 + 0.02),
            xytext=(cx, ys[i] - box_h / 2 - 0.02),
            arrowprops=dict(
                arrowstyle="-|>", color="#2C3E50", lw=1.4
            ),
        )

    ax.set_title(
        "Data Flow Diagram",
        fontsize=TITLE_FONT_SIZE, fontweight="bold", pad=10,
    )

    out = output_path or (FIGURES_DIR / "data_flow_diagram.png")
    ensure_dir(out)
    fig.tight_layout()
    fig.savefig(str(out), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Data flow diagram saved: %s", out)
    return out