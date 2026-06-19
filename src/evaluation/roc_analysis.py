
# src/evaluation/roc_analysis.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Compute and plot ROC curves with AUC scores for
#          all classes and all models.  Chapter 4 figure.

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc  # type: ignore
from sklearn.preprocessing import label_binarize  # type: ignore

from src.utils.constants import (
    NSL_KDD_CLASS_NAMES,
    CLASS_COLORS,
    FIGURE_DPI,
    FIGURE_SIZE,
    FONT_SIZE,
    TITLE_FONT_SIZE,
    FIG_ROC_CURVE,
)
from src.utils.helpers import save_json
from src.utils.logger import get_logger
from src.utils.paths import FIGURES_DIR, METRICS_DIR, ensure_dir

logger = get_logger(__name__)
warnings.filterwarnings("ignore")


def compute_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
    dataset: str = "nsl_kdd",
) -> Dict:
    """
    Compute per-class and macro-average ROC curves.

    Parameters
    ----------
    y_true : np.ndarray
        1-D integer ground-truth labels.
    y_prob : np.ndarray
        2-D probability matrix (n_samples, n_classes).
    class_names : list of str, optional
    dataset : str

    Returns
    -------
    dict
        Per-class and macro-average fpr, tpr, auc values.
    """
    names = class_names or (
        NSL_KDD_CLASS_NAMES if dataset == "nsl_kdd" else None
    )
    n_classes = y_prob.shape[1]
    labels    = list(range(n_classes))

    # Binarise labels for OvR
    y_bin = label_binarize(y_true, classes=labels)

    roc_data: Dict = {"per_class": {}, "macro": {}}

    all_fpr: List[np.ndarray] = []
    all_tpr: List[np.ndarray] = []

    for i in labels:
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = float(auc(fpr, tpr))
        name = (
            names[i] if names and i < len(names) else str(i)
        )
        roc_data["per_class"][name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": round(roc_auc, 4),
        }
        all_fpr.append(fpr)
        all_tpr.append(tpr)

    # Macro-average
    if all_fpr:
        mean_fpr = np.linspace(0, 1, 200)
        mean_tpr = np.mean(
            [np.interp(mean_fpr, fpr, tpr)
             for fpr, tpr in zip(all_fpr, all_tpr)],
            axis=0,
        )
        macro_auc = float(auc(mean_fpr, mean_tpr))
        roc_data["macro"] = {
            "fpr": mean_fpr.tolist(),
            "tpr": mean_tpr.tolist(),
            "auc": round(macro_auc, 4),
        }

    return roc_data


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
    dataset: str = "nsl_kdd",
    model_name: str = "LSTM",
    output_path: Optional[Path] = None,
) -> Path:
    """
    Plot per-class and macro-average ROC curves.

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
    roc_data = compute_roc_curves(y_true, y_prob, class_names, dataset)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Per-class curves
    colors = CLASS_COLORS if len(CLASS_COLORS) >= len(
        roc_data["per_class"]
    ) else plt.cm.tab10.colors  # type: ignore

    for i, (name, data) in enumerate(
        roc_data["per_class"].items()
    ):
        ax.plot(
            data["fpr"], data["tpr"],
            color=colors[i % len(colors)],
            lw=1.8,
            label=f"{name} (AUC = {data['auc']:.4f})",
        )

    # Macro-average
    if "fpr" in roc_data.get("macro", {}):
        m = roc_data["macro"]
        ax.plot(
            m["fpr"], m["tpr"],
            color="black", lw=2.5, linestyle="--",
            label=f"Macro Average (AUC = {m['auc']:.4f})",
        )

    # Diagonal reference
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4,
            label="Random Classifier (AUC = 0.5)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=FONT_SIZE)
    ax.set_ylabel("True Positive Rate",  fontsize=FONT_SIZE)
    ax.set_title(
        f"ROC Curves — {model_name} on "
        f"{dataset.upper().replace('_', '-')}",
        fontsize=TITLE_FONT_SIZE, fontweight="bold", pad=12,
    )
    ax.legend(loc="lower right", fontsize=FONT_SIZE - 2)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_path = output_path or (FIGURES_DIR / FIG_ROC_CURVE)
    ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curve saved: %s", out_path)
    return out_path


def save_roc_scores(
    roc_data: Dict,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Save ROC AUC scores to JSON.

    Parameters
    ----------
    roc_data : dict
    output_path : Path, optional

    Returns
    -------
    Path
    """
    # Extract just the AUC values (not fpr/tpr arrays)
    auc_scores = {
        name: data["auc"]
        for name, data in roc_data.get("per_class", {}).items()
    }
    if "auc" in roc_data.get("macro", {}):
        auc_scores["macro_average"] = roc_data["macro"]["auc"]

    out = output_path or (METRICS_DIR / "roc_auc_scores.json")
    save_json(auc_scores, out)
    logger.info("ROC AUC scores saved: %s", out)
    return out