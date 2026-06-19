
# src/evaluation/comparison.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Compare LSTM against all baseline models and
#          generate the model comparison bar chart and
#          summary table required for Chapter 4.

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.constants import (
    FIGURE_DPI, FIGURE_SIZE_WIDE, FIGURE_SIZE,
    FONT_SIZE, TITLE_FONT_SIZE, COLOR_PALETTE,
    FIG_MODEL_COMPARISON,
)
from src.utils.helpers import save_json
from src.utils.logger import get_logger
from src.utils.paths import FIGURES_DIR, TABLES_DIR, METRICS_DIR, ensure_dir

logger = get_logger(__name__)
warnings.filterwarnings("ignore")


def build_comparison_table(
    all_metrics: Dict[str, Dict],
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Build a comparison table from a dict of per-model metrics.

    Parameters
    ----------
    all_metrics : dict
        ``{model_name: metrics_dict}``
    output_path : Path, optional

    Returns
    -------
    pd.DataFrame
        Comparison table saved to CSV.
    """
    rows = []
    for model_name, m in all_metrics.items():
        rows.append({
            "Model":              model_name,
            "Accuracy":           m.get("accuracy",           "—"),
            "Precision (Macro)":  m.get("precision_macro",    "—"),
            "Recall (Macro)":     m.get("recall_macro",       "—"),
            "F1-Score (Macro)":   m.get("f1_macro",           "—"),
            "F1-Score (Wtd)":     m.get("f1_weighted",        "—"),
            "ROC-AUC":            m.get("roc_auc",            "—"),
        })

    df = pd.DataFrame(rows)

    out_path = output_path or (TABLES_DIR / "baseline_metrics.csv")
    ensure_dir(out_path)
    df.to_csv(out_path, index=False)
    logger.info("Comparison table saved: %s", out_path)
    return df


def plot_model_comparison(
    all_metrics: Dict[str, Dict],
    metrics_to_plot: Optional[List[str]] = None,
    dataset: str = "nsl_kdd",
    output_path: Optional[Path] = None,
) -> Path:
    """
    Generate a grouped bar chart comparing all models across
    key metrics for Chapter 4.

    Parameters
    ----------
    all_metrics : dict
        ``{model_name: metrics_dict}``
    metrics_to_plot : list of str, optional
        Metric keys to include.  Defaults to the four main
        metrics: accuracy, precision, recall, f1_macro.
    dataset : str
    output_path : Path, optional

    Returns
    -------
    Path
    """
    if metrics_to_plot is None:
        metrics_to_plot = [
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
        ]

    metric_labels = {
        "accuracy":           "Accuracy",
        "precision_macro":    "Precision\n(Macro)",
        "recall_macro":       "Recall\n(Macro)",
        "f1_macro":           "F1-Score\n(Macro)",
        "f1_weighted":        "F1-Score\n(Weighted)",
        "roc_auc":            "ROC-AUC",
    }

    model_names = list(all_metrics.keys())
    n_models    = len(model_names)
    n_metrics   = len(metrics_to_plot)

    values = np.zeros((n_models, n_metrics))
    for i, name in enumerate(model_names):
        m = all_metrics[name]
        for j, metric in enumerate(metrics_to_plot):
            v = m.get(metric, 0)
            values[i, j] = float(v) if v is not None else 0.0

    x = np.arange(n_metrics)
    bar_width = 0.8 / n_models
    palette   = sns.color_palette(COLOR_PALETTE, n_models)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for i, (name, color) in enumerate(zip(model_names, palette)):
        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            values[i],
            width=bar_width * 0.9,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=name,
            alpha=0.9,
        )
        # Value labels on bars
        for bar, val in zip(bars, values[i]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=FONT_SIZE - 3,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [metric_labels.get(m, m) for m in metrics_to_plot],
        fontsize=FONT_SIZE,
    )
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=FONT_SIZE)
    ax.set_title(
        f"Model Performance Comparison — "
        f"{dataset.upper().replace('_', '-')}",
        fontsize=TITLE_FONT_SIZE, fontweight="bold", pad=12,
    )
    ax.legend(
        loc="upper right", fontsize=FONT_SIZE - 2,
        framealpha=0.9,
    )
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_path = output_path or (FIGURES_DIR / FIG_MODEL_COMPARISON)
    ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Model comparison chart saved: %s", out_path)
    return out_path


def save_evaluation_results(
    all_metrics: Dict[str, Dict],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Save all model evaluation results to a JSON file.

    Parameters
    ----------
    all_metrics : dict
    output_path : Path, optional

    Returns
    -------
    Path
    """
    # Remove large arrays before saving JSON
    serialisable: Dict = {}
    for model, m in all_metrics.items():
        serialisable[model] = {
            k: v for k, v in m.items()
            if k not in ("confusion_matrix", "per_class_metrics")
        }

    out = output_path or (METRICS_DIR / "evaluation_results.json")
    save_json(serialisable, out)
    logger.info("Evaluation results saved: %s", out)
    return out