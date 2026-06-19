
# src/data/exploratory.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Exploratory Data Analysis (EDA) module.
#          Generates all statistical summaries and
#          publication-quality visualisations required for
#          Chapter 3 and Chapter 4 of the project report:
#            - Dataset shape and dtype summaries
#            - Class distribution bar charts
#            - Missing value heatmaps / bar charts
#            - Feature correlation heatmaps
#            - Feature distribution histograms
#            - Dataset summary CSV tables
#
#          All figures are saved to reports/figures/ at
#          300 DPI (FIGURE_DPI constant) so they can be
#          inserted directly into the written report.
#
#          Aligned with Chapter 3, Section 3.5.1 —
#          Data Acquisition and Exploratory Analysis.

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.constants import (
    NSL_KDD_TARGET_COLUMN,
    NSL_KDD_CLASS_NAMES,
    NSL_KDD_CATEGORY_TO_INT,
    NSL_KDD_NUMERICAL_FEATURES,
    CICIDS2017_TARGET_COLUMN,
    UNSW_NB15_BINARY_LABEL_COLUMN,
    UNSW_NB15_TARGET_COLUMN,
    FIGURE_DPI,
    FIGURE_SIZE,
    FIGURE_SIZE_SQUARE,
    FIGURE_SIZE_WIDE,
    PLOT_STYLE,
    COLOR_PALETTE,
    FONT_SIZE,
    TITLE_FONT_SIZE,
    CLASS_COLORS,
    SUPPORTED_DATASETS,
    FIG_CLASS_DISTRIBUTION,
    FIG_CORRELATION_HEATMAP,
    TABLE_DATASET_SUMMARY,
)
from src.utils.logger import get_logger
from src.utils.paths import (
    FIGURES_DIR,
    TABLES_DIR,
    ensure_dir,
)

logger = get_logger(__name__)

# Apply consistent matplotlib style throughout
warnings.filterwarnings("ignore", category=UserWarning)
try:
    plt.style.use(PLOT_STYLE)
except OSError:
    plt.style.use("seaborn-v0_8-whitegrid")


# Internal Helpers

def _save_figure(
    fig: plt.Figure,
    path: Path,
    tight: bool = True,
) -> None:
    """
    Save *fig* to *path* at publication-quality DPI.

    Parameters
    ----------
    fig : plt.Figure
    path : Path
    tight : bool
        Apply tight_layout before saving.
    """
    ensure_dir(path)
    if tight:
        fig.tight_layout()
    fig.savefig(str(path), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved: %s", path)


def _setup_axes(
    ax: plt.Axes,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
) -> None:
    """Apply consistent axis styling."""
    ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=FONT_SIZE - 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# Dataset Shape & Dtype Summary

def describe_dataset(
    df: pd.DataFrame,
    dataset: str = "nsl_kdd",
    split: str = "full",
) -> Dict:
    """
    Print and return a comprehensive textual summary of a
    loaded DataFrame — shape, dtypes, memory usage, and
    basic statistics.

    Parameters
    ----------
    df : pd.DataFrame
    dataset : str
    split : str

    Returns
    -------
    dict
        Summary statistics dictionary.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    summary = {
        "dataset": dataset,
        "split": split,
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "n_numeric_features": len(numeric_cols),
        "n_categorical_features": len(categorical_cols),
        "memory_usage_mb": round(
            df.memory_usage(deep=True).sum() / (1024 ** 2), 2
        ),
        "total_missing_cells": int(df.isnull().sum().sum()),
        "missing_pct": round(
            df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 4
        ),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_pct": round(df.duplicated().sum() / len(df) * 100, 4),
    }

    logger.info("=" * 60)
    logger.info("DATASET DESCRIPTION — %s (%s)", dataset.upper(), split)
    logger.info("=" * 60)
    logger.info("  Rows            : %d", summary["n_rows"])
    logger.info("  Columns         : %d", summary["n_columns"])
    logger.info("  Numeric features: %d", summary["n_numeric_features"])
    logger.info("  Categorical feat: %d", summary["n_categorical_features"])
    logger.info("  Memory usage    : %.2f MB", summary["memory_usage_mb"])
    logger.info(
        "  Missing cells   : %d (%.4f%%)",
        summary["total_missing_cells"], summary["missing_pct"],
    )
    logger.info(
        "  Duplicate rows  : %d (%.4f%%)",
        summary["n_duplicate_rows"], summary["duplicate_pct"],
    )

    return summary


# Class Distribution

def plot_class_distribution(
    df: pd.DataFrame,
    dataset: str = "nsl_kdd",
    output_path: Optional[Path] = None,
    show_percentages: bool = True,
) -> Path:
    """
    Generate a bar chart of the class label distribution.

    For NSL-KDD the raw attack-type labels are grouped into
    the 5 macro-categories (Normal, DoS, Probe, R2L, U2R)
    defined in Chapter 3.

    The figure is saved to ``reports/figures/`` and its path
    is returned so the pipeline can log it.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded dataset DataFrame with a target column.
    dataset : str
        Dataset identifier — determines which target column
        and label names to use.
    output_path : Path, optional
        Override the default output path.
    show_percentages : bool
        Annotate each bar with its percentage of total.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    target_col, class_names = _get_target_info(dataset, df)

    if target_col not in df.columns:
        logger.warning(
            "Target column '%s' not found — skipping class "
            "distribution plot.", target_col,
        )
        return FIGURES_DIR / FIG_CLASS_DISTRIBUTION

    # For NSL-KDD: map raw attack types to 5-class categories
    if dataset == "nsl_kdd":
        from src.utils.constants import NSL_KDD_ATTACK_TO_CATEGORY
        labels = (
            df[target_col]
            .str.lower()
            .str.strip()
            .map(NSL_KDD_ATTACK_TO_CATEGORY)
            .fillna("unknown")
        )
        ordered_cats = ["normal", "dos", "probe", "r2l", "u2r"]
        dist = labels.value_counts().reindex(ordered_cats, fill_value=0)
        display_names = NSL_KDD_CLASS_NAMES
        colors = CLASS_COLORS
    else:
        dist = df[target_col].value_counts()
        display_names = dist.index.tolist()
        palette = sns.color_palette(COLOR_PALETTE, len(dist))
        colors = list(palette)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    bars = ax.bar(
        x=range(len(dist)),
        height=dist.values,
        color=colors[: len(dist)],
        edgecolor="white",
        linewidth=0.8,
        alpha=0.9,
    )

    ax.set_xticks(range(len(dist)))
    ax.set_xticklabels(display_names, fontsize=FONT_SIZE, rotation=15)

    # Annotate bars
    total = dist.sum()
    for bar, count in zip(bars, dist.values):
        pct = count / total * 100 if total > 0 else 0
        label_text = (
            f"{count:,}\n({pct:.1f}%)"
            if show_percentages
            else f"{count:,}"
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            label_text,
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE - 1,
        )

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

    _setup_axes(
        ax,
        title=f"Class Distribution — {dataset.upper().replace('_', '-')}",
        xlabel="Traffic Class",
        ylabel="Number of Records",
    )

    out_path = output_path or (FIGURES_DIR / FIG_CLASS_DISTRIBUTION)
    _save_figure(fig, out_path)
    return out_path


# Missing Values

def plot_missing_values(
    df: pd.DataFrame,
    dataset: str = "nsl_kdd",
    output_path: Optional[Path] = None,
    top_n: int = 30,
) -> Optional[Path]:
    """
    Generate a horizontal bar chart showing the percentage of
    missing values per column (top-N columns by missingness).

    If no missing values exist, logs an info message and
    returns None without writing a file.

    Parameters
    ----------
    df : pd.DataFrame
    dataset : str
    output_path : Path, optional
    top_n : int
        Maximum number of columns to show in the chart.

    Returns
    -------
    Path or None
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0].head(top_n)

    if missing_pct.empty:
        logger.info(
            "No missing values in %s — skipping missing value plot.",
            dataset,
        )
        return None

    fig, ax = plt.subplots(
        figsize=(FIGURE_SIZE[0], max(6, len(missing_pct) * 0.4))
    )

    colors = sns.color_palette("Reds_r", len(missing_pct))
    ax.barh(
        y=range(len(missing_pct)),
        width=missing_pct.values,
        color=colors,
        edgecolor="white",
    )
    ax.set_yticks(range(len(missing_pct)))
    ax.set_yticklabels(missing_pct.index.tolist(), fontsize=FONT_SIZE - 1)
    ax.invert_yaxis()

    for i, (col, pct) in enumerate(missing_pct.items()):
        ax.text(
            pct + 0.2, i, f"{pct:.2f}%",
            va="center", fontsize=FONT_SIZE - 2,
        )

    _setup_axes(
        ax,
        title=f"Missing Values by Feature — "
              f"{dataset.upper().replace('_', '-')}",
        xlabel="Missing (%)",
        ylabel="Feature",
    )

    fname = f"missing_values_{dataset}.png"
    out_path = output_path or (FIGURES_DIR / fname)
    _save_figure(fig, out_path)
    return out_path


# Correlation Heatmap

def plot_correlation_heatmap(
    df: pd.DataFrame,
    dataset: str = "nsl_kdd",
    output_path: Optional[Path] = None,
    top_n_features: int = 30,
    method: str = "pearson",
) -> Path:
    """
    Generate a correlation heatmap for the numeric features.

    For large feature sets (> top_n_features), only the
    top-N features by mean absolute correlation with all
    other features are shown to keep the figure readable.

    Parameters
    ----------
    df : pd.DataFrame
    dataset : str
    output_path : Path, optional
    top_n_features : int
        Maximum number of features to include.
    method : str
        Correlation method — ``"pearson"``, ``"spearman"``,
        or ``"kendall"``.

    Returns
    -------
    Path
    """
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    # Remove target / difficulty / split indicator columns
    cols_to_drop = [
        c for c in numeric_df.columns
        if c in [
            NSL_KDD_TARGET_COLUMN, "difficulty",
            "_split", UNSW_NB15_BINARY_LABEL_COLUMN,
        ]
    ]
    numeric_df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    if numeric_df.shape[1] > top_n_features:
        corr_full = numeric_df.corr(method=method, numeric_only=True)
        mean_abs_corr = corr_full.abs().mean().sort_values(ascending=False)
        top_cols = mean_abs_corr.head(top_n_features).index.tolist()
        numeric_df = numeric_df[top_cols]
        logger.info(
            "Correlation heatmap: showing top %d / %d features.",
            top_n_features, corr_full.shape[0],
        )

    corr_matrix = numeric_df.corr(method=method, numeric_only=True)

    n = corr_matrix.shape[0]
    fig_size = (max(12, n * 0.45), max(10, n * 0.4))
    fig, ax = plt.subplots(figsize=fig_size)

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        ax=ax,
        annot=(n <= 20),
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.3,
        cbar_kws={"shrink": 0.8, "label": f"{method.capitalize()} r"},
        annot_kws={"size": max(6, FONT_SIZE - 4)},
    )

    ax.set_title(
        f"Feature Correlation Heatmap — "
        f"{dataset.upper().replace('_', '-')} "
        f"(top {n} features, {method})",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        pad=14,
    )
    ax.tick_params(axis="both", labelsize=max(7, FONT_SIZE - 3))

    out_path = output_path or (FIGURES_DIR / FIG_CORRELATION_HEATMAP)
    _save_figure(fig, out_path, tight=False)
    return out_path


# Feature Distribution Histograms

def plot_feature_distributions(
    df: pd.DataFrame,
    dataset: str = "nsl_kdd",
    features: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    n_cols: int = 4,
    top_n: int = 16,
) -> Path:
    """
    Generate a grid of histograms for numeric features.

    Parameters
    ----------
    df : pd.DataFrame
    dataset : str
    features : list of str, optional
        Specific features to plot.  Defaults to the top-N
        numeric features by variance.
    output_path : Path, optional
    n_cols : int
        Number of columns in the subplot grid.
    top_n : int
        Maximum number of features to plot.

    Returns
    -------
    Path
    """
    numeric_df = df.select_dtypes(include=[np.number])
    cols_to_drop = [
        c for c in numeric_df.columns
        if c in ["difficulty", "_split", UNSW_NB15_BINARY_LABEL_COLUMN]
    ]
    numeric_df = numeric_df.drop(columns=cols_to_drop, errors="ignore")

    if features is None:
        variances = numeric_df.var().sort_values(ascending=False)
        features = variances.head(top_n).index.tolist()
    else:
        features = [f for f in features if f in numeric_df.columns][:top_n]

    if not features:
        logger.warning("No numeric features to plot distributions for.")
        fname = f"feature_distributions_{dataset}.png"
        return FIGURES_DIR / fname

    n_rows = (len(features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.5, n_rows * 2.8),
    )
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    palette = sns.color_palette(COLOR_PALETTE, len(features))

    for i, feat in enumerate(features):
        ax = axes_flat[i]
        data = numeric_df[feat].dropna()
        ax.hist(
            data,
            bins=40,
            color=palette[i],
            edgecolor="white",
            alpha=0.85,
        )
        ax.set_title(feat, fontsize=FONT_SIZE - 1, fontweight="bold")
        ax.set_xlabel("Value", fontsize=FONT_SIZE - 2)
        ax.set_ylabel("Count", fontsize=FONT_SIZE - 2)
        ax.tick_params(labelsize=FONT_SIZE - 3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused subplots
    for j in range(len(features), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Feature Distributions — {dataset.upper().replace('_', '-')}",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        y=1.01,
    )

    fname = f"feature_distributions_{dataset}.png"
    out_path = output_path or (FIGURES_DIR / fname)
    _save_figure(fig, out_path)
    return out_path


# Dataset Summary Table

def generate_dataset_summary_table(
    summaries: Dict[str, Dict],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Save a multi-dataset comparison summary as a CSV table
    for direct inclusion in Chapter 4 tables.

    Parameters
    ----------
    summaries : dict
        ``{dataset_name: summary_dict}`` from
        ``get_dataset_summary()``.
    output_path : Path, optional

    Returns
    -------
    Path
        Path to the saved CSV.
    """
    rows = []
    for name, s in summaries.items():
        rows.append({
            "Dataset": s.get("dataset", name),
            "Total Samples": s.get("n_samples", "—"),
            "Raw Features": s.get("n_features_raw", "—"),
            "Classes": s.get("n_classes", s.get("n_classes_raw", "—")),
            "Missing Values (%)": s.get("missing_value_pct", "—"),
        })

    summary_df = pd.DataFrame(rows)
    out_path = output_path or (TABLES_DIR / TABLE_DATASET_SUMMARY)
    ensure_dir(out_path)
    summary_df.to_csv(out_path, index=False)
    logger.info("Dataset summary table saved: %s", out_path)
    return out_path


# Full EDA Pipeline

def run_eda(
    df: pd.DataFrame,
    dataset: str = "nsl_kdd",
    output_dir: Optional[Path] = None,
    top_n_features: int = 30,
) -> Dict[str, Path]:
    """
    Run the full EDA pipeline for a loaded dataset and
    save all output figures and tables.

    Generates:
    - Class distribution bar chart
    - Missing values chart (if applicable)
    - Feature correlation heatmap
    - Feature distribution histograms
    - Dataset summary statistics log

    Parameters
    ----------
    df : pd.DataFrame
        Loaded dataset DataFrame.
    dataset : str
        Dataset identifier.
    output_dir : Path, optional
        Override output directory for all figures.
    top_n_features : int
        Max features for correlation heatmap.

    Returns
    -------
    dict
        ``{plot_name: saved_path}`` for all generated figures.
    """
    figures_dir = output_dir or FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("EXPLORATORY DATA ANALYSIS — %s", dataset.upper())
    logger.info("=" * 60)

    saved: Dict[str, Path] = {}

    # 1. Dataset description
    describe_dataset(df, dataset=dataset)

    # 2. Class distribution
    logger.info("Generating class distribution plot ...")
    p = plot_class_distribution(
        df,
        dataset=dataset,
        output_path=figures_dir / FIG_CLASS_DISTRIBUTION,
    )
    saved["class_distribution"] = p

    # 3. Missing values
    logger.info("Generating missing values plot ...")
    p = plot_missing_values(
        df,
        dataset=dataset,
        output_path=figures_dir / f"missing_values_{dataset}.png",
    )
    if p:
        saved["missing_values"] = p

    # 4. Correlation heatmap
    logger.info("Generating correlation heatmap ...")
    p = plot_correlation_heatmap(
        df,
        dataset=dataset,
        output_path=figures_dir / FIG_CORRELATION_HEATMAP,
        top_n_features=top_n_features,
    )
    saved["correlation_heatmap"] = p

    # 5. Feature distributions
    logger.info("Generating feature distribution plots ...")
    p = plot_feature_distributions(
        df,
        dataset=dataset,
        output_path=figures_dir / f"feature_distributions_{dataset}.png",
    )
    saved["feature_distributions"] = p

    logger.info("EDA complete. %d figures saved.", len(saved))
    for name, path in saved.items():
        logger.info("  %-30s → %s", name, path)

    return saved


# Internal Helpers

def _get_target_info(
    dataset: str,
    df: pd.DataFrame,
) -> Tuple[str, List[str]]:
    """
    Return the target column name and class name list for the
    given dataset.

    Parameters
    ----------
    dataset : str
    df : pd.DataFrame

    Returns
    -------
    tuple of (target_column_str, class_names_list)
    """
    if dataset == "nsl_kdd":
        return NSL_KDD_TARGET_COLUMN, NSL_KDD_CLASS_NAMES
    elif dataset == "cicids2017":
        col = CICIDS2017_TARGET_COLUMN.strip()
        names = (
            df[col].unique().tolist()
            if col in df.columns
            else []
        )
        return col, names
    elif dataset == "unsw_nb15":
        names = (
            df[UNSW_NB15_TARGET_COLUMN].unique().tolist()
            if UNSW_NB15_TARGET_COLUMN in df.columns
            else []
        )
        return UNSW_NB15_TARGET_COLUMN, names
    else:
        return "label", []