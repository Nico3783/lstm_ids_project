
# src/data/feature_engineering.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Feature engineering and selection operations
#          applied after basic preprocessing and before
#          sequence construction.
#
#          For the primary NSL-KDD dataset the preprocessed
#          41-feature set is used directly, so this module
#          focuses on:
#            - Removing zero-variance (constant) features
#            - Removing highly correlated redundant features
#            - Computing permutation-based feature importance
#              (post-training, used for Chapter 4 analysis)
#            - Generating the feature importance bar chart
#
#          Aligned with Chapter 3, Section 3.5.2 —
#          Preprocessing Pipeline and Chapter 3, Section 3.7
#          — Data Analysis Techniques.

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.constants import (
    FIGURE_DPI,
    FIGURE_SIZE,
    FIGURE_SIZE_WIDE,
    FONT_SIZE,
    TITLE_FONT_SIZE,
    COLOR_PALETTE,
    PLOT_STYLE,
    FIG_FEATURE_IMPORTANCE,
)
from src.utils.logger import get_logger
from src.utils.paths import FIGURES_DIR, ensure_dir

logger = get_logger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)
try:
    plt.style.use(PLOT_STYLE)
except OSError:
    plt.style.use("seaborn-v0_8-whitegrid")


# Zero-Variance Feature Removal

def remove_zero_variance_features(
    X: np.ndarray,
    feature_names: List[str],
    threshold: float = 0.0,
) -> Tuple[np.ndarray, List[str]]:
    """
    Remove features whose variance is at or below *threshold*.

    Constant features (variance = 0) carry no information
    and can cause numerical instability during training.
    The MinMaxScaler maps constant columns to all-zeros,
    making them harmless but wasteful.

    Parameters
    ----------
    X : np.ndarray
        2-D feature array (n_samples, n_features).
    feature_names : list of str
        Feature names corresponding to X columns.
    threshold : float
        Variance threshold.  Features at or below this value
        are removed.  Default 0.0 removes only constants.

    Returns
    -------
    tuple of (X_reduced, kept_feature_names)
    """
    if X.ndim != 2:
        raise ValueError(
            f"X must be 2-D (samples, features), got {X.ndim}-D."
        )

    variances = np.var(X, axis=0)
    keep_mask = variances > threshold
    removed = [
        n for n, keep in zip(feature_names, keep_mask) if not keep
    ]
    kept = [
        n for n, keep in zip(feature_names, keep_mask) if keep
    ]

    if removed:
        logger.info(
            "Zero-variance feature removal: dropped %d features "
            "(%d remaining): %s",
            len(removed), len(kept), removed[:20],
        )
        X_reduced = X[:, keep_mask]
    else:
        logger.info(
            "No zero-variance features found — all %d features kept.",
            len(feature_names),
        )
        X_reduced = X

    return X_reduced, kept


# High-Correlation Feature Removal

def remove_highly_correlated_features(
    X: np.ndarray,
    feature_names: List[str],
    threshold: float = 0.98,
) -> Tuple[np.ndarray, List[str]]:
    """
    Remove features that are highly correlated with another
    feature (Pearson |r| ≥ *threshold*).

    When two features are almost perfectly correlated, one
    is redundant.  Removing it reduces feature space
    dimensionality without information loss.

    Parameters
    ----------
    X : np.ndarray
        2-D feature array (n_samples, n_features).
    feature_names : list of str
    threshold : float
        Correlation threshold.  Default 0.98 removes only
        near-perfectly correlated pairs.

    Returns
    -------
    tuple of (X_reduced, kept_feature_names)
    """
    if X.shape[1] != len(feature_names):
        raise ValueError(
            "X columns and feature_names length mismatch: "
            f"{X.shape[1]} vs {len(feature_names)}."
        )

    df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = df.corr(method="pearson").abs()

    # Upper triangle mask — avoid double-counting pairs
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    )

    to_drop = [
        col for col in upper.columns
        if any(upper[col] >= threshold)
    ]

    if to_drop:
        logger.info(
            "High-correlation removal (threshold=%.2f): "
            "dropping %d features: %s",
            threshold, len(to_drop), to_drop[:20],
        )
        kept = [f for f in feature_names if f not in to_drop]
        X_reduced = df[kept].values
    else:
        logger.info(
            "No highly correlated features found at threshold %.2f. "
            "All %d features kept.",
            threshold, len(feature_names),
        )
        kept = feature_names
        X_reduced = X

    return X_reduced, kept


# Permutation Feature Importance

def compute_permutation_importance(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 10,
    random_state: int = 42,
    scoring: str = "accuracy",
) -> pd.DataFrame:
    """
    Compute permutation feature importance for a trained model.

    Permutation importance shuffles each feature column
    independently *n_repeats* times and records the drop in
    model performance.  Features that cause large performance
    drops when shuffled are more important.

    This analysis is described in Chapter 3, Section 3.7 —
    Data Analysis Techniques (Predictive analysis subsection)
    and generates Chapter 4 feature importance figures.

    Parameters
    ----------
    model : fitted model
        Any fitted model with a ``predict`` method compatible
        with scikit-learn (Keras models are wrapped below).
    X : np.ndarray
        2-D or 3-D test feature array.
        3-D arrays (from LSTM) are flattened to 2-D for
        permutation testing.
    y : np.ndarray
        1-D integer label array.
    feature_names : list of str
        Feature names corresponding to the last axis of X.
    n_repeats : int
        Number of permutation repeats per feature.
    random_state : int
    scoring : str
        Metric to use — ``"accuracy"`` (default).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ``feature``, ``importance_mean``, ``importance_std``,
        sorted by ``importance_mean`` descending.
    """
    from sklearn.inspection import permutation_importance  # type: ignore
    from sklearn.base import BaseEstimator  # type: ignore

    rng = np.random.default_rng(random_state)

    # Flatten 3-D LSTM input to 2-D for permutation
    if X.ndim == 3:
        n_samples, window, n_feat = X.shape
        X_flat = X[:, -1, :]   # Use the last timestep as representative
        logger.info(
            "LSTM 3-D input flattened to last-timestep 2-D "
            "for permutation importance: %s → %s.",
            X.shape, X_flat.shape,
        )
    else:
        X_flat = X
        n_feat = X.shape[1]

    if len(feature_names) != X_flat.shape[1]:
        logger.warning(
            "feature_names length (%d) does not match X columns (%d). "
            "Truncating/padding feature names.",
            len(feature_names), X_flat.shape[1],
        )
        if len(feature_names) > X_flat.shape[1]:
            feature_names = feature_names[: X_flat.shape[1]]
        else:
            feature_names = feature_names + [
                f"feature_{i}"
                for i in range(X_flat.shape[1] - len(feature_names))
            ]

    logger.info(
        "Computing permutation importance — %d features, "
        "%d repeats ...",
        X_flat.shape[1], n_repeats,
    )

    def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(y_true == y_pred))

    # Wrap Keras model for sklearn compatibility
    class _KerasWrapper:
        def __init__(self, keras_model: object) -> None:
            self._model = keras_model

        def predict(self, X_2d: np.ndarray) -> np.ndarray:
            # Reshape back to 3-D using the last timestep
            X_3d = X_2d[:, np.newaxis, :]  # (n, 1, features)
            probs = self._model.predict(X_3d, verbose=0)
            return np.argmax(probs, axis=1)

        def score(
            self, X_2d: np.ndarray, y_true: np.ndarray
        ) -> float:
            preds = self.predict(X_2d)
            return _accuracy(y_true, preds)

    # Detect Keras model
    is_keras = hasattr(model, "predict") and hasattr(model, "layers")
    wrapped_model = _KerasWrapper(model) if is_keras else model

    # Manual permutation importance (avoids sklearn version issues)
    baseline_score = wrapped_model.score(X_flat, y)
    logger.info("Baseline score: %.4f", baseline_score)

    importances: List[List[float]] = []
    for i in range(X_flat.shape[1]):
        col_scores: List[float] = []
        for _ in range(n_repeats):
            X_permuted = X_flat.copy()
            perm_idx = rng.permutation(len(X_permuted))
            X_permuted[:, i] = X_permuted[perm_idx, i]
            permuted_score = wrapped_model.score(X_permuted, y)
            col_scores.append(baseline_score - permuted_score)
        importances.append(col_scores)

    importance_means = [np.mean(scores) for scores in importances]
    importance_stds  = [np.std(scores)  for scores in importances]

    result_df = pd.DataFrame({
        "feature":          feature_names,
        "importance_mean":  importance_means,
        "importance_std":   importance_stds,
    })
    result_df = result_df.sort_values(
        "importance_mean", ascending=False
    ).reset_index(drop=True)

    logger.info(
        "Permutation importance computed. "
        "Top 5 features: %s",
        result_df["feature"].head(5).tolist(),
    )
    return result_df


def plot_feature_importance(
    importance_df: pd.DataFrame,
    dataset: str = "nsl_kdd",
    top_n: int = 20,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Generate a horizontal bar chart of the top-N most
    important features with error bars.

    This figure is saved to ``reports/figures/`` and is
    used directly in Chapter 4.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Output of ``compute_permutation_importance()``.
    dataset : str
    top_n : int
        Number of features to show.
    output_path : Path, optional

    Returns
    -------
    Path
        Path to the saved figure.
    """
    top_df = importance_df.head(top_n)
    palette = sns.color_palette(COLOR_PALETTE, len(top_df))

    fig, ax = plt.subplots(
        figsize=(FIGURE_SIZE[0], max(6, len(top_df) * 0.45))
    )

    colors = [str(c) for c in palette]
    ax.barh(
        y=range(len(top_df)),
        width=top_df["importance_mean"].values,
        xerr=top_df["importance_std"].values,
        color=colors,
        edgecolor="white",
        capsize=3,
        alpha=0.9,
    )

    ax.set_yticks(range(len(top_df)))
    ax.set_yticklabels(top_df["feature"].tolist(), fontsize=FONT_SIZE - 1)
    ax.invert_yaxis()
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")

    ax.set_title(
        f"Top {top_n} Features by Permutation Importance — "
        f"{dataset.upper().replace('_', '-')}",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel(
        "Mean Accuracy Decrease on Permutation",
        fontsize=FONT_SIZE,
    )
    ax.set_ylabel("Feature", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=FONT_SIZE - 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_path = output_path or (FIGURES_DIR / FIG_FEATURE_IMPORTANCE)
    ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature importance plot saved: %s", out_path)
    return out_path


# Feature Engineering Summary

def get_feature_engineering_summary(
    original_features: List[str],
    final_features: List[str],
    dataset: str = "nsl_kdd",
) -> Dict:
    """
    Return a summary of the feature engineering steps applied,
    for logging and Chapter 3 documentation.

    Parameters
    ----------
    original_features : list of str
    final_features : list of str
    dataset : str

    Returns
    -------
    dict
    """
    removed = set(original_features) - set(final_features)
    added = set(final_features) - set(original_features)
    summary = {
        "dataset": dataset,
        "n_features_before": len(original_features),
        "n_features_after": len(final_features),
        "n_features_removed": len(removed),
        "n_features_added": len(added),
        "removed_features": sorted(list(removed)),
        "added_features": sorted(list(added)),
    }
    logger.info(
        "Feature engineering summary — %s: %d → %d features "
        "(%d removed, %d added).",
        dataset,
        summary["n_features_before"],
        summary["n_features_after"],
        summary["n_features_removed"],
        summary["n_features_added"],
    )
    return summary