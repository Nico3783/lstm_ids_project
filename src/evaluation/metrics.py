
# src/evaluation/metrics.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Compute all evaluation metrics described in
#          Chapter 3, Section 3.5.5 — Model Evaluation.
#
#          Metrics computed:
#            - Accuracy
#            - Precision (macro + weighted)
#            - Recall    (macro + weighted)
#            - F1-Score  (macro + weighted)
#            - Confusion Matrix
#            - ROC-AUC   (macro OvR)
#            - Per-class metrics
#
#          All results are returned as structured dicts
#          ready for JSON serialisation and CSV export.

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)

from src.utils.constants import NSL_KDD_CLASS_NAMES, SUPPORTED_DATASETS
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Core Metrics

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    dataset: str = "nsl_kdd",
    model_name: str = "model",
) -> Dict:
    """
    Compute the full suite of classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        1-D ground truth integer labels.
    y_pred : np.ndarray
        1-D predicted integer labels.
    y_prob : np.ndarray, optional
        2-D probability matrix (n_samples, n_classes).
        Required for ROC-AUC computation.
    class_names : list of str, optional
        Human-readable class names.
    dataset : str
    model_name : str

    Returns
    -------
    dict
        Complete metrics dictionary with scalar values and
        per-class breakdowns.
    """
    names = class_names or (
        NSL_KDD_CLASS_NAMES if dataset == "nsl_kdd" else None
    )
    labels = sorted(np.unique(
        np.concatenate([y_true, y_pred])
    ).tolist())

    # ---- Scalar metrics ----
    accuracy = float(accuracy_score(y_true, y_pred))

    precision_macro    = float(precision_score(
        y_true, y_pred, average="macro",    zero_division=0
    ))
    precision_weighted = float(precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    ))
    recall_macro       = float(recall_score(
        y_true, y_pred, average="macro",    zero_division=0
    ))
    recall_weighted    = float(recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    ))
    f1_macro           = float(f1_score(
        y_true, y_pred, average="macro",    zero_division=0
    ))
    f1_weighted        = float(f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    ))

    # ---- ROC-AUC ----
    roc_auc: Optional[float] = None
    if y_prob is not None:
        try:
            n_classes = y_prob.shape[1]
            if n_classes == 2:
                roc_auc = float(roc_auc_score(
                    y_true, y_prob[:, 1]
                ))
            else:
                roc_auc = float(roc_auc_score(
                    y_true, y_prob,
                    multi_class="ovr",
                    average="macro",
                    labels=labels,
                ))
        except Exception as exc:  # noqa: BLE001
            logger.warning("ROC-AUC computation failed: %s", exc)

    # ---- Confusion matrix ----
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # ---- Per-class metrics ----
    per_class = _per_class_metrics(y_true, y_pred, labels, names)

    # ---- Assemble result ----
    metrics = {
        "model_name":          model_name,
        "dataset":             dataset,
        "n_samples":           int(len(y_true)),
        "accuracy":            round(accuracy,            4),
        "precision_macro":     round(precision_macro,     4),
        "precision_weighted":  round(precision_weighted,  4),
        "recall_macro":        round(recall_macro,        4),
        "recall_weighted":     round(recall_weighted,     4),
        "f1_macro":            round(f1_macro,            4),
        "f1_weighted":         round(f1_weighted,         4),
        "roc_auc":             round(roc_auc, 4) if roc_auc is not None
                               else None,
        "confusion_matrix":    cm.tolist(),
        "per_class_metrics":   per_class,
        "class_names":         names or [str(l) for l in labels],
        "labels":              labels,
    }

    _log_metrics(metrics)
    return metrics


def _per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[int],
    class_names: Optional[List[str]],
) -> Dict[str, Dict[str, float]]:
    """Compute per-class precision, recall, F1, support."""
    p = precision_score(
        y_true, y_pred, labels=labels,
        average=None, zero_division=0,
    )
    r = recall_score(
        y_true, y_pred, labels=labels,
        average=None, zero_division=0,
    )
    f = f1_score(
        y_true, y_pred, labels=labels,
        average=None, zero_division=0,
    )
    support = np.bincount(
        np.asarray(y_true, dtype=int),
        minlength=max(labels) + 1,
    )

    result: Dict[str, Dict[str, float]] = {}
    for i, lbl in enumerate(labels):
        name = (
            class_names[lbl]
            if class_names and lbl < len(class_names)
            else str(lbl)
        )
        result[name] = {
            "precision": round(float(p[i]), 4),
            "recall":    round(float(r[i]), 4),
            "f1_score":  round(float(f[i]), 4),
            "support":   int(support[lbl]) if lbl < len(support) else 0,
        }
    return result


def _log_metrics(metrics: Dict) -> None:
    """Log a clean summary of the computed metrics."""
    logger.info("-" * 55)
    logger.info(
        "Evaluation Results — %s on %s",
        metrics["model_name"], metrics["dataset"].upper(),
    )
    logger.info("-" * 55)
    logger.info("  Accuracy          : %.4f", metrics["accuracy"])
    logger.info(
        "  Precision (macro) : %.4f", metrics["precision_macro"]
    )
    logger.info(
        "  Recall    (macro) : %.4f", metrics["recall_macro"]
    )
    logger.info(
        "  F1-Score  (macro) : %.4f", metrics["f1_macro"]
    )
    logger.info(
        "  F1-Score  (wtd)   : %.4f", metrics["f1_weighted"]
    )
    if metrics["roc_auc"] is not None:
        logger.info("  ROC-AUC           : %.4f", metrics["roc_auc"])
    logger.info("-" * 55)


# Prediction Helpers

def predict_lstm(
    model: object,
    X: np.ndarray,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate class predictions and probabilities from a
    trained Keras LSTM model.

    Parameters
    ----------
    model : tf.keras.Model
    X : np.ndarray
        3-D input array (samples, window, features).
    batch_size : int

    Returns
    -------
    tuple of (y_pred, y_prob)
        y_pred : 1-D integer class predictions.
        y_prob : 2-D probability matrix.
    """
    y_prob = model.predict(X, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(y_prob, axis=1).astype(np.int64)
    return y_pred, y_prob.astype(np.float32)


def predict_baseline_model(
    model: object,
    X: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Generate predictions from a scikit-learn baseline model.

    Parameters
    ----------
    model : sklearn estimator
    X : np.ndarray

    Returns
    -------
    tuple of (y_pred, y_prob or None)
    """
    from src.models.baseline_models import predict_baseline
    return predict_baseline(model, X)