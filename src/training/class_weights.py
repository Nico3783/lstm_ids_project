
# src/training/class_weights.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Compute and format class weights for use in
#          model.fit(class_weight=...).
#
#          Chapter 3, Section 3.5.4:
#          "Class imbalance was addressed by computing
#          inverse-frequency class weights and incorporating
#          them into the training loss."
#
#          Formula:
#              weight_c = n_samples / (n_classes * count_c)
#
#          This encourages the model to attend to minority
#          attack classes (U2R, R2L) rather than optimising
#          primarily for the majority Normal and DoS classes.

from typing import Dict, List, Optional

import numpy as np

from src.utils.constants import NSL_KDD_CLASS_NAMES
from src.utils.helpers import compute_class_weights
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_class_weights(
    y_train: np.ndarray,
    class_names: Optional[List[str]] = None,
    strategy: str = "inverse_frequency",
) -> Dict[int, float]:
    """
    Compute class weights for training.

    Parameters
    ----------
    y_train : np.ndarray
        1-D integer training label array.
    class_names : list of str, optional
        Human-readable names for logging.
    strategy : str
        ``"inverse_frequency"`` — weight_c = n / (k × n_c)
        ``"uniform"``           — all weights = 1.0
        ``"sqrt"``              — weight_c = sqrt(n / n_c)

    Returns
    -------
    dict
        ``{class_int: weight_float}`` ready for Keras
        ``model.fit(class_weight=...)``.
    """
    unique, counts = np.unique(y_train, return_counts=True)
    n_samples = len(y_train)
    n_classes  = len(unique)

    if strategy == "uniform":
        weights = {int(c): 1.0 for c in unique}

    elif strategy == "sqrt":
        weights = {}
        for cls, cnt in zip(unique, counts):
            weights[int(cls)] = float(
                np.sqrt(n_samples / (n_classes * cnt))
            )

    else:   # inverse_frequency (default, matches Chapter 3)
        weights = compute_class_weights(y_train)

    # Log with class names
    names = class_names or (
        NSL_KDD_CLASS_NAMES
        if n_classes <= len(NSL_KDD_CLASS_NAMES)
        else None
    )
    logger.info(
        "Class weights (strategy='%s'):", strategy
    )
    for cls_int, wt in sorted(weights.items()):
        name = (
            names[cls_int]
            if names and cls_int < len(names)
            else str(cls_int)
        )
        count = int(counts[list(unique).index(cls_int)])
        logger.info(
            "  Class %d (%s): n=%d, weight=%.4f",
            cls_int, name, count, wt,
        )

    return weights


def weights_to_sample_weights(
    y: np.ndarray,
    class_weights: Dict[int, float],
) -> np.ndarray:
    """
    Expand class weights to a per-sample weight array.

    Keras ``model.fit`` accepts either ``class_weight`` (dict)
    or ``sample_weight`` (array).  This function converts the
    dict to an array — needed when using a data generator
    or custom training loop.

    Parameters
    ----------
    y : np.ndarray
        1-D integer label array.
    class_weights : dict

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(n_samples,)``.
    """
    sample_weights = np.ones(len(y), dtype=np.float32)
    for cls_int, wt in class_weights.items():
        sample_weights[y == cls_int] = wt
    return sample_weights