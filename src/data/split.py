
# src/data/split.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Stratified train / validation / test splitting of
#          the processed sequence arrays.
#
#          Split specification (Chapter 3, Section 3.5.2 —
#          Data Splitting):
#            Training   : 70 %
#            Validation : 15 %
#            Test       : 15 %
#            Method     : Stratified sampling — preserves
#                         class proportions across all three
#                         partitions.
#
#          The validation set is used exclusively for
#          hyperparameter tuning and early stopping decisions.
#          The test set is held out until final evaluation
#          (Chapter 3, Section 3.5.5).
#
#          This module also provides:
#            - Class weight computation for the training set
#            - Split summary statistics for Chapter 4 tables
#            - Saved split artifacts to data/processed/

import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.constants import (
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    NSL_KDD_CLASS_NAMES,
    SUPPORTED_DATASETS,
)
from src.utils.helpers import (
    compute_class_weights,
    compute_class_distribution,
    check_label_consistency,
    validate_array_shapes,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Chunk size for disk-based splitting (must be defined before
# split_and_save which references it as a default parameter).
SPLIT_CHUNK_SIZE = 100_000
from src.utils.paths import PROCESSED_DATA_DIR
from src.utils.serialization import save_processed_arrays

logger = get_logger(__name__)


# Primary Split Function
def split_sequences(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    stratified: bool = True,
    random_state: int = RANDOM_SEED,
    shuffle: bool = True,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
]:
    """
    Split sequence arrays into training, validation, and test
    partitions using stratified sampling.

    Stratification ensures that the class proportions across
    all three splits are as close as possible to the original
    distribution — critical for the minority attack classes
    (R2L, U2R in NSL-KDD) that would otherwise be absent
    from some splits by chance.

    Ratio validation
    ----------------
    The three ratios must sum to 1.0 (within floating-point
    tolerance).  Default: 0.70 / 0.15 / 0.15.

    Parameters
    ----------
    X : np.ndarray
        3-D sequence array of shape
        ``(n_sequences, window_size, n_features)``.
    y : np.ndarray
        1-D integer label array of shape ``(n_sequences,)``.
    train_ratio : float
        Proportion of data for training.  Default: 0.70.
    val_ratio : float
        Proportion of data for validation.  Default: 0.15.
    test_ratio : float
        Proportion of data for testing.  Default: 0.15.
    stratified : bool
        Use stratified splitting (recommended).  Default True.
    random_state : int
        Random seed for reproducibility.
    shuffle : bool
        Shuffle before splitting.  Default True.

    Returns
    -------
    tuple of six np.ndarray
        ``(X_train, X_val, X_test, y_train, y_val, y_test)``

    Raises
    ------
    ValueError
        If ratios do not sum to 1.0, or if any split contains
        fewer samples than the number of classes.
    """
    # ---- Validate ratios ----
    _validate_ratios(train_ratio, val_ratio, test_ratio)

    # ---- Validate inputs ----
    validate_array_shapes(
        {"X": X, "y": y},
        expected_ndim={"X": 3, "y": 1},
    )
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X has {X.shape[0]} samples but y has {y.shape[0]}."
        )

    n_total = X.shape[0]
    n_classes = len(np.unique(y))

    logger.info("=" * 60)
    logger.info("TRAIN / VAL / TEST SPLIT")
    logger.info("=" * 60)
    logger.info(
        "Total sequences : %d", n_total,
    )
    logger.info(
        "Ratios          : %.0f%% / %.0f%% / %.0f%%",
        train_ratio * 100, val_ratio * 100, test_ratio * 100,
    )
    logger.info("Stratified      : %s", stratified)
    logger.info("Random seed     : %d", random_state)

    # ---- Step 1: split off test set ----
    # test_size is the proportion of the full dataset reserved
    # for testing.
    test_size = test_ratio
    stratify_y = y if stratified else None

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify_y,
    )

    # ---- Step 2: split trainval into train and validation ----
    # val_size_relative is the fraction of the train+val set
    # that becomes validation.
    # val_ratio / (train_ratio + val_ratio) gives the correct
    # relative proportion.
    val_size_relative = val_ratio / (train_ratio + val_ratio)
    stratify_trainval = y_trainval if stratified else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size_relative,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify_trainval,
    )

    # ---- Validate label consistency across splits ----
    check_label_consistency(y_train, y_val, y_test)

    # ---- Log split summary ----
    _log_split_summary(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        n_classes,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# Convenience: Split + Save

def split_and_save(
    X: np.ndarray = None,
    y: np.ndarray = None,
    output_dir: Optional[Path] = None,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    stratified: bool = True,
    random_state: int = RANDOM_SEED,
    dataset: str = "nsl_kdd",
    chunk_size: int = SPLIT_CHUNK_SIZE,
    x_path: str = None,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
]:
    """
    Split sequence arrays and save all six numpy arrays.

    For large datasets (> 200K sequences), automatically
    delegates to the disk-based ``split_and_save_disk`` to
    avoid excessive RAM usage.

    If ``x_path`` is provided and ``X`` is None, reads directly
    from the file — avoids loading a 7.5 GB array into RAM
    just to write it to a temp file again.

    Parameters
    ----------
    X : np.ndarray, optional
        3-D sequence array.  Ignored when ``x_path`` is given.
    y : np.ndarray
        1-D label array.
    output_dir : Path, optional
        Save directory.  Defaults to ``data/processed/``.
    train_ratio, val_ratio, test_ratio : float
        Split proportions.
    stratified : bool
    random_state : int
    dataset : str
        Dataset identifier — used only for logging.
    chunk_size : int
        Rows per chunk for disk-based splitting.
    x_path : str, optional
        Path to an existing ``.npy`` file on disk.  When
        provided, skips loading into RAM entirely.

    Returns
    -------
    tuple of six np.ndarray
        ``(X_train, X_val, X_test, y_train, y_val, y_test)``
    """
    # Disk-based path: either x_path given, or X is large
    LARGE_THRESHOLD = 200_000
    use_disk = False
    effective_x_path = x_path

    if x_path is not None and X is None:
        use_disk = True
    elif X is not None and X.shape[0] > LARGE_THRESHOLD:
        use_disk = True
        # Write X to temp file and free from RAM
        import tempfile
        with tempfile.NamedTemporaryFile(
            suffix=".npy", delete=False,
        ) as tmp:
            effective_x_path = tmp.name
        np.save(effective_x_path, X)
        del X
        import gc
        gc.collect()

    if use_disk:
        logger.info(
            "Dataset uses disk-based split — dataset: %s",
            dataset,
        )
        return split_and_save_disk(
            effective_x_path, y,
            output_dir=output_dir,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            stratified=stratified,
            random_state=random_state,
            dataset=dataset,
            chunk_size=chunk_size,
        )

    out_dir = output_dir or PROCESSED_DATA_DIR

    logger.info(
        "Splitting and saving processed arrays — dataset: %s",
        dataset,
    )

    X_train, X_val, X_test, y_train, y_val, y_test = split_sequences(
        X, y,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stratified=stratified,
        random_state=random_state,
    )

    save_processed_arrays(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        output_dir=out_dir,
    )

    logger.info(
        "Split arrays saved to: %s", out_dir,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# Memory-Efficient Disk-Based Split for Large Datasets


def split_and_save_disk(
    x_path: str,
    y: np.ndarray,
    output_dir: Optional[Path] = None,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    stratified: bool = True,
    random_state: int = RANDOM_SEED,
    dataset: str = "nsl_kdd",
    chunk_size: int = SPLIT_CHUNK_SIZE,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
]:
    """
    Memory-efficient split for large datasets.

    Memory-maps the source file, splits indices, writes each
    split directly to .npy via chunked reads — no memmap
    intermediate, no temp copies, no reload at the end.

    Peak RAM: ~1 chunk (~0.8 GB for CICIDS2017).
    """
    import gc

    out_dir = Path(output_dir or PROCESSED_DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(
        "DISK-BASED SPLIT — dataset: %s (x_path: %s)",
        dataset, x_path,
    )
    logger.info("=" * 60)

    # ---- Step 1: mmap source (zero extra RAM) ----
    X_mmap = np.load(x_path, mmap_mode="r")
    n_total, window_size, n_features = X_mmap.shape
    logger.info(
        "Loaded X via mmap — shape %s, dtype %s.",
        X_mmap.shape, X_mmap.dtype,
    )

    # ---- Step 2: split indices ----
    _validate_ratios(train_ratio, val_ratio, test_ratio)
    idx = np.arange(n_total)

    # Stratified split requires ≥2 samples per class; fall back if not met
    effective_stratify = stratified
    if stratified:
        _, class_counts = np.unique(y, return_counts=True)
        min_count = int(class_counts.min())
        if min_count < 2:
            logger.warning(
                "Stratified split requires ≥2 samples per class (min=%d) — "
                "falling back to non-stratified split.", min_count,
            )
            effective_stratify = False

    idx_trainval, idx_test, _, _ = train_test_split(
        idx, y,
        test_size=test_ratio,
        random_state=random_state,
        shuffle=True,
        stratify=y if effective_stratify else None,
    )
    val_size_relative = val_ratio / (train_ratio + val_ratio)
    idx_train, idx_val, _, _ = train_test_split(
        idx_trainval, y[idx_trainval],
        test_size=val_size_relative,
        random_state=random_state,
        shuffle=True,
        stratify=y[idx_trainval] if effective_stratify else None,
    )
    logger.info(
        "Index split — train: %d, val: %d, test: %d",
        len(idx_train), len(idx_val), len(idx_test),
    )
    del idx, idx_trainval
    gc.collect()

    # ---- Step 3: write each split directly to .npy ----
    from src.utils.constants import (
        X_TRAIN_NPY, X_VAL_NPY, X_TEST_NPY,
        Y_TRAIN_NPY, Y_VAL_NPY, Y_TEST_NPY,
    )

    split_specs = [
        (idx_train, X_TRAIN_NPY, Y_TRAIN_NPY, "train"),
        (idx_val,   X_VAL_NPY,   Y_VAL_NPY,   "val"),
        (idx_test,  X_TEST_NPY,  Y_TEST_NPY,  "test"),
    ]

    for indices, x_name, y_name, split_name in split_specs:
        n_split = len(indices)
        sorted_idx = np.sort(indices)

        # Pre-allocate output array in RAM
        X_split = np.empty((n_split, window_size, n_features), dtype=np.float32)

        for chunk_start in range(0, n_split, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_split)
            chunk_idx = sorted_idx[chunk_start:chunk_end]
            X_split[chunk_start:chunk_end] = X_mmap[chunk_idx]
            # Log every chunk to prevent Colab cell timeout
            logger.info(
                "  %s: rows %d–%d / %d written.",
                split_name, chunk_start, chunk_end, n_split,
            )

        # Save directly — no memmap intermediate
        x_out_path = out_dir / x_name
        np.save(str(x_out_path), X_split)
        del X_split
        gc.collect()

        # y split — small, write directly
        y_split = y[indices]
        np.save(str(out_dir / y_name), y_split)
        del y_split
        gc.collect()

        logger.info(
            "  %s saved: X (%d, %d, %d) → %s",
            split_name, n_split, window_size, n_features,
            x_out_path,
        )

    del X_mmap
    gc.collect()

    # ---- Step 4: validate label consistency ----
    y_train = np.load(str(out_dir / Y_TRAIN_NPY))
    y_val   = np.load(str(out_dir / Y_VAL_NPY))
    y_test  = np.load(str(out_dir / Y_TEST_NPY))
    check_label_consistency(y_train, y_val, y_test)
    del y_train, y_val, y_test
    gc.collect()

    # ---- Step 5: log summary (mmap — no extra RAM) ----
    X_tr = np.load(str(out_dir / X_TRAIN_NPY), mmap_mode="r")
    X_v  = np.load(str(out_dir / X_VAL_NPY),   mmap_mode="r")
    X_te = np.load(str(out_dir / X_TEST_NPY),  mmap_mode="r")
    y_tr = np.load(str(out_dir / Y_TRAIN_NPY))
    y_v  = np.load(str(out_dir / Y_VAL_NPY))
    y_te = np.load(str(out_dir / Y_TEST_NPY))

    _log_split_summary(X_tr, X_v, X_te, y_tr, y_v, y_te,
                        len(np.unique(np.concatenate([y_tr, y_v, y_te]))))
    logger.info("Split arrays saved to: %s", out_dir)

    return X_tr, X_v, X_te, y_tr, y_v, y_te


# Class Weight Computation

def compute_split_class_weights(
    y_train: np.ndarray,
) -> Dict[int, float]:
    """
    Compute inverse-frequency class weights from the training
    labels.

    These weights are passed to ``model.fit(class_weight=...)``
    to encourage the LSTM to attend to minority attack classes
    (R2L, U2R) rather than optimising primarily for the
    majority Normal and DoS classes.

    Formula (Chapter 3, Section 3.5.4):
        weight_c = n_samples / (n_classes * count_c)

    Parameters
    ----------
    y_train : np.ndarray
        1-D training label array.

    Returns
    -------
    dict
        ``{class_int: weight_float}``
    """
    weights = compute_class_weights(y_train)
    logger.info("Class weights for training:")
    for cls, wt in sorted(weights.items()):
        logger.info("  Class %d : %.4f", cls, wt)
    return weights


# Split Statistics

def get_split_summary(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None,
    dataset: str = "nsl_kdd",
) -> Dict:
    """
    Return a comprehensive summary dictionary describing all
    three splits.  Populates the Dataset Summary and
    Experimental Setup tables in Chapter 4.

    Parameters
    ----------
    X_train, X_val, X_test : np.ndarray
    y_train, y_val, y_test : np.ndarray
    class_names : list of str, optional
    dataset : str

    Returns
    -------
    dict
        Nested summary with per-split statistics.
    """
    names = class_names or (
        NSL_KDD_CLASS_NAMES if dataset == "nsl_kdd" else None
    )

    n_total = (
        X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    )

    summary = {
        "dataset": dataset,
        "total_sequences": n_total,
        "window_size": int(X_train.shape[1]),
        "n_features": int(X_train.shape[2]),
        "n_classes": int(len(np.unique(
            np.concatenate([y_train, y_val, y_test])
        ))),
        "splits": {
            "train": _split_stats(
                X_train, y_train, "train", n_total, names
            ),
            "val": _split_stats(
                X_val, y_val, "val", n_total, names
            ),
            "test": _split_stats(
                X_test, y_test, "test", n_total, names
            ),
        },
    }

    logger.info("Split summary:")
    for split_name, stats in summary["splits"].items():
        logger.info(
            "  %-6s %d sequences (%.1f%%) — distribution: %s",
            split_name,
            stats["n_sequences"],
            stats["pct_of_total"],
            stats["class_distribution"],
        )

    return summary


def save_split_summary(
    summary: Dict,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Save the split summary dictionary as a JSON file.

    Parameters
    ----------
    summary : dict
    output_path : Path, optional

    Returns
    -------
    Path
    """
    from src.utils.helpers import save_json
    from src.utils.paths import METRICS_DIR

    out = output_path or (METRICS_DIR / "split_summary.json")
    save_json(summary, out)
    logger.info("Split summary saved: %s", out)
    return out


# Cross-Validation Fold Generator (optional)

def stratified_kfold_splits(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = RANDOM_SEED,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generate *n_splits* stratified K-fold train/test splits.

    Provided as an optional alternative evaluation strategy.
    Not used by the primary pipeline (which uses the fixed
    70/15/15 split from Chapter 3) but available for
    sensitivity analysis.

    Parameters
    ----------
    X : np.ndarray
        3-D sequence array.
    y : np.ndarray
        1-D label array.
    n_splits : int
        Number of folds.
    random_state : int

    Returns
    -------
    list of (X_train, X_test, y_train, y_test) tuples
        One tuple per fold.
    """
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(
        skf.split(X, y)
    ):
        logger.info(
            "Fold %d/%d — train: %d, test: %d",
            fold_idx + 1, n_splits,
            len(train_idx), len(test_idx),
        )
        folds.append((
            X[train_idx], X[test_idx],
            y[train_idx], y[test_idx],
        ))
    return folds


# Internal Helpers

def _validate_ratios(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:
    """
    Raise ValueError if the three ratios do not sum to
    approximately 1.0 or if any ratio is non-positive.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must equal 1.0, "
            f"got {train_ratio} + {val_ratio} + {test_ratio} = {total:.6f}."
        )
    for name, ratio in [
        ("train_ratio", train_ratio),
        ("val_ratio", val_ratio),
        ("test_ratio", test_ratio),
    ]:
        if ratio <= 0.0 or ratio >= 1.0:
            raise ValueError(
                f"{name} must be in (0, 1), got {ratio}."
            )


def _split_stats(
    X: np.ndarray,
    y: np.ndarray,
    split_name: str,
    n_total: int,
    class_names: Optional[List[str]],
) -> Dict:
    """Return per-split statistics dictionary."""
    dist = compute_class_distribution(y, class_names)
    return {
        "n_sequences": int(X.shape[0]),
        "pct_of_total": round(X.shape[0] / n_total * 100, 2),
        "input_shape": str(X.shape),
        "class_distribution": dist,
        "n_classes": len(dist),
    }


def _log_split_summary(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
) -> None:
    """Log a clean split summary table to the logger."""
    n_total = (
        X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    )
    logger.info("-" * 60)
    logger.info(
        "  %-10s  %8s  %8s  %-20s",
        "Split", "Sequences", "% Total", "Shape",
    )
    logger.info("-" * 60)
    for name, X_s, y_s in [
        ("Train",      X_train, y_train),
        ("Validation", X_val,   y_val),
        ("Test",       X_test,  y_test),
    ]:
        pct = X_s.shape[0] / n_total * 100
        logger.info(
            "  %-10s  %8d  %7.1f%%  %-20s",
            name, X_s.shape[0], pct, str(X_s.shape),
        )
    logger.info("-" * 60)
    logger.info(
        "  %-10s  %8d  %8s  Classes: %d",
        "Total", n_total, "100.0%", n_classes,
    )
    logger.info("-" * 60)


# 2D Split for Leakage-Safe Pipeline

def split_and_save_2d(
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Optional[Path] = None,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    stratified: bool = True,
    random_state: int = RANDOM_SEED,
    dataset: str = "nsl_kdd",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """
    Split 2-D feature arrays (before sequence building and
    scaling) into train/val/test and save to disk.

    This is the leakage-safe entry point: the scaler will be
    fitted later on the training split only, and sequences are
    built per-split to avoid train/val contamination from
    overlapping windows.

    Parameters
    ----------
    X : np.ndarray
        2-D feature array of shape ``(n_samples, n_features)``.
    y : np.ndarray
        1-D integer label array of shape ``(n_samples,)``.
    output_dir : Path, optional
    train_ratio, val_ratio, test_ratio : float
    stratified : bool
    random_state : int
    dataset : str

    Returns
    -------
    tuple of six np.ndarray
        ``(X_train, X_val, X_test, y_train, y_val, y_test)``
        All X arrays are 2-D (n_samples, n_features).
    """
    out_dir = Path(output_dir or PROCESSED_DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("2D SPLIT (leakage-safe) — dataset: %s", dataset)
    logger.info("=" * 60)
    logger.info("Input shape: %s", X.shape)

    _validate_ratios(train_ratio, val_ratio, test_ratio)

    # Check stratification feasibility
    effective_stratify = stratified
    if stratified:
        _, class_counts = np.unique(y, return_counts=True)
        min_count = int(class_counts.min())
        if min_count < 2:
            logger.warning(
                "Stratified split requires >=2 samples per class "
                "(min=%d) -- falling back to non-stratified.",
                min_count,
            )
            effective_stratify = False

    # Step 1: split off test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_ratio,
        random_state=random_state,
        shuffle=True,
        stratify=y if effective_stratify else None,
    )

    # Step 2: split trainval into train and validation
    val_size_relative = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size_relative,
        random_state=random_state,
        shuffle=True,
        stratify=y_trainval if effective_stratify else None,
    )

    del X, y, X_trainval, y_trainval
    gc.collect()

    # Validate label consistency
    check_label_consistency(y_train, y_val, y_test)

    # Save splits
    from src.utils.constants import (
        X_TRAIN_NPY, X_VAL_NPY, X_TEST_NPY,
        Y_TRAIN_NPY, Y_VAL_NPY, Y_TEST_NPY,
    )

    for arr, name in [
        (X_train, X_TRAIN_NPY), (X_val, X_VAL_NPY), (X_test, X_TEST_NPY),
        (y_train, Y_TRAIN_NPY), (y_val, Y_VAL_NPY), (y_test, Y_TEST_NPY),
    ]:
        np.save(str(out_dir / name), arr)

    logger.info("2D split saved to: %s", out_dir)
    logger.info(
        "  train: %s  val: %s  test: %s",
        X_train.shape, X_val.shape, X_test.shape,
    )

    _log_split_summary(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        int(len(np.unique(np.concatenate([y_train, y_val, y_test])))),
    )

    return X_train, X_val, X_test, y_train, y_val, y_test