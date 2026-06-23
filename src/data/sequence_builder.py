
# src/data/sequence_builder.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Convert the 2-D scaled feature matrix produced by
#          the preprocessing pipeline into 3-D sequential
#          input tensors required by the LSTM model.
#
#          Method (Chapter 3, Section 3.5.2 — Sequence
#          Construction):
#            A sliding window of width W is passed over the
#            ordered traffic records.  Each window contains
#            W consecutive connection records.  The label
#            assigned to each window is the class of the
#            FINAL timestep in that window.
#
#          Window size W = 10 was selected through
#          hyperparameter search (Chapter 3, Section 3.5.4).
#
#          Output shape:
#            X : (n_sequences, W, n_features)   float32
#            y : (n_sequences,)                 int64
#
#          This module also provides utilities for:
#            - Memory-efficient chunked sequence building
#            - Sequence statistics and logging
#            - Validation of output shapes

import math
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np

from src.utils.constants import (
    DEFAULT_WINDOW_SIZE,
    DEFAULT_STEP_SIZE,
    LABEL_POSITION,
    RANDOM_SEED,
)
from src.utils.logger import get_logger
from src.utils.helpers import Timer

logger = get_logger(__name__)


# Core Sequence Builder

def build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int = DEFAULT_WINDOW_SIZE,
    step_size: int = DEFAULT_STEP_SIZE,
    label_position: str = LABEL_POSITION,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a 2-D feature matrix and 1-D label vector into
    3-D sequential input/output arrays using a sliding window.

    Each sequence X[i] contains *window_size* consecutive
    network connection records.  The corresponding label y[i]
    is taken from the *label_position* timestep within the
    window (``"last"`` → final record's class, per Chapter 3).

    Parameters
    ----------
    X : np.ndarray
        2-D scaled feature array of shape
        ``(n_samples, n_features)`` produced by the
        preprocessing pipeline.
    y : np.ndarray
        1-D integer label array of shape ``(n_samples,)``.
    window_size : int
        Number of consecutive timesteps per sequence.
        Default: 10 (Chapter 3, Section 3.5.2).
    step_size : int
        Stride between consecutive windows.  ``1`` gives
        maximum overlap (default); larger values reduce the
        number of sequences and memory usage.
    label_position : str
        Which timestep's label to assign to the sequence.
        ``"last"``  → y[i + window_size - 1]  (default)
        ``"first"`` → y[i]
        ``"majority"`` → most frequent label in the window

    Returns
    -------
    X_seq : np.ndarray
        3-D array of shape ``(n_sequences, window_size,
        n_features)`` with dtype ``float32``.
    y_seq : np.ndarray
        1-D array of shape ``(n_sequences,)`` with dtype
        ``int64``.

    Raises
    ------
    ValueError
        If inputs are invalid or the window is larger than
        the dataset.

    Examples
    --------
    >>> X_raw = np.random.rand(1000, 41).astype(np.float32)
    >>> y_raw = np.random.randint(0, 5, 1000).astype(np.int64)
    >>> X_seq, y_seq = build_sequences(X_raw, y_raw, window_size=10)
    >>> X_seq.shape
    (991, 10, 41)
    >>> y_seq.shape
    (991,)
    """
    # ---- Input validation ----
    _validate_inputs(X, y, window_size)

    n_samples, n_features = X.shape
    n_sequences = _count_sequences(n_samples, window_size, step_size)

    logger.info(
        "Building sequences — samples: %d, window: %d, "
        "step: %d, label: '%s' → %d sequences × "
        "(%d, %d).",
        n_samples, window_size, step_size,
        label_position, n_sequences, window_size, n_features,
    )

    with Timer("Sequence construction"):
        X_seq = np.empty(
            (n_sequences, window_size, n_features),
            dtype=np.float32,
        )
        y_seq = np.empty(n_sequences, dtype=np.int64)

        for seq_idx, start in enumerate(
            range(0, n_samples - window_size + 1, step_size)
        ):
            end = start + window_size
            X_seq[seq_idx] = X[start:end]
            y_seq[seq_idx] = _get_label(
                y, start, end, label_position
            )

    logger.info(
        "Sequence construction complete — "
        "X_seq: %s (%.1f MB), y_seq: %s.",
        X_seq.shape,
        X_seq.nbytes / (1024 ** 2),
        y_seq.shape,
    )
    _log_sequence_stats(y_seq)
    return X_seq, y_seq


def build_sequences_chunked(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int = DEFAULT_WINDOW_SIZE,
    step_size: int = DEFAULT_STEP_SIZE,
    label_position: str = LABEL_POSITION,
    chunk_size: int = 50_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Memory-efficient version of ``build_sequences`` that
    processes the input in *chunk_size* windows at a time.

    Use this variant when building sequences from CICIDS2017
    or UNSW-NB15 datasets that contain millions of records
    and the full pre-allocated approach exceeds available RAM.

    Chunks are written to temporary .npy files on disk and
    loaded back one at a time, keeping peak RAM to
    ``final_array + 1_chunk`` instead of ``2 × final_array``.

    Parameters
    ----------
    X : np.ndarray
        2-D feature array.
    y : np.ndarray
        1-D label array.
    window_size : int
    step_size : int
    label_position : str
    chunk_size : int
        Number of sequences to build per chunk before
        concatenating.  Default: 50,000.

    Returns
    -------
    tuple of (X_seq, y_seq)
    """
    import gc
    import tempfile
    import shutil

    _validate_inputs(X, y, window_size)

    n_samples, n_features = X.shape
    n_sequences = _count_sequences(n_samples, window_size, step_size)

    logger.info(
        "Building sequences (chunked) — %d total sequences, "
        "chunk size: %d ...",
        n_sequences, chunk_size,
    )

    # --- Phase 1: write chunks to temp files on disk ---
    tmp_dir = Path(tempfile.mkdtemp(prefix="seq_chunks_"))
    chunk_meta: List[Tuple[Path, int]] = []  # (path, count)

    starts = range(0, n_samples - window_size + 1, step_size)
    total_chunks = math.ceil(len(starts) / chunk_size)  # type: ignore

    with Timer("Chunked sequence construction"):
        for chunk_idx, chunk_start in enumerate(
            range(0, len(starts), chunk_size)  # type: ignore
        ):
            chunk_ends = list(starts)[chunk_start: chunk_start + chunk_size]
            n_chunk = len(chunk_ends)

            X_chunk = np.empty(
                (n_chunk, window_size, n_features),
                dtype=np.float32,
            )
            y_chunk = np.empty(n_chunk, dtype=np.int64)

            for seq_idx, start in enumerate(chunk_ends):
                end = start + window_size
                X_chunk[seq_idx] = X[start:end]
                y_chunk[seq_idx] = _get_label(
                    y, start, end, label_position
                )

            # Write to disk and free immediately
            x_path = tmp_dir / f"X_chunk_{chunk_idx:04d}.npy"
            y_path = tmp_dir / f"y_chunk_{chunk_idx:04d}.npy"
            np.save(x_path, X_chunk)
            np.save(y_path, y_chunk)
            chunk_meta.append((x_path, y_path, n_chunk))
            del X_chunk, y_chunk
            gc.collect()

            logger.debug(
                "  Chunk %d/%d — %d sequences written to disk.",
                chunk_idx + 1, total_chunks, n_chunk,
            )

        # --- Phase 2: load back into final array ---
        X_seq = np.empty(
            (n_sequences, window_size, n_features),
            dtype=np.float32,
        )
        y_seq = np.empty(n_sequences, dtype=np.int64)

        offset = 0
        for x_path, y_path, n_chunk in chunk_meta:
            X_seq[offset:offset + n_chunk] = np.load(x_path)
            y_seq[offset:offset + n_chunk] = np.load(y_path)
            offset += n_chunk

    # Clean up temp files
    shutil.rmtree(tmp_dir, ignore_errors=True)
    gc.collect()

    logger.info(
        "Chunked sequence construction complete — "
        "X_seq: %s, y_seq: %s.",
        X_seq.shape, y_seq.shape,
    )
    return X_seq, y_seq


# Generator-Based Sequence Builder (for very large datasets)

def sequence_generator(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int = DEFAULT_WINDOW_SIZE,
    step_size: int = DEFAULT_STEP_SIZE,
    label_position: str = LABEL_POSITION,
    batch_size: int = 64,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Yield mini-batches of (X_batch, y_batch) sequences
    without pre-allocating the full sequence array.

    Intended for use with Keras ``model.fit(generator, ...)``
    on extremely large datasets where even chunked building
    is too memory-intensive.

    Parameters
    ----------
    X : np.ndarray
        2-D feature array.
    y : np.ndarray
        1-D label array.
    window_size : int
    step_size : int
    label_position : str
    batch_size : int
        Number of sequences per yielded batch.

    Yields
    ------
    tuple of (X_batch, y_batch)
        X_batch shape: (batch_size, window_size, n_features)
        y_batch shape: (batch_size,)
    """
    n_samples, n_features = X.shape
    starts = list(range(0, n_samples - window_size + 1, step_size))

    X_batch_list: List[np.ndarray] = []
    y_batch_list: List[int] = []

    for start in starts:
        end = start + window_size
        X_batch_list.append(X[start:end])
        y_batch_list.append(int(_get_label(y, start, end, label_position)))

        if len(X_batch_list) == batch_size:
            yield (
                np.array(X_batch_list, dtype=np.float32),
                np.array(y_batch_list, dtype=np.int64),
            )
            X_batch_list = []
            y_batch_list = []

    # Yield remaining sequences (last incomplete batch)
    if X_batch_list:
        yield (
            np.array(X_batch_list, dtype=np.float32),
            np.array(y_batch_list, dtype=np.int64),
        )


# Sequence Statistics

def get_sequence_stats(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Return a statistics dictionary for a built sequence
    array pair.  Used to populate the Chapter 4 Dataset
    Summary table entries for sequence counts and shapes.

    Parameters
    ----------
    X_seq : np.ndarray
        3-D sequence array (n_sequences, window_size,
        n_features).
    y_seq : np.ndarray
        1-D label array (n_sequences,).
    class_names : list of str, optional
        Human-readable class names for the distribution.

    Returns
    -------
    dict
        Summary statistics.
    """
    if X_seq.ndim != 3:
        raise ValueError(
            f"Expected 3-D X_seq, got {X_seq.ndim}-D shape {X_seq.shape}."
        )

    unique_labels, counts = np.unique(y_seq, return_counts=True)
    distribution: Dict = {}
    for lbl, cnt in zip(unique_labels, counts):
        name = (
            class_names[int(lbl)]
            if class_names and int(lbl) < len(class_names)
            else str(int(lbl))
        )
        distribution[name] = int(cnt)

    majority = int(max(counts))
    minority = int(min(counts))

    stats = {
        "n_sequences": int(X_seq.shape[0]),
        "window_size": int(X_seq.shape[1]),
        "n_features": int(X_seq.shape[2]),
        "input_shape": f"({X_seq.shape[0]}, {X_seq.shape[1]}, "
                       f"{X_seq.shape[2]})",
        "n_classes": int(len(unique_labels)),
        "class_distribution": distribution,
        "majority_class_count": majority,
        "minority_class_count": minority,
        "imbalance_ratio": round(majority / minority, 2)
        if minority > 0
        else float("inf"),
        "memory_mb": round(X_seq.nbytes / (1024 ** 2), 2),
        "dtype_X": str(X_seq.dtype),
        "dtype_y": str(y_seq.dtype),
    }
    return stats


def estimate_sequence_count(
    n_samples: int,
    window_size: int = DEFAULT_WINDOW_SIZE,
    step_size: int = DEFAULT_STEP_SIZE,
) -> int:
    """
    Calculate the number of sequences that will be produced
    by ``build_sequences`` without actually building them.

    Useful for pre-flight memory estimation before committing
    to sequence building on large datasets.

    Parameters
    ----------
    n_samples : int
        Total number of records.
    window_size : int
    step_size : int

    Returns
    -------
    int
        Expected number of output sequences.

    Examples
    --------
    >>> estimate_sequence_count(125973, window_size=10)
    125964
    """
    return _count_sequences(n_samples, window_size, step_size)


def estimate_memory_mb(
    n_sequences: int,
    window_size: int,
    n_features: int,
    dtype: np.dtype = np.float32,
) -> float:
    """
    Estimate the memory footprint of the sequence arrays
    in megabytes.

    Parameters
    ----------
    n_sequences : int
    window_size : int
    n_features : int
    dtype : np.dtype

    Returns
    -------
    float
        Estimated memory in MB.
    """
    bytes_per_element = np.dtype(dtype).itemsize
    total_bytes = n_sequences * window_size * n_features * bytes_per_element
    return round(total_bytes / (1024 ** 2), 2)


# Sequence Shuffling

def shuffle_sequences(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    random_state: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shuffle sequence arrays in unison along the first axis.

    Shuffling is applied AFTER sequence construction and
    AFTER the train/val/test split so that temporal order
    is preserved within each window but the order of windows
    presented to the model is randomised — preventing the
    model from exploiting the sequential order of training
    batches.

    Parameters
    ----------
    X_seq : np.ndarray
        3-D sequence array.
    y_seq : np.ndarray
        1-D label array.
    random_state : int

    Returns
    -------
    tuple of (X_shuffled, y_shuffled)
    """
    if X_seq.shape[0] != y_seq.shape[0]:
        raise ValueError(
            f"X_seq ({X_seq.shape[0]}) and y_seq "
            f"({y_seq.shape[0]}) have different lengths."
        )
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(X_seq.shape[0])
    logger.info(
        "Sequences shuffled — %d sequences, seed %d.",
        len(idx), random_state,
    )
    return X_seq[idx], y_seq[idx]


# Rebuild from Saved Arrays

def rebuild_sequences_from_flat(
    X_flat: np.ndarray,
    y_flat: np.ndarray,
    window_size: int = DEFAULT_WINDOW_SIZE,
    step_size: int = DEFAULT_STEP_SIZE,
    label_position: str = LABEL_POSITION,
    use_chunked: bool = False,
    chunk_size: int = 50_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper that selects between the standard and
    chunked sequence builders based on dataset size.

    For datasets with fewer than 500,000 records the standard
    pre-allocated builder is used.  For larger datasets the
    chunked builder is used automatically.

    Parameters
    ----------
    X_flat : np.ndarray
        2-D scaled feature array.
    y_flat : np.ndarray
        1-D label array.
    window_size : int
    step_size : int
    label_position : str
    use_chunked : bool
        Force chunked building regardless of size.
    chunk_size : int
        Chunk size for chunked building.

    Returns
    -------
    tuple of (X_seq, y_seq)
    """
    n_samples = X_flat.shape[0]

    # Estimate memory requirement before building
    n_seqs = estimate_sequence_count(n_samples, window_size, step_size)
    n_features = X_flat.shape[1]
    est_mb = estimate_memory_mb(n_seqs, window_size, n_features)

    logger.info(
        "Sequence builder — %d samples → ~%d sequences "
        "(est. %.1f MB RAM).",
        n_samples, n_seqs, est_mb,
    )

    if use_chunked or n_samples > 200_000:
        return build_sequences_chunked(
            X_flat, y_flat,
            window_size=window_size,
            step_size=step_size,
            label_position=label_position,
            chunk_size=chunk_size,
        )
    else:
        return build_sequences(
            X_flat, y_flat,
            window_size=window_size,
            step_size=step_size,
            label_position=label_position,
        )


# Internal Helpers

def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int,
) -> None:
    """
    Raise descriptive errors for invalid inputs before any
    memory is allocated.
    """
    if X.ndim != 2:
        raise ValueError(
            f"X must be 2-D (n_samples, n_features), "
            f"got shape {X.shape} ({X.ndim}-D)."
        )
    if y.ndim != 1:
        raise ValueError(
            f"y must be 1-D (n_samples,), "
            f"got shape {y.shape} ({y.ndim}-D)."
        )
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X has {X.shape[0]} samples but y has {y.shape[0]}. "
            "Sample counts must match."
        )
    if window_size < 1:
        raise ValueError(
            f"window_size must be ≥ 1, got {window_size}."
        )
    if window_size > X.shape[0]:
        raise ValueError(
            f"window_size ({window_size}) exceeds the number of "
            f"available samples ({X.shape[0]}). "
            "Reduce window_size or increase dataset size."
        )


def _count_sequences(
    n_samples: int,
    window_size: int,
    step_size: int,
) -> int:
    """Return the number of sequences produced by a sliding window."""
    if n_samples < window_size:
        return 0
    return max(0, (n_samples - window_size) // step_size + 1)


def _get_label(
    y: np.ndarray,
    start: int,
    end: int,
    label_position: str,
) -> int:
    """
    Extract the label for a window [start:end] from y.

    Parameters
    ----------
    y : np.ndarray
    start : int
    end : int
    label_position : str
        ``"last"`` | ``"first"`` | ``"majority"``

    Returns
    -------
    int
    """
    if label_position == "last":
        return int(y[end - 1])
    elif label_position == "first":
        return int(y[start])
    elif label_position == "majority":
        window_labels = y[start:end]
        values, counts = np.unique(window_labels, return_counts=True)
        return int(values[np.argmax(counts)])
    else:
        raise ValueError(
            f"Unknown label_position '{label_position}'. "
            "Choose from: 'last', 'first', 'majority'."
        )


def _log_sequence_stats(y_seq: np.ndarray) -> None:
    """
    Log class distribution of the built sequence labels.
    """
    from src.utils.constants import NSL_KDD_CLASS_NAMES
    unique, counts = np.unique(y_seq, return_counts=True)
    logger.info("  Sequence label distribution:")
    for lbl, cnt in zip(unique, counts):
        pct = cnt / len(y_seq) * 100
        name = (
            NSL_KDD_CLASS_NAMES[int(lbl)]
            if int(lbl) < len(NSL_KDD_CLASS_NAMES)
            else str(int(lbl))
        )
        logger.info(
            "    Class %d (%s): %d sequences (%.2f%%)",
            lbl, name, cnt, pct,
        )