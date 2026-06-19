"""
tests/test_sequence_builder.py
Unit tests for the sliding window sequence builder.
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_output_shape():
    from src.data.sequence_builder import build_sequences
    X = np.random.rand(500, 41).astype(np.float32)
    y = np.random.randint(0, 5, 500).astype(np.int64)
    X_seq, y_seq = build_sequences(X, y, window_size=10)
    assert X_seq.shape == (491, 10, 41)
    assert y_seq.shape == (491,)


def test_label_is_last_timestep():
    from src.data.sequence_builder import build_sequences
    X = np.random.rand(100, 5).astype(np.float32)
    y = np.arange(100, dtype=np.int64)
    _, y_seq = build_sequences(X, y, window_size=10, label_position="last")
    # First sequence covers indices 0-9, label = y[9] = 9
    assert y_seq[0] == 9


def test_window_larger_than_data_raises():
    from src.data.sequence_builder import build_sequences
    with pytest.raises(ValueError):
        build_sequences(
            np.random.rand(5, 10).astype(np.float32),
            np.zeros(5, dtype=np.int64),
            window_size=10,
        )


def test_estimate_count_nsl_kdd():
    from src.data.sequence_builder import estimate_sequence_count
    assert estimate_sequence_count(125973, 10, 1) == 125964