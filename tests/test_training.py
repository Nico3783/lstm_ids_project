"""
tests/test_training.py
Unit tests for training utilities.
"""
import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_callbacks_build():
    from src.training.callbacks import build_callbacks
    cbs = build_callbacks(
        enable_tensorboard=False,
        enable_csv_logger=False,
        enable_epoch_logger=True,
    )
    cb_types = [type(c).__name__ for c in cbs]
    assert "EarlyStopping"     in cb_types
    assert "ModelCheckpoint"   in cb_types
    assert "ReduceLROnPlateau" in cb_types


def test_class_weights_inverse_frequency():
    from src.training.class_weights import get_class_weights
    y = np.array([0] * 100 + [1] * 20 + [2] * 10)
    weights = get_class_weights(y)
    assert weights[2] > weights[1] > weights[0]


def test_class_weights_all_classes_present():
    from src.training.class_weights import get_class_weights
    y = np.array([0, 1, 2, 3, 4] * 20)
    weights = get_class_weights(y)
    assert set(weights.keys()) == {0, 1, 2, 3, 4}


def test_weights_to_sample_weights():
    from src.training.class_weights import (
        get_class_weights, weights_to_sample_weights
    )
    y = np.array([0] * 50 + [1] * 10)
    cw = get_class_weights(y)
    sw = weights_to_sample_weights(y, cw)
    assert len(sw) == 60
    assert sw[0] == pytest.approx(cw[0], rel=1e-4)
    assert sw[50] == pytest.approx(cw[1], rel=1e-4)