"""
tests/test_data_loading.py
Unit tests for data loaders.
"""
import sys
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.constants import NSL_KDD_COLUMNS


def _write_synthetic_nsl_kdd(path: Path, n: int = 50) -> None:
    """Write a minimal synthetic NSL-KDD-like TXT file."""
    row = ["0"] * 38 + ["tcp", "http", "SF", "normal", "15"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for _ in range(n):
            writer.writerow(row)


def test_read_nsl_kdd_file(tmp_path):
    from src.data.loaders import _read_nsl_kdd_file
    fpath = tmp_path / "KDDTrain+.txt"
    _write_synthetic_nsl_kdd(fpath, 50)
    df = _read_nsl_kdd_file(fpath)
    assert len(df) == 50
    assert len(df.columns) == len(NSL_KDD_COLUMNS)
    assert "label" in df.columns


def test_get_nsl_kdd_summary():
    from src.data.loaders import get_nsl_kdd_summary
    df = pd.DataFrame({
        "label": ["normal", "neptune"] * 15,
        "duration": range(30),
    })
    summary = get_nsl_kdd_summary(df)
    assert summary["n_samples"] == 30
    assert "attack_type_distribution" in summary


def test_get_dataset_summary_dispatch():
    from src.data.loaders import get_dataset_summary
    df = pd.DataFrame({"label": ["normal"] * 10, "duration": range(10)})
    summary = get_dataset_summary(df, "nsl_kdd")
    assert summary["dataset"] == "NSL-KDD"