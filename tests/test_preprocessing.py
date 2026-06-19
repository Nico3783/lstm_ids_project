"""
tests/test_preprocessing.py
Unit tests for the preprocessing pipeline.
"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.constants import (
    NSL_KDD_COLUMNS, NSL_KDD_CATEGORY_TO_INT, NSL_KDD_TARGET_COLUMN
)


def _make_nsl_kdd_df(n: int = 200) -> pd.DataFrame:
    """Create a minimal NSL-KDD-like DataFrame for testing."""
    rng = np.random.default_rng(42)
    numeric_cols = [
        c for c in NSL_KDD_COLUMNS
        if c not in ["protocol_type", "service", "flag",
                     NSL_KDD_TARGET_COLUMN, "difficulty"]
    ]
    data = {c: rng.random(n) for c in numeric_cols}
    data["protocol_type"] = rng.choice(["tcp", "udp", "icmp"], n)
    data["service"]       = rng.choice(["http", "ftp", "smtp"], n)
    data["flag"]          = rng.choice(["SF", "S0", "REJ"], n)
    data["label"]         = rng.choice(
        ["normal", "neptune", "portsweep", "guess_passwd",
         "buffer_overflow"], n
    )
    data["difficulty"]    = rng.integers(1, 21, n)
    return pd.DataFrame(data)


class TestDropIrrelevantColumns:
    def test_drops_difficulty(self):
        from src.data.preprocessing import drop_irrelevant_columns
        df = _make_nsl_kdd_df()
        assert "difficulty" in df.columns
        result = drop_irrelevant_columns(df, dataset="nsl_kdd")
        assert "difficulty" not in result.columns

    def test_drops_split_column(self):
        from src.data.preprocessing import drop_irrelevant_columns
        df = _make_nsl_kdd_df()
        df["_split"] = "train"
        result = drop_irrelevant_columns(df, dataset="nsl_kdd")
        assert "_split" not in result.columns


class TestHandleMissingInfinite:
    def test_fills_numeric_nan(self):
        from src.data.preprocessing import handle_missing_and_infinite
        df = _make_nsl_kdd_df(100)
        df.loc[0:10, "duration"] = np.nan
        result = handle_missing_and_infinite(df)
        assert result["duration"].isnull().sum() == 0

    def test_replaces_inf(self):
        from src.data.preprocessing import handle_missing_and_infinite
        df = _make_nsl_kdd_df(50)
        df.loc[0, "duration"] = np.inf
        result = handle_missing_and_infinite(df)
        assert not np.isinf(result["duration"]).any()


class TestRemoveDuplicates:
    def test_removes_exact_duplicates(self):
        from src.data.preprocessing import remove_duplicates
        df = _make_nsl_kdd_df(50)
        df_dup = pd.concat([df, df.head(10)], ignore_index=True)
        result = remove_duplicates(df_dup)
        assert len(result) == 50


class TestMapNSLKDDLabels:
    def test_maps_to_integers(self):
        from src.data.preprocessing import map_nsl_kdd_labels
        df = _make_nsl_kdd_df(100)
        result = map_nsl_kdd_labels(df)
        assert result["label"].dtype in [np.int32, np.int64, int]
        assert set(result["label"].unique()).issubset(
            set(NSL_KDD_CATEGORY_TO_INT.values())
        )

    def test_normal_maps_to_zero(self):
        from src.data.preprocessing import map_nsl_kdd_labels
        df = pd.DataFrame({"label": ["normal"] * 10})
        # Add minimal required columns
        for col in NSL_KDD_COLUMNS:
            if col not in df.columns:
                df[col] = 0
        df["label"] = "normal"
        result = map_nsl_kdd_labels(df)
        assert (result["label"] == 0).all()


class TestEncoding:
    def test_one_hot_creates_columns(self):
        from src.data.preprocessing import encode_categorical_features
        df = _make_nsl_kdd_df(50)
        df.drop(columns=["difficulty"], inplace=True, errors="ignore")
        # Map labels first
        from src.data.preprocessing import map_nsl_kdd_labels
        df = map_nsl_kdd_labels(df)
        result = encode_categorical_features(df, dataset="nsl_kdd")
        assert "protocol_type" not in result.columns
        # Should have one-hot columns
        ohe_cols = [c for c in result.columns
                    if c.startswith("protocol_type_")]
        assert len(ohe_cols) >= 2


class TestScaler:
    def test_minmax_range(self):
        from src.data.preprocessing import fit_scaler, apply_scaler
        X = pd.DataFrame(np.random.rand(100, 5) * 100)
        scaler = fit_scaler(X)
        X_scaled = apply_scaler(X, scaler)
        assert X_scaled.min() >= -0.01
        assert X_scaled.max() <= 1.01

    def test_scaler_fitted_on_train_only(self):
        from src.data.preprocessing import (
            fit_scaler, apply_scaler, preprocess_train_val_test
        )
        X_train = pd.DataFrame(np.random.rand(80, 5))
        X_val   = pd.DataFrame(np.random.rand(10, 5))
        X_test  = pd.DataFrame(np.random.rand(10, 5))
        X_tr_s, X_v_s, X_te_s, scaler = preprocess_train_val_test(
            X_train, X_val, X_test
        )
        assert X_tr_s.shape == (80, 5)
        assert X_v_s.shape  == (10, 5)
        assert X_te_s.shape == (10, 5)