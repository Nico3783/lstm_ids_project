"""
tests/test_pipeline.py
End-to-end integration tests using fully synthetic data.
No dataset files required — all data is generated in-memory.
These tests verify the complete flow:
  preprocess → build sequences → split → train → evaluate
"""
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Helpers

def _synthetic_scaled(n: int = 1000, features: int = 41) -> tuple:
    """Return a 2-D scaled array and integer labels (5-class)."""
    rng = np.random.default_rng(42)
    X = rng.random((n, features)).astype(np.float32)
    y = rng.integers(0, 5, n).astype(np.int64)
    return X, y


def _synthetic_nsl_kdd_df(n: int = 200) -> pd.DataFrame:
    """Return a minimal NSL-KDD-like DataFrame for preprocessing tests."""
    from src.utils.constants import NSL_KDD_COLUMNS
    rng = np.random.default_rng(0)
    non_cat = [c for c in NSL_KDD_COLUMNS
               if c not in ["protocol_type", "service", "flag", "label", "difficulty"]]
    data = {c: rng.random(n) for c in non_cat}
    data["protocol_type"] = rng.choice(["tcp", "udp", "icmp"], n)
    data["service"]       = rng.choice(["http", "ftp", "smtp"], n)
    data["flag"]          = rng.choice(["SF", "S0", "REJ"], n)
    data["label"]         = rng.choice(
        ["normal", "neptune", "portsweep", "guess_passwd",
         "buffer_overflow"], n
    )
    data["difficulty"] = rng.integers(1, 21, n)
    return pd.DataFrame(data)


# Stage 1 — Preprocessing → Sequence → Split flow
class TestPreprocessToSplitFlow:

    def test_sequence_shapes_correct(self):
        """build_sequences output must be (N, window, features) + (N,)."""
        from src.data.sequence_builder import build_sequences
        X, y = _synthetic_scaled(1000, 41)
        X_seq, y_seq = build_sequences(X, y, window_size=10)
        assert X_seq.ndim == 3
        assert y_seq.ndim == 1
        assert X_seq.shape[0] == y_seq.shape[0]
        assert X_seq.shape[1] == 10
        assert X_seq.shape[2] == 41

    def test_split_preserves_total_count(self):
        """Sum of split sizes must equal total sequence count."""
        from src.data.sequence_builder import build_sequences
        from src.data.split import split_sequences
        X, y = _synthetic_scaled(1000, 41)
        X_seq, y_seq = build_sequences(X, y, window_size=10)
        X_tr, X_v, X_te, y_tr, y_v, y_te = split_sequences(
            X_seq, y_seq,
            train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
        )
        total = X_tr.shape[0] + X_v.shape[0] + X_te.shape[0]
        assert total == X_seq.shape[0]

    def test_split_output_shapes_3d(self):
        """All split arrays must remain 3-D."""
        from src.data.sequence_builder import build_sequences
        from src.data.split import split_sequences
        X, y = _synthetic_scaled(800, 20)
        X_seq, y_seq = build_sequences(X, y, window_size=5)
        X_tr, X_v, X_te, y_tr, y_v, y_te = split_sequences(
            X_seq, y_seq,
            train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
        )
        for arr in [X_tr, X_v, X_te]:
            assert arr.ndim == 3
        for arr in [y_tr, y_v, y_te]:
            assert arr.ndim == 1

    def test_stratification_preserves_classes(self):
        """All 5 classes must appear in every split."""
        from src.data.sequence_builder import build_sequences
        from src.data.split import split_sequences
        # Build data with exactly 5 balanced classes
        rng = np.random.default_rng(7)
        X = rng.random((2000, 10)).astype(np.float32)
        y = np.tile(np.arange(5), 400).astype(np.int64)
        X_seq, y_seq = build_sequences(X, y, window_size=5)
        X_tr, X_v, X_te, y_tr, y_v, y_te = split_sequences(
            X_seq, y_seq, stratified=True
        )
        for split_y in [y_tr, y_v, y_te]:
            assert len(np.unique(split_y)) == 5

    def test_ratio_mismatch_raises(self):
        """split_sequences must raise ValueError for bad ratios."""
        from src.data.sequence_builder import build_sequences
        from src.data.split import split_sequences
        X, y = _synthetic_scaled(200, 5)
        X_seq, y_seq = build_sequences(X, y, window_size=5)
        with pytest.raises(ValueError, match="must equal 1.0"):
            split_sequences(X_seq, y_seq,
                            train_ratio=0.6, val_ratio=0.2, test_ratio=0.1)


# Stage 2 — Full Preprocessing Pipeline
class TestPreprocessingPipeline:

    def test_preprocess_dataset_output_types(self):
        """preprocess_dataset must return correct types for all outputs."""
        from src.data.preprocessing import preprocess_dataset
        df = _synthetic_nsl_kdd_df(300)
        X_scaled, y, scaler, feature_names, metadata = preprocess_dataset(
            df, dataset="nsl_kdd", save_interim_files=False
        )
        assert isinstance(X_scaled, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X_scaled.ndim == 2
        assert y.ndim == 1
        assert len(X_scaled) == len(y)
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert "n_classes" in metadata

    def test_preprocess_scaling_range(self):
        """All features after preprocessing must be in [0, 1]."""
        from src.data.preprocessing import preprocess_dataset
        df = _synthetic_nsl_kdd_df(300)
        X_scaled, _, _, _, _ = preprocess_dataset(
            df, dataset="nsl_kdd", save_interim_files=False
        )
        assert X_scaled.min() >= -0.01
        assert X_scaled.max() <= 1.01

    def test_preprocess_label_mapping(self):
        """Labels must be mapped to integer classes 0–4."""
        from src.data.preprocessing import preprocess_dataset
        df = _synthetic_nsl_kdd_df(300)
        _, y, _, _, metadata = preprocess_dataset(
            df, dataset="nsl_kdd", save_interim_files=False
        )
        assert set(np.unique(y)).issubset({0, 1, 2, 3, 4})
        assert metadata["n_classes"] <= 5

    def test_no_missing_values_after_preprocessing(self):
        """Preprocessed X must contain no NaN or Inf values."""
        from src.data.preprocessing import preprocess_dataset
        df = _synthetic_nsl_kdd_df(200)
        # Inject some NaNs
        import random
        cols = [c for c in df.columns if df[c].dtype != object]
        for col in random.sample(cols, min(3, len(cols))):
            df.loc[df.sample(5).index, col] = np.nan
        X_scaled, _, _, _, _ = preprocess_dataset(
            df, dataset="nsl_kdd", save_interim_files=False
        )
        assert not np.isnan(X_scaled).any()
        assert not np.isinf(X_scaled).any()


# Stage 3 — LSTM Training Flow (synthetic, 2 epochs)
class TestLSTMTrainingFlow:

    def test_lstm_mini_training_smoke_test(self):
        """LSTM must complete 2 epochs and produce valid probability output."""
        from src.data.sequence_builder import build_sequences
        from src.data.split import split_sequences
        from src.models.lstm_model import build_lstm_model
        from src.utils.helpers import labels_to_one_hot

        X, y = _synthetic_scaled(n=600, features=10)
        X_seq, y_seq = build_sequences(X, y, window_size=5)
        X_tr, X_v, X_te, y_tr, y_v, y_te = split_sequences(
            X_seq, y_seq, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )
        model = build_lstm_model(
            input_shape=(5, 10), n_classes=5, lstm_units=[16, 8]
        )
        y_tr_oh = labels_to_one_hot(y_tr, 5)
        y_v_oh  = labels_to_one_hot(y_v,  5)

        history = model.fit(
            X_tr, y_tr_oh,
            validation_data=(X_v, y_v_oh),
            epochs=2, batch_size=32, verbose=0,
        )
        assert "loss" in history.history
        assert "val_loss" in history.history
        assert len(history.history["loss"]) == 2

        probs = model.predict(X_te, verbose=0)
        assert probs.shape == (X_te.shape[0], 5)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)

    def test_lstm_predictions_are_argmax_integers(self):
        """argmax predictions must be valid class integers."""
        from src.data.sequence_builder import build_sequences
        from src.models.lstm_model import build_lstm_model
        from src.utils.helpers import labels_to_one_hot

        X, y = _synthetic_scaled(n=300, features=10)
        X_seq, y_seq = build_sequences(X, y, window_size=5)
        model = build_lstm_model(input_shape=(5, 10), n_classes=5,
                                 lstm_units=[16, 8])
        y_oh = labels_to_one_hot(y_seq, 5)
        model.fit(X_seq, y_oh, epochs=1, batch_size=32, verbose=0)

        probs = model.predict(X_seq[:50], verbose=0)
        preds = np.argmax(probs, axis=1)
        assert preds.dtype in [np.int32, np.int64, int]
        assert set(preds).issubset({0, 1, 2, 3, 4})

    def test_class_weights_applied_without_error(self):
        """model.fit with class_weight dict must complete without error."""
        from src.data.sequence_builder import build_sequences
        from src.models.lstm_model import build_lstm_model
        from src.utils.helpers import labels_to_one_hot
        from src.training.class_weights import get_class_weights

        X, y = _synthetic_scaled(n=400, features=8)
        X_seq, y_seq = build_sequences(X, y, window_size=5)
        model = build_lstm_model(input_shape=(5, 8), n_classes=5,
                                 lstm_units=[16, 8])
        y_oh = labels_to_one_hot(y_seq, 5)
        class_weights = get_class_weights(y_seq)

        history = model.fit(
            X_seq, y_oh,
            epochs=1, batch_size=32,
            class_weight=class_weights, verbose=0,
        )
        assert "loss" in history.history


# Stage 4 — Evaluation Pipeline
class TestEvaluationPipeline:

    def test_compute_metrics_structure(self):
        """compute_metrics must return all required keys."""
        from src.evaluation.metrics import compute_metrics
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 5, 200).astype(np.int64)
        y_pred = rng.integers(0, 5, 200).astype(np.int64)
        metrics = compute_metrics(y_true, y_pred, model_name="test_model")

        required_keys = [
            "accuracy", "precision_macro", "recall_macro",
            "f1_macro", "f1_weighted", "confusion_matrix",
            "per_class_metrics",
        ]
        for key in required_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_accuracy_bounds(self):
        """Accuracy must be in [0, 1]."""
        from src.evaluation.metrics import compute_metrics
        y_true = np.array([0, 1, 2, 3, 4] * 20, dtype=np.int64)
        y_pred = np.array([0, 1, 2, 4, 3] * 20, dtype=np.int64)
        m = compute_metrics(y_true, y_pred)
        assert 0.0 <= m["accuracy"] <= 1.0

    def test_perfect_predictions_give_accuracy_one(self):
        """Perfect predictions must yield accuracy = 1.0."""
        from src.evaluation.metrics import compute_metrics
        y = np.array([0, 1, 2, 3, 4] * 10, dtype=np.int64)
        m = compute_metrics(y, y.copy())
        assert m["accuracy"] == pytest.approx(1.0, abs=1e-6)
        assert m["f1_macro"] == pytest.approx(1.0, abs=1e-4)

    def test_confusion_matrix_dimensions(self):
        """Confusion matrix must be square with n_classes rows and cols."""
        from src.evaluation.metrics import compute_metrics
        y_true = np.array([0, 1, 2, 3, 4] * 10, dtype=np.int64)
        y_pred = np.array([0, 1, 2, 4, 3] * 10, dtype=np.int64)
        m = compute_metrics(y_true, y_pred)
        cm = m["confusion_matrix"]
        assert len(cm) == 5
        assert all(len(row) == 5 for row in cm)

    def test_per_class_metrics_keys(self):
        """per_class_metrics must have precision, recall, f1, support per class."""
        from src.evaluation.metrics import compute_metrics
        y_true = np.array([0, 1, 2] * 30, dtype=np.int64)
        y_pred = np.array([0, 1, 1] * 30, dtype=np.int64)
        m = compute_metrics(y_true, y_pred)
        for cls_data in m["per_class_metrics"].values():
            for key in ["precision", "recall", "f1_score", "support"]:
                assert key in cls_data


# Stage 5 — Validators
class TestValidatorPipeline:

    def test_nsl_kdd_validator_passes_valid_df(self):
        """Valid NSL-KDD DataFrame must pass all error checks."""
        from src.data.validators import validate_nsl_kdd_dataframe
        from src.utils.constants import NSL_KDD_COLUMNS
        data = {}
        for col in NSL_KDD_COLUMNS:
            if col in ["protocol_type", "service", "flag"]:
                data[col] = ["tcp"] * 100
            elif col == "label":
                data[col] = ["normal"] * 100
            elif col == "difficulty":
                data[col] = [1] * 100
            else:
                data[col] = np.zeros(100)
        df = pd.DataFrame(data)
        report = validate_nsl_kdd_dataframe(df, split="test")
        assert report.passed, f"Report failed: {report.errors}"

    def test_empty_df_fails_validation(self):
        """Empty DataFrame must fail the not_empty check."""
        from src.data.validators import validate_nsl_kdd_dataframe
        import pandas as pd
        df = pd.DataFrame()
        report = validate_nsl_kdd_dataframe(df, split="empty")
        assert not report.passed

    def test_processed_arrays_validator(self):
        """Valid 3-D arrays must pass processed array validation."""
        from src.data.validators import validate_processed_arrays
        X = np.random.rand(200, 10, 41).astype(np.float32)
        y = np.random.randint(0, 5, 200).astype(np.int64)
        report = validate_processed_arrays(
            X, y, window_size=10, split="test", dataset="nsl_kdd"
        )
        assert report.passed

    def test_shape_mismatch_fails_validation(self):
        """Mismatched X and y sample counts must fail validation."""
        from src.data.validators import validate_processed_arrays
        X = np.random.rand(200, 10, 41).astype(np.float32)
        y = np.random.randint(0, 5, 150).astype(np.int64)  # Wrong size
        report = validate_processed_arrays(X, y, window_size=10)
        error_names = [c.name for c in report.errors]
        assert "sample_count_match" in error_names