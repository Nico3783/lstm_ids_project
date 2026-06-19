
# src/deployment/predictor.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Load saved model + preprocessing artifacts and
#          generate predictions on new, unseen network
#          traffic data.  Used by predict.py CLI script.

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.paths import FINAL_MODEL_DIR, PREDICTIONS_DIR, ensure_dir
from src.utils.serialization import (
    load_keras_model,
    load_preprocessing_artifacts,
    load_metadata,
)
from src.utils.constants import NSL_KDD_CLASS_NAMES

logger = get_logger(__name__)


class IDSPredictor:
    """
    Encapsulates the trained LSTM model together with its
    preprocessing artifacts for end-to-end inference on
    new network traffic records.

    Usage
    -----
    >>> predictor = IDSPredictor.from_saved(FINAL_MODEL_DIR)
    >>> predictions = predictor.predict_from_csv("new_data.csv")
    >>> predictions.to_csv("predictions.csv", index=False)
    """

    def __init__(
        self,
        model: object,
        scaler: object,
        label_encoder: object,
        feature_names: List[str],
        metadata: Dict,
    ) -> None:
        self.model         = model
        self.scaler        = scaler
        self.label_encoder = label_encoder
        self.feature_names = feature_names
        self.metadata      = metadata
        self.window_size   = int(metadata.get("window_size", 10))
        self.n_features    = int(metadata.get("n_features",  41))
        self.dataset       = metadata.get("dataset", "nsl_kdd")
        self.class_names   = metadata.get(
            "class_names", NSL_KDD_CLASS_NAMES
        )
        logger.info(
            "IDSPredictor initialised — dataset: %s, "
            "window: %d, features: %d, classes: %d.",
            self.dataset, self.window_size,
            self.n_features, len(self.class_names),
        )

    # Factory
    @classmethod
    def from_saved(
        cls,
        model_dir: Optional[Path] = None,
    ) -> "IDSPredictor":
        """
        Load the predictor from saved model artifacts.

        Parameters
        ----------
        model_dir : Path, optional
            Directory containing model files.
            Defaults to ``models/final/``.

        Returns
        -------
        IDSPredictor
        """
        from src.utils.constants import FINAL_MODEL_KERAS

        model_dir = model_dir or FINAL_MODEL_DIR
        model_path = model_dir / FINAL_MODEL_KERAS

        logger.info("Loading IDSPredictor from: %s", model_dir)
        model = load_keras_model(model_path)
        scaler, label_encoder, feature_names, metadata = \
            load_preprocessing_artifacts(model_dir)

        return cls(model, scaler, label_encoder,
                   feature_names, metadata)

    # Core Prediction
    def predict(
        self,
        X_raw: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions from a raw 2-D feature array.

        The array must already have the correct feature
        columns in the correct order (matching feature_names).
        Scaling and sequence construction are applied here.

        Parameters
        ----------
        X_raw : np.ndarray
            2-D array (n_samples, n_features).

        Returns
        -------
        tuple of (y_pred, y_prob)
        """
        from src.data.sequence_builder import build_sequences
        from src.evaluation.metrics import predict_lstm

        # Scale
        X_scaled = self.scaler.transform(X_raw).astype(np.float32)

        # Build sequences
        if len(X_scaled) < self.window_size:
            raise ValueError(
                f"Need at least {self.window_size} records for "
                f"sequence construction, got {len(X_scaled)}."
            )

        # Dummy labels for sequence builder
        y_dummy = np.zeros(len(X_scaled), dtype=np.int64)
        X_seq, _ = build_sequences(
            X_scaled, y_dummy, window_size=self.window_size
        )

        y_pred, y_prob = predict_lstm(self.model, X_seq)
        return y_pred, y_prob

    def predict_from_dataframe(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Preprocess a raw DataFrame and return a predictions
        DataFrame.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Columns: sequence_id, predicted_class,
            predicted_label, confidence, + per-class probs.
        """
        from src.data.preprocessing import preprocess_new_data

        X_scaled = preprocess_new_data(
            df,
            dataset=self.dataset,
            scaler=self.scaler,
            feature_names=self.feature_names,
        )

        from src.data.sequence_builder import build_sequences
        from src.evaluation.metrics import predict_lstm

        y_dummy = np.zeros(len(X_scaled), dtype=np.int64)
        X_seq, _ = build_sequences(
            X_scaled, y_dummy,
            window_size=self.window_size,
        )

        y_pred, y_prob = predict_lstm(self.model, X_seq)

        return self._build_output_df(y_pred, y_prob)

    def predict_from_csv(
        self,
        csv_path: str,
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load a CSV file, generate predictions, and optionally
        save results.

        Parameters
        ----------
        csv_path : str
        output_path : str, optional

        Returns
        -------
        pd.DataFrame
        """
        df = pd.read_csv(csv_path, low_memory=False)
        logger.info(
            "Loaded %d records from %s.", len(df), csv_path
        )

        results = self.predict_from_dataframe(df)

        if output_path:
            ensure_dir(Path(output_path))
            results.to_csv(output_path, index=False)
            logger.info("Predictions saved: %s", output_path)

        return results

    # Helpers
    def _build_output_df(
        self,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
    ) -> pd.DataFrame:
        """Assemble the predictions output DataFrame."""
        labels = [
            self.class_names[int(p)]
            if int(p) < len(self.class_names)
            else str(int(p))
            for p in y_pred
        ]
        confidence = np.max(y_prob, axis=1)

        output: Dict = {
            "sequence_id":      np.arange(len(y_pred)),
            "predicted_class":  y_pred.astype(int),
            "predicted_label":  labels,
            "confidence":       np.round(confidence, 4),
        }
        for i, name in enumerate(self.class_names):
            if i < y_prob.shape[1]:
                output[f"prob_{name}"] = np.round(
                    y_prob[:, i], 4
                )

        return pd.DataFrame(output)