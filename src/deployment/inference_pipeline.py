
# src/deployment/inference_pipeline.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: High-level inference pipeline that wraps the
#          IDSPredictor for use by predict.py and any
#          downstream integration.

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger, log_section_header
from src.utils.paths import PREDICTIONS_DIR, ensure_dir
from src.utils.constants import NSL_KDD_CLASS_NAMES

logger = get_logger(__name__)


def run_inference(
    input_path: str,
    model_dir: Optional[Path] = None,
    output_path: Optional[str] = None,
    dataset: str = "nsl_kdd",
) -> pd.DataFrame:
    """
    Run end-to-end inference on new data from a CSV file.

    Parameters
    ----------
    input_path : str
        Path to the input CSV file containing raw network
        traffic features.
    model_dir : Path, optional
        Directory containing saved model artifacts.
    output_path : str, optional
        Path to save the predictions CSV.  Auto-generated
        if not provided.
    dataset : str

    Returns
    -------
    pd.DataFrame
        Predictions DataFrame.
    """
    from src.deployment.predictor import IDSPredictor

    log_section_header(logger, "INFERENCE PIPELINE")
    logger.info("Input  : %s", input_path)
    logger.info("Dataset: %s", dataset)

    # Load predictor
    predictor = IDSPredictor.from_saved(model_dir)

    # Run prediction
    results = predictor.predict_from_csv(input_path)

    # Determine output path
    if output_path is None:
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(
            PREDICTIONS_DIR / "new_data_predictions.csv"
        )

    ensure_dir(Path(output_path))
    results.to_csv(output_path, index=False)

    # Summary
    label_counts = results["predicted_label"].value_counts()
    logger.info(
        "Inference complete — %d sequences classified.",
        len(results),
    )
    logger.info("Prediction distribution:")
    for label, count in label_counts.items():
        pct = count / len(results) * 100
        logger.info("  %-12s : %d (%.1f%%)", label, count, pct)
    logger.info("Predictions saved: %s", output_path)

    return results


def run_test_set_inference(
    model_dir: Optional[Path] = None,
    dataset: str = "nsl_kdd",
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run inference on the held-out test set using saved
    processed arrays.  Used by evaluate.py to regenerate
    predictions without re-training.

    Parameters
    ----------
    model_dir : Path, optional
    dataset : str
    output_path : str, optional

    Returns
    -------
    pd.DataFrame
        Test set predictions with true labels attached.
    """
    from src.deployment.predictor import IDSPredictor
    from src.utils.serialization import load_processed_arrays
    from src.utils.paths import PROCESSED_DATA_DIR
    from src.evaluation.metrics import predict_lstm

    log_section_header(logger, "TEST SET INFERENCE")

    # Load processed arrays
    X_train, X_val, X_test, y_train, y_val, y_test = \
        load_processed_arrays(PROCESSED_DATA_DIR)

    logger.info("Test set: %s", X_test.shape)

    # Load model
    predictor = IDSPredictor.from_saved(model_dir)
    y_pred, y_prob = predict_lstm(predictor.model, X_test)

    # Build output DataFrame
    class_names = predictor.class_names
    labels = [
        class_names[int(p)]
        if int(p) < len(class_names) else str(int(p))
        for p in y_pred
    ]
    true_labels = [
        class_names[int(t)]
        if int(t) < len(class_names) else str(int(t))
        for t in y_test
    ]

    output = pd.DataFrame({
        "true_class":      y_test.astype(int),
        "true_label":      true_labels,
        "predicted_class": y_pred.astype(int),
        "predicted_label": labels,
        "confidence":      np.round(np.max(y_prob, axis=1), 4),
        "correct":         (y_pred == y_test).astype(int),
    })

    if output_path is None:
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(PREDICTIONS_DIR / "test_predictions.csv")

    ensure_dir(Path(output_path))
    output.to_csv(output_path, index=False)

    acc = float((y_pred == y_test).mean())
    logger.info(
        "Test accuracy: %.4f | Predictions saved: %s",
        acc, output_path,
    )
    return output