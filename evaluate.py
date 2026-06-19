#!/usr/bin/env python3
# evaluate.py — Standalone evaluation script
# Usage:
#   python evaluate.py --dataset nsl_kdd
#   python evaluate.py --model models/final/lstm_ids_model.keras

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate the trained LSTM IDS model."
    )
    p.add_argument("--dataset", default="nsl_kdd",
                   choices=["nsl_kdd", "cicids2017", "unsw_nb15"])
    p.add_argument("--model", type=str, default=None,
                   help="Path to saved .keras model file.")
    return p.parse_args()


def main():
    args = parse_args()

    from src.utils.helpers import set_global_seed, print_banner
    from src.utils.logger import get_pipeline_logger, log_section_header
    from src.utils.paths import PROCESSED_DATA_DIR, FINAL_MODEL_DIR
    from src.utils.serialization import (
        load_processed_arrays, load_preprocessing_artifacts
    )
    from src.utils.constants import METADATA_JSON, FINAL_MODEL_KERAS
    from src.config import get_config
    from src.evaluation.metrics import compute_metrics, predict_lstm
    from src.evaluation.classification_report import (
        generate_classification_report
    )
    from src.evaluation.confusion_matrix import plot_confusion_matrix
    from src.evaluation.roc_analysis import (
        compute_roc_curves, plot_roc_curves, save_roc_scores
    )
    from src.evaluation.comparison import save_evaluation_results
    from src.visualization.training_curves import (
        save_training_history_csv
    )
    import json

    logger = get_pipeline_logger()
    print_banner("LSTM IDS — Evaluation")

    cfg = get_config()
    set_global_seed(cfg.seed)

    # Load processed arrays
    X_train, X_val, X_test, y_train, y_val, y_test = \
        load_processed_arrays(PROCESSED_DATA_DIR)

    # Load metadata
    meta_path = FINAL_MODEL_DIR / METADATA_JSON
    if not meta_path.exists():
        meta_path = PROCESSED_DATA_DIR / METADATA_JSON
    with open(meta_path) as f:
        metadata = json.load(f)
    n_classes   = metadata["n_classes"]
    class_names = metadata.get("class_names")

    # Load model
    from src.utils.serialization import load_keras_model
    model_path = Path(args.model) if args.model \
        else FINAL_MODEL_DIR / FINAL_MODEL_KERAS
    model = load_keras_model(model_path)

    logger.info("Evaluating on test set: %s", X_test.shape)

    # Predict
    y_pred, y_prob = predict_lstm(model, X_test)

    # Metrics
    metrics = compute_metrics(
        y_test, y_pred, y_prob,
        class_names=class_names,
        dataset=args.dataset,
        model_name="LSTM",
    )
    save_evaluation_results({"LSTM": metrics})

    # Classification report
    generate_classification_report(
        y_test, y_pred,
        class_names=class_names,
        dataset=args.dataset,
        model_name="LSTM",
    )

    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred,
                          class_names=class_names,
                          dataset=args.dataset)

    # ROC curves
    roc_data = compute_roc_curves(
        y_test, y_prob, class_names, args.dataset
    )
    plot_roc_curves(y_test, y_prob,
                    class_names=class_names,
                    dataset=args.dataset)
    save_roc_scores(roc_data)

    log_section_header(logger, "EVALUATION SUMMARY")
    logger.info("  Accuracy          : %.4f", metrics["accuracy"])
    logger.info("  Precision (macro) : %.4f", metrics["precision_macro"])
    logger.info("  Recall    (macro) : %.4f", metrics["recall_macro"])
    logger.info("  F1-Score  (macro) : %.4f", metrics["f1_macro"])
    if metrics["roc_auc"]:
        logger.info("  ROC-AUC           : %.4f", metrics["roc_auc"])
    logger.info("All evaluation files saved to reports/")


if __name__ == "__main__":
    main()