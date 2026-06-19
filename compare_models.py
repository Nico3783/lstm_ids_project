#!/usr/bin/env python3
# compare_models.py — Compare all models and generate
#                     Chapter 4 comparison figures/tables.
# Usage:
#   python compare_models.py --dataset nsl_kdd

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare LSTM vs baseline models."
    )
    p.add_argument("--dataset", default="nsl_kdd",
                   choices=["nsl_kdd", "cicids2017", "unsw_nb15"])
    return p.parse_args()


def main():
    args = parse_args()

    from src.utils.helpers import set_global_seed, print_banner
    from src.utils.logger import get_pipeline_logger, log_section_header
    from src.utils.paths import (
        PROCESSED_DATA_DIR, FINAL_MODEL_DIR, BASELINES_DIR
    )
    from src.utils.serialization import (
        load_processed_arrays, load_keras_model
    )
    from src.utils.constants import FINAL_MODEL_KERAS, METADATA_JSON
    from src.config import get_config
    from src.evaluation.metrics import compute_metrics, predict_lstm
    from src.models.baseline_models import (
        load_all_baselines, predict_baseline
    )
    from src.evaluation.comparison import (
        build_comparison_table, plot_model_comparison,
        save_evaluation_results
    )
    import json

    logger = get_pipeline_logger()
    print_banner("LSTM IDS — Model Comparison")

    cfg = get_config()
    set_global_seed(cfg.seed)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = \
        load_processed_arrays(PROCESSED_DATA_DIR)

    meta_path = FINAL_MODEL_DIR / METADATA_JSON
    if not meta_path.exists():
        meta_path = PROCESSED_DATA_DIR / METADATA_JSON
    with open(meta_path) as f:
        metadata = json.load(f)
    class_names = metadata.get("class_names")

    all_metrics = {}

    # ---- LSTM ----
    log_section_header(logger, "EVALUATING LSTM")
    lstm_model = load_keras_model(FINAL_MODEL_DIR / FINAL_MODEL_KERAS)
    y_pred_lstm, y_prob_lstm = predict_lstm(lstm_model, X_test)
    all_metrics["LSTM"] = compute_metrics(
        y_test, y_pred_lstm, y_prob_lstm,
        class_names=class_names,
        dataset=args.dataset,
        model_name="LSTM",
    )

    # ---- Baselines ----
    log_section_header(logger, "EVALUATING BASELINES")
    baselines = load_all_baselines(BASELINES_DIR)
    for name, model in baselines.items():
        display = name.replace("_", " ").title()
        logger.info("Evaluating %s ...", display)
        y_pred_bl, y_prob_bl = predict_baseline(model, X_test, name)
        all_metrics[display] = compute_metrics(
            y_test, y_pred_bl, y_prob_bl,
            class_names=class_names,
            dataset=args.dataset,
            model_name=display,
        )

    # ---- Generate outputs ----
    log_section_header(logger, "COMPARISON OUTPUTS")
    table = build_comparison_table(all_metrics)
    plot_model_comparison(all_metrics, dataset=args.dataset)
    save_evaluation_results(all_metrics)

    logger.info("\nModel Comparison Results:")
    print(table.to_string(index=False))
    logger.info("Comparison chart saved to reports/figures/")


if __name__ == "__main__":
    main()