#!/usr/bin/env python3
# train.py — Standalone model training script
# Usage:
#   python train.py --dataset nsl_kdd
#   python train.py --dataset nsl_kdd --tune

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser(
        description="Train the LSTM IDS model."
    )
    p.add_argument("--dataset", default="nsl_kdd",
                   choices=["nsl_kdd", "cicids2017", "unsw_nb15"])
    p.add_argument("--tune", action="store_true",
                   help="Run hyperparameter tuning first.")
    p.add_argument("--epochs",      type=int, default=None)
    p.add_argument("--batch-size",  type=int, default=None)
    p.add_argument("--lr",          type=float, default=None)
    p.add_argument("--use-20pct",   action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    import json
    from src.utils.helpers import set_global_seed, print_banner
    from src.utils.logger import get_training_logger, log_section_header
    from src.utils.paths import (
        create_project_directories, PROCESSED_DATA_DIR, FINAL_MODEL_DIR
    )
    from src.config import get_config, override_dataset, override_hyperparameters
    from src.utils.serialization import load_processed_arrays
    from src.utils.constants import METADATA_JSON
    from src.visualization.training_curves import (
        plot_training_curves, save_training_history_csv
    )

    logger = get_training_logger()
    print_banner("LSTM IDS — Training")

    cfg = get_config()
    override_dataset(args.dataset)
    set_global_seed(cfg.seed)
    create_project_directories()

    # Override hyperparameters from CLI
    override_hyperparameters(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    # Load processed arrays
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = \
            load_processed_arrays(PROCESSED_DATA_DIR)
        import json
        meta_path = FINAL_MODEL_DIR / METADATA_JSON
        if not meta_path.exists():
            meta_path = PROCESSED_DATA_DIR / METADATA_JSON
        with open(meta_path) as f:
            metadata = json.load(f)
        n_classes = metadata["n_classes"]
        class_names = metadata.get("class_names")
    except FileNotFoundError:
        logger.error(
            "Processed arrays not found in %s.\n"
            "Run the full pipeline first:\n"
            "  python run_pipeline.py --dataset %s",
            PROCESSED_DATA_DIR, args.dataset,
        )
        sys.exit(1)

    logger.info(
        "Loaded arrays — X_train: %s, classes: %d",
        X_train.shape, n_classes,
    )

    # Optional hyperparameter tuning
    if args.tune:
        log_section_header(logger, "HYPERPARAMETER TUNING")
        from src.training.hyperparameter_tuning import run_random_search
        best_params, _ = run_random_search(
            X_train, y_train, X_val, y_val,
            n_classes=n_classes, config=cfg,
        )
        override_hyperparameters(
            learning_rate=best_params.get("learning_rate"),
            batch_size=best_params.get("batch_size"),
        )

    # Build and train LSTM
    from src.models.model_factory import create_model
    from src.models.model_utils import log_model_summary
    from src.training.trainer import train_lstm

    model = create_model(
        model_type="lstm",
        input_shape=(X_train.shape[1], X_train.shape[2]),
        n_classes=n_classes,
        config=cfg,
    )
    log_model_summary(model)

    result = train_lstm(
        model=model,
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        n_classes=n_classes,
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        use_class_weights=cfg.preprocessing.use_class_weights,
        config=cfg,
        dataset=args.dataset,
    )

    # Save training history CSV and generate training curve plots
    save_training_history_csv(result.history)
    plot_training_curves(
        result.history,
        model_name="LSTM",
        dataset=args.dataset,
    )

    logger.info(
        "Training done — best epoch: %d | val_loss: %.4f | "
        "val_acc: %.4f | model: %s",
        result.best_epoch, result.best_val_loss,
        result.best_val_accuracy, result.model_path,
    )
    logger.info(
        "Outputs:\n"
        "  Model            : %s\n"
        "  Training history : reports/logs/training_history.csv\n"
        "  Training curves  : reports/figures/",
        result.model_path,
    )


if __name__ == "__main__":
    main()