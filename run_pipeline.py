#!/usr/bin/env python3
# run_pipeline.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Full end-to-end pipeline runner.
#
#          Running:
#              python run_pipeline.py
#          or
#              python run_pipeline.py --dataset nsl_kdd
#
#          Automatically executes every stage:
#            1.  Validate dataset availability
#            2.  Load raw data
#            3.  Exploratory Data Analysis (EDA)
#            4.  Preprocess data
#            5.  Build sequences
#            6.  Split into train / val / test
#            7.  Save processed arrays + artifacts
#            8.  Train baseline models
#            9.  Train LSTM model
#           10.  Evaluate all models
#           11.  Generate all Chapter 4 figures
#           12.  Save reports and tables
#           13.  Export Chapter 4 ZIP archive

import argparse
import gc
import sys
from pathlib import Path

# To ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deep Learning IDS Using LSTM — Full Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["nsl_kdd", "cicids2017", "unsw_nb15"],
        default="nsl_kdd",
        help="Dataset to use (default: nsl_kdd).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume LSTM training from the last saved checkpoint.",
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip exploratory data analysis (faster runs).",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip baseline model training.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning before final training.",
    )
    parser.add_argument(
        "--use-20pct",
        action="store_true",
        help="Use NSL-KDD 20%% training subset (faster).",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=None,
        help="Use only this fraction of the dataset (e.g. 0.5 for 50%%). "
             "Useful for large datasets on memory-constrained machines.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Set log level to DEBUG.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---- Setup ----
    from src.utils.helpers import set_global_seed, print_banner, Timer
    from src.utils.logger import (
        get_pipeline_logger, set_global_log_level, log_section_header
    )
    from src.utils.paths import create_project_directories
    from src.config import get_config, override_dataset

    # GPU memory growth — prevent TF from grabbing all VRAM at once
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[GPU] Memory growth enabled for {len(gpus)} GPU(s).")
    except Exception:
        pass  # TF not available or CPU-only

    if args.verbose:
        set_global_log_level("DEBUG")

    logger = get_pipeline_logger()
    print_banner("Deep Learning IDS Using LSTM — Full Pipeline")

    cfg = get_config()
    override_dataset(args.dataset)
    set_global_seed(cfg.seed)
    create_project_directories()

    logger.info("Dataset  : %s", args.dataset)
    logger.info("Seed     : %d", cfg.seed)
    logger.info("Window   : %d", cfg.sequence.window_size)
    logger.info("Epochs   : %d", cfg.training.epochs)
    logger.info("Batch    : %d", cfg.training.batch_size)

    pipeline_timer = Timer("Full pipeline")
    pipeline_timer.start()

    # Compute dataset-specific processed data directory
    from src.utils.paths import get_processed_data_dir
    from src.utils.constants import (
        X_TRAIN_NPY, X_VAL_NPY, X_TEST_NPY,
        Y_TRAIN_NPY, Y_VAL_NPY, Y_TEST_NPY,
        SCALER_PKL, LABEL_ENCODER_PKL,
        FEATURE_NAMES_PKL, METADATA_JSON,
    )
    import numpy as np

    processed_dir = get_processed_data_dir(args.dataset)
    logger.info("Processed data dir: %s", processed_dir)

    # Check if we can resume from preprocessed data
    can_resume = True
    for fname in [X_TRAIN_NPY, X_VAL_NPY, X_TEST_NPY, Y_TRAIN_NPY, Y_VAL_NPY, Y_TEST_NPY, SCALER_PKL, LABEL_ENCODER_PKL, FEATURE_NAMES_PKL, METADATA_JSON]:
        if not (processed_dir / fname).exists():
            can_resume = False
            break

    if args.resume and can_resume:
        from src.utils.serialization import load_processed_arrays, load_preprocessing_artifacts

        X_train, X_val, X_test, y_train, y_val, y_test = load_processed_arrays(processed_dir)
        scaler, label_encoder, feature_names, metadata = load_preprocessing_artifacts(processed_dir)

        # Validate that the processed data belongs to the requested dataset
        saved_dataset = metadata.get("dataset", "")
        if saved_dataset != args.dataset:
            logger.error(
                "Dataset mismatch: processed data is for '%s' "
                "but '%s' was requested. Delete the old data and "
                "re-run without --resume.",
                saved_dataset, args.dataset,
            )
            sys.exit(1)

        n_classes = metadata["n_classes"]
        logger.info(
            "Resuming pipeline: skipping EDA, preprocessing, "
            "and sequence building stages."
        )
    else:
        if args.resume:
            logger.warning(
                "Resume flag passed, but processed arrays or "
                "preprocessing artifacts are missing. "
                "Running full preprocessing pipeline."
            )

        # STAGE 1 — Validate dataset
        log_section_header(logger, "STAGE 1 — DATASET VALIDATION")
        from src.data.download import acquire_dataset
        ok = acquire_dataset(args.dataset)
        if not ok:
            logger.error(
                "Dataset '%s' is not ready. "
                "Follow the instructions above and re-run.",
                args.dataset,
            )
            sys.exit(1)

        # STAGE 2 — Load data
        log_section_header(logger, "STAGE 2 — DATA LOADING")
        from src.data.loaders import load_dataset, get_dataset_summary
        from src.utils.helpers import save_json
        from src.utils.paths import TABLES_DIR

        main_df, _ = load_dataset(
            args.dataset,
            merge=True,
            validate=True,
            use_20pct_train=args.use_20pct,
        )

        # Subsample for memory-constrained environments
        if args.subsample and 0.0 < args.subsample < 1.0:
            from sklearn.model_selection import train_test_split
            n_before = len(main_df)
            main_df, _ = train_test_split(
                main_df,
                train_size=args.subsample,
                stratify=main_df[main_df.columns[-1]],
                random_state=42,
            )
            main_df = main_df.reset_index(drop=True)
            logger.info(
                "Subsampled to %.0f%% — %d → %d rows.",
                args.subsample * 100, n_before, len(main_df),
            )

        summary = get_dataset_summary(main_df, args.dataset)
        save_json(summary, TABLES_DIR / "dataset_summary.json")
        logger.info("Dataset loaded: %d rows.", len(main_df))

        # STAGE 3 — EDA
        if not args.skip_eda:
            log_section_header(logger, "STAGE 3 — EXPLORATORY DATA ANALYSIS")
            from src.data.exploratory import run_eda
            run_eda(main_df, dataset=args.dataset)
        else:
            logger.info("EDA skipped (--skip-eda).")

        # STAGE 4 — Preprocessing
        log_section_header(logger, "STAGE 4 — PREPROCESSING")
        from src.data.preprocessing import preprocess_dataset

        X_scaled, y, scaler, feature_names, metadata = preprocess_dataset(
            main_df,
            dataset=args.dataset,
            strategy_continuous=cfg.preprocessing.missing_strategy_continuous,
            strategy_categorical=cfg.preprocessing.missing_strategy_categorical,
            drop_first=cfg.preprocessing.drop_first,
            feature_range=cfg.preprocessing.scaler_feature_range,
            save_interim_files=True,
            artifacts_dir=processed_dir,
        )
        n_classes = metadata["n_classes"]
        logger.info(
            "Preprocessing complete — X: %s, classes: %d.",
            X_scaled.shape, n_classes,
        )
        # Free raw DataFrame from memory
        del main_df
        gc.collect()

        # STAGE 5 — Sequence building
        log_section_header(logger, "STAGE 5 — SEQUENCE BUILDING")
        from src.data.sequence_builder import (
            rebuild_sequences_from_flat, get_sequence_stats
        )

        X_seq, y_seq = rebuild_sequences_from_flat(
            X_scaled, y,
            window_size=cfg.sequence.window_size,
            step_size=cfg.sequence.step_size,
            label_position=cfg.sequence.label_position,
        )
        seq_stats = get_sequence_stats(
            X_seq, y_seq, metadata.get("class_names")
        )
        logger.info(
            "Sequences built: %d × (%d, %d).",
            X_seq.shape[0], X_seq.shape[1], X_seq.shape[2],
        )

        # STAGE 6 — Split + save
        log_section_header(logger, "STAGE 6 — TRAIN / VAL / TEST SPLIT")
        from src.data.split import split_and_save, get_split_summary
        from src.utils.serialization import save_preprocessing_artifacts
        from sklearn.preprocessing import LabelEncoder

        X_train, X_val, X_test, y_train, y_val, y_test = split_and_save(
            X_seq, y_seq,
            output_dir=processed_dir,
            train_ratio=cfg.split.train_ratio,
            val_ratio=cfg.split.val_ratio,
            test_ratio=cfg.split.test_ratio,
            stratified=cfg.split.stratified,
            dataset=args.dataset,
        )

        # Save preprocessing artifacts to models/final/ as well
        from src.utils.paths import FINAL_MODEL_DIR
        le = LabelEncoder()
        le.classes_ = np.array(metadata["class_names"])
        save_preprocessing_artifacts(
            scaler=scaler,
            label_encoder=le,
            feature_names=feature_names,
            metadata=metadata,
            output_dir=FINAL_MODEL_DIR,
        )

        # Free sequence arrays from memory (splits are now in .npy files)
        del X_seq, y_seq, X_scaled, y
        gc.collect()

        # STAGE 7 — Hyperparameter tuning (optional)
        if args.tune or cfg.hyperparameter_tuning.enabled:
            log_section_header(logger, "STAGE 7 — HYPERPARAMETER TUNING")
            from src.training.hyperparameter_tuning import (
                run_grid_search, run_random_search
            )
            from src.config import override_hyperparameters

            method = cfg.hyperparameter_tuning.method
            if method == "random_search":
                best_params, _ = run_random_search(
                    X_train, y_train, X_val, y_val,
                    n_classes=n_classes,
                    n_trials=cfg.hyperparameter_tuning.n_trials,
                    config=cfg,
                )
            else:
                best_params, _ = run_grid_search(
                    X_train, y_train, X_val, y_val,
                    n_classes=n_classes,
                    config=cfg,
                )
            override_hyperparameters(
                learning_rate=best_params.get("learning_rate"),
                batch_size=best_params.get("batch_size"),
            )
            logger.info("Best params applied: %s", best_params)
        else:
            logger.info("Hyperparameter tuning skipped.")

    # STAGE 8 — Training
    log_section_header(logger, "STAGE 8 — MODEL TRAINING")
    from src.training.trainer import run_full_training

    training_results = run_full_training(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        n_classes=n_classes,
        dataset=args.dataset,
        config=cfg,
        resume=args.resume,
    )

    # STAGE 9 — Evaluation
    log_section_header(logger, "STAGE 9 — EVALUATION")
    from src.evaluation.metrics import compute_metrics, predict_lstm
    from src.evaluation.classification_report import (
        generate_classification_report
    )
    from src.evaluation.roc_analysis import (
        compute_roc_curves, plot_roc_curves, save_roc_scores
    )
    from src.evaluation.confusion_matrix import plot_confusion_matrix
    from src.evaluation.comparison import (
        build_comparison_table, save_evaluation_results
    )
    from src.models.baseline_models import predict_baseline

    all_metrics = {}
    class_names = metadata.get("class_names")

    # Evaluate LSTM
    lstm_model = training_results["lstm"].model
    y_pred_lstm, y_prob_lstm = predict_lstm(lstm_model, X_test)
    lstm_metrics = compute_metrics(
        y_test, y_pred_lstm, y_prob_lstm,
        class_names=class_names,
        dataset=args.dataset,
        model_name="LSTM",
    )
    all_metrics["LSTM"] = lstm_metrics

    # Evaluate baselines
    if not args.skip_baselines:
        for name in ["random_forest", "svm", "logistic_regression"]:
            if name in training_results:
                bl_model = training_results[name].model
                y_pred_bl, y_prob_bl = predict_baseline(
                    bl_model, X_test, name
                )
                bl_metrics = compute_metrics(
                    y_test, y_pred_bl, y_prob_bl,
                    class_names=class_names,
                    dataset=args.dataset,
                    model_name=name.replace("_", " ").title(),
                )
                all_metrics[name.replace("_", " ").title()] = bl_metrics

    # Classification report
    generate_classification_report(
        y_test, y_pred_lstm,
        class_names=class_names,
        dataset=args.dataset,
        model_name="LSTM",
    )

    # ROC curves
    roc_data = compute_roc_curves(
        y_test, y_prob_lstm, class_names, args.dataset
    )
    plot_roc_curves(
        y_test, y_prob_lstm,
        class_names=class_names,
        dataset=args.dataset,
    )
    save_roc_scores(roc_data)

    # Confusion matrix
    plot_confusion_matrix(
        y_test, y_pred_lstm,
        class_names=class_names,
        dataset=args.dataset,
    )

    # Save all metrics
    save_evaluation_results(all_metrics)
    build_comparison_table(all_metrics)

    # STAGE 10 — Generate all Chapter 4 figures
    log_section_header(logger, "STAGE 10 — GENERATING REPORT FIGURES")
    from src.visualization.dashboard import (
        generate_all_report_figures, export_chapter4_zip
    )

    generate_all_report_figures(
        history=training_results["lstm"].history,
        y_true=y_test,
        y_pred_lstm=y_pred_lstm,
        y_prob_lstm=y_prob_lstm,
        all_metrics=all_metrics,
        dataset=args.dataset,
        class_names=class_names,
        input_shape=(cfg.sequence.window_size, X_train.shape[2]),
        n_classes=n_classes,
    )

    # STAGE 11 — Export ZIP
    log_section_header(logger, "STAGE 11 — EXPORTING RESULTS")
    zip_path = export_chapter4_zip()
    logger.info("Chapter 4 ZIP: %s", zip_path)

    # DONE
    elapsed = pipeline_timer.stop()
    log_section_header(logger, "PIPELINE COMPLETE")
    logger.info("Dataset      : %s", args.dataset)
    logger.info("Total time   : %.1f s", elapsed)
    logger.info("LSTM accuracy: %.4f",
                all_metrics.get("LSTM", {}).get("accuracy", 0))
    logger.info("LSTM F1 macro: %.4f",
                all_metrics.get("LSTM", {}).get("f1_macro", 0))
    logger.info(
        "\nAll outputs saved to:\n"
        "  Figures  : reports/figures/\n"
        "  Tables   : reports/tables/\n"
        "  Metrics  : reports/metrics/\n"
        "  Model    : models/final/\n"
        "  ZIP      : %s", zip_path,
    )


if __name__ == "__main__":
    main()