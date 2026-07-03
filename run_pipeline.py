#!/usr/bin/env python3
"""
run_pipeline.py — Master orchestrator for the LSTM-IDS project.

Runs the full pipeline end-to-end for a single dataset:
    1. Load raw data
    2. Preprocess (impute, encode, scale)
    3. Build sliding-window sequences
    4. Split into train / val / test and persist to disk
    5. (Optional) Hyperparameter tuning with Keras Tuner
    6. Train baseline models (RF, SVM, LR)
    7. Train LSTM classifier
    8. Evaluate on the test set
    9. Generate tabular and textual reports

Resume support:
    --resume   Skip stages whose .done markers and artifacts exist.
    --force S  Force re-run of specific stages (comma-separated or 'all').

Auto-sampling (Req 5-6):
    Baseline training automatically subsamples to 200K rows when the
    training set exceeds 200K sequences.  SVM uses LinearSVC for
    datasets with >50K samples.

Progress reporting (Req 10):
    Elapsed time, ETA, and RAM/GPU/disk usage are printed between stages.

Checkpointing (Req 7-8):
    LSTM training saves checkpoints every N epochs.  On resume the
    training resumes from the last saved checkpoint, including the
    optimizer state.

Usage
-----
    # Fresh run (full pipeline)
    python run_pipeline.py --dataset nsl_kdd

    # Resume after interruption — skips completed stages
    python run_pipeline.py --dataset nsl_kdd --resume

    # Force re-run specific stages
    python run_pipeline.py --dataset nsl_kdd --resume --force baselines,lstm_train

    # Force re-run everything
    python run_pipeline.py --dataset nsl_kdd --force all
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.loaders import load_dataset
from src.data.preprocessing import preprocess_dataset
from src.data.sequence_builder import build_sequences, rebuild_sequences_from_flat
from src.data.split import split_and_save
from src.evaluation.classification_report import generate_classification_report
from src.evaluation.confusion_matrix import plot_confusion_matrix
from src.evaluation.metrics import compute_metrics
from src.models.baseline_models import (
    train_all_baselines,
    save_all_baselines,
    load_all_baselines,
    predict_baseline,
)
from src.models.lstm_model import build_lstm_model
from src.models.model_factory import create_model
from src.training.trainer import run_full_training
from src.utils.config import load_config
from src.utils.helpers import set_global_seed as set_seed
from src.utils.logger import get_logger
from src.utils.paths import get_dataset_output_dirs
from src.utils.serialization import (
    save_processed_arrays,
    load_processed_arrays,
    load_split_data,
    load_split_data_train,
    load_split_data_test,
)
from src.visualization.dashboard import export_chapter4_zip
from src.visualization.training_curves import plot_training_curves
from src.evaluation.roc_analysis import plot_roc_curves

# Pipeline modules (new)
from src.pipeline.resume import ResumeManager
from src.pipeline.checkpoint import CheckpointManager
from src.pipeline.config_hash import ConfigFingerprint
from src.pipeline.progress import ProgressTracker

logger = get_logger("run_pipeline")


# ─── CLI ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full LSTM-IDS pipeline for one dataset.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["nsl_kdd", "cicids2017", "unsw_nb15"],
        help="Dataset to process.",
    )
    p.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override the base output directory (default: outputs/).",
    )

    # Resume / force
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip stages whose .done markers and artifacts already exist.",
    )
    p.add_argument(
        "--force",
        nargs="?",
        const="all",
        default=None,
        metavar="STAGES",
        help=(
            "Force re-run of stages.  Accepts:\n"
            "  'all'           — re-run every stage\n"
            "  'stage1,stage2' — comma-separated stage names\n"
            "Without a value, equivalent to --force all."
        ),
    )

    # Stage skips
    p.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip the exploratory data analysis step.",
    )
    p.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip baseline model training (RF, SVM, LR).",
    )
    p.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning before final training.",
    )

    # Data sub-sampling
    p.add_argument(
        "--subsample",
        type=float,
        default=None,
        help="Sub-sample the training data to this fraction (e.g. 0.3).",
    )

    # Training overrides
    p.add_argument("--epochs", type=int, default=None, help="Override max epochs.")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    p.add_argument("--learning-rate", type=float, default=None, help="Override learning rate.")
    p.add_argument("--sequence-length", type=int, default=None, help="Override sliding window length.")

    # Misc
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse arguments and print plan, then exit.",
    )

    return p.parse_args()


def _parse_force_stages(raw: Optional[str], all_stages: List[str]) -> List[str]:
    """Parse --force value into a list of stage names."""
    if raw is None:
        return []
    if raw.strip().lower() == "all":
        return list(all_stages)
    return [s.strip() for s in raw.split(",") if s.strip()]


# ─── GPU setup ─────────────────────────────────────────────────────────

def setup_gpu() -> None:
    """Configure GPU memory growth to prevent TF from grabbing all VRAM."""
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("GPU memory growth enabled for %d device(s).", len(gpus))
        else:
            logger.info("No GPU devices detected — running on CPU.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("GPU setup failed: %s", exc)


# ─── Helpers ───────────────────────────────────────────────────────────

def _load_n_classes(preprocessed_dir: Path, dataset: str) -> int:
    """Load the number of classes from preprocessed metadata.

    When resuming and skipping preprocessing, ``n_classes`` is not
    derived from the raw data.  This helper loads it from the saved
    label encoder or metadata.
    """
    import pickle

    le_path = preprocessed_dir / "label_encoder.pkl"
    if le_path.exists():
        with open(le_path, "rb") as f:
            le = pickle.load(f)
        return len(le.classes_)

    meta_path = preprocessed_dir / "metadata.json"
    if meta_path.exists():
        import json

        with open(meta_path, "r") as f:
            meta = json.load(f)
        return meta.get("n_classes", 15)

    # Fallback: count unique labels in y_labels.npy
    y_path = preprocessed_dir / "y_labels.npy"
    if y_path.exists():
        import numpy as np

        y = np.load(y_path)
        return int(len(set(y.tolist())))

    raise FileNotFoundError(
        f"Cannot determine n_classes from {preprocessed_dir}. "
        "Run preprocessing first or provide a valid preprocessed directory."
    )


def _setup_output_dirs(output_base: str, dataset: str) -> Dict[str, Path]:
    """Create and return the per-dataset output directory structure."""
    dirs = get_dataset_output_dirs(dataset, output_base)
    for key in ("preprocessed", "baselines", "final", "tables", "figures",
                "metrics", "logs", "config", "exported", "predictions"):
        d = dirs.get(key) or (dirs["root"] / key)
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ─── Main pipeline ─────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Logging ──────────────────────────────────────────────────────
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("Pipeline started — dataset: %s", args.dataset)

    # ── Output directories ──────────────────────────────────────────
    output_base = args.output_dir or "outputs"
    dirs = _setup_output_dirs(output_base, args.dataset)
    preprocessed_dir = dirs["preprocessed"]
    baselines_dir = dirs["baselines"]
    final_dir = dirs["final"]
    tables_dir = dirs["tables"]
    figures_dir = dirs["figures"]
    config_dir = dirs["config"]

    # ── GPU ──────────────────────────────────────────────────────────
    setup_gpu()

    # ── Config ──────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)
    config = load_config(str(config_path))
    logger.info("Config loaded: %s", config_path)

    # ── Seed ─────────────────────────────────────────────────────────
    set_seed(args.seed)

    # ── Pipeline modules ────────────────────────────────────────────
    rm = ResumeManager(args.dataset, output_base)
    fp = ConfigFingerprint(args.dataset, output_base)
    pm = ProgressTracker(args.dataset)

    # ── Resume / force plan ─────────────────────────────────────────
    force_stages = _parse_force_stages(args.force, CheckpointManager.STAGES)
    skip_stages: List[str] = []
    if args.skip_baselines:
        skip_stages.append("baselines")
    if args.skip_eda:
        # EDA is not a checkpointed stage; skip_eda only affects logging
        pass

    plan: Optional[Dict[str, Any]] = None
    if args.resume or force_stages:
        plan = rm.plan_resume(
            str(config_path),
            skip_stages=skip_stages,
            force_stages=force_stages,
        )
        logger.info(
            "Resume plan — completed: %d, to_run: %d, config_changed: %s",
            len(plan["completed"]),
            len(plan["to_run"]),
            plan["config_changed"],
        )
        if plan["config_changed"]:
            logger.warning(
                "Config fingerprint changed since last run. "
                "All stages invalidated."
            )
        print("\n" + rm.resume_summary() + "\n")

        # If nothing to run, we're done
        if not plan["to_run"]:
            logger.info("All stages completed — nothing to do.")
            return

    # ── Dry run ─────────────────────────────────────────────────────
    if args.dry_run:
        logger.info("Dry run — would execute:")
        stages = plan["to_run"] if plan else CheckpointManager.STAGES
        for s in stages:
            logger.info("  - %s", s)
        return

    # ── Config fingerprint ──────────────────────────────────────────
    fp.save_fingerprint(str(config_path), stage="pipeline_start")

    # ── Training config overrides ──────────────────────────────────
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
    if args.sequence_length is not None:
        config["preprocessing"]["sequence_length"] = args.sequence_length

    epochs = config["training"]["epochs"]

    # Determine which stages to run
    stages_to_run = plan["to_run"] if plan else list(CheckpointManager.STAGES)

    # ── Determine if we need to load preprocessed data ──────────────
    # When resuming, some stages may be skipped but later stages need
    # the preprocessed arrays.  We load them lazily.
    preprocessed_arrays: Optional[Dict[str, Any]] = None

    def _ensure_preprocessed() -> Dict[str, Any]:
        """Load preprocessed arrays if not already loaded."""
        nonlocal preprocessed_arrays
        if preprocessed_arrays is not None:
            return preprocessed_arrays
        logger.info("Loading preprocessed arrays from: %s", preprocessed_dir)
        preprocessed_arrays = load_processed_arrays(preprocessed_dir)
        return preprocessed_arrays

    # =================================================================
    # STAGE 1: Preprocessing
    # =================================================================
    if "preprocessing" in stages_to_run:
        pm.start_stage("preprocessing")
        logger.info("━━━ Stage 1/9: Preprocessing ━━━")

        logger.info("Loading raw dataset: %s ...", args.dataset)
        df_raw = load_dataset(args.dataset)
        logger.info("Raw shape: %s", df_raw.shape)

        df, feat_enc, label_enc, scaler, X_seq, y_labels = preprocess_dataset(
            df_raw, config, dataset_name=args.dataset
        )
        logger.info(
            "After preprocessing — sequences: %s, classes: %d",
            X_seq.shape, len(label_enc.classes_),
        )

        # Persist preprocessed arrays
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        np_save = __import__("numpy").save
        np_save(str(preprocessed_dir / "X_sequences.npy"), X_seq)
        np_save(str(preprocessed_dir / "y_labels.npy"), y_labels)
        import pickle as _pkl
        with open(preprocessed_dir / "label_encoder.pkl", "wb") as f:
            _pkl.dump(label_enc, f)
        with open(preprocessed_dir / "scaler.pkl", "wb") as f:
            _pkl.dump(scaler, f)
        with open(preprocessed_dir / "feature_names.pkl", "wb") as f:
            _pkl.dump(
                list(df.columns.drop("label", errors="ignore")),
                f,
            )

        n_classes = len(label_enc.classes_)
        preprocessed_arrays = {
            "X_seq": X_seq,
            "y_labels": y_labels,
            "label_enc": label_enc,
            "scaler": scaler,
            "n_classes": n_classes,
        }

        import json as _json
        with open(preprocessed_dir / "metadata.json", "w") as f:
            _json.dump({
                "dataset": args.dataset,
                "n_classes": n_classes,
                "n_samples": int(X_seq.shape[0]),
                "n_features": int(X_seq.shape[2]),
                "sequence_length": int(X_seq.shape[1]),
                "class_names": label_enc.classes_.tolist(),
            }, f, indent=2)

        rm.stage_complete("preprocessing", metadata={
            "n_samples": int(X_seq.shape[0]),
            "n_classes": n_classes,
        })
        # Free raw DataFrame from memory — no longer needed
        del df_raw, df
        pm.end_stage("preprocessing")
        gc.collect()
        logger.info("Preprocessing done.")

    # =================================================================
    # STAGE 2: Sequence building (already done above)
    # =================================================================
    if "sequence_build" in stages_to_run:
        # Sequences were built during preprocessing.
        # If resumed past preprocessing, this stage just validates.
        rm.stage_complete("sequence_build")
        logger.info("Sequence build — already complete (built during preprocessing).")

    # =================================================================
    # STAGE 3: Train/Val/Test split
    # =================================================================
    if "split_save" in stages_to_run:
        pm.start_stage("split_save")
        logger.info("━━━ Stage 3/9: Splitting data ━━━")

        data = _ensure_preprocessed()
        X_seq = data["X_seq"]
        y_labels = data["y_labels"]

        # Sub-sample if requested
        if args.subsample is not None and 0 < args.subsample < 1.0:
            import numpy as np

            n_total = X_seq.shape[0]
            n_keep = int(n_total * args.subsample)
            rng = np.random.RandomState(args.seed)
            indices = rng.choice(n_total, size=n_keep, replace=False)
            indices.sort()
            X_seq = X_seq[indices]
            y_labels = y_labels[indices]
            logger.info(
                "Sub-sampled %s → %s rows (%.1f%%)",
                f"{n_total:,}", f"{n_keep:,}", args.subsample * 100,
            )
            preprocessed_arrays["X_seq"] = X_seq
            preprocessed_arrays["y_labels"] = y_labels

        # Split into train / val / test
        X_train, X_val, X_test, y_train, y_val, y_test, scaler, label_enc = (
            split_and_save(
                X_seq,
                y_labels,
                data["scaler"],
                data["label_enc"],
                preprocessed_dir,
                seed=args.seed,
            )
        )
        logger.info(
            "Split — train: %s, val: %s, test: %s",
            X_train.shape, X_val.shape, X_test.shape,
        )

        rm.stage_complete("split_save", metadata={
            "train_shape": list(X_train.shape),
            "val_shape": list(X_val.shape),
            "test_shape": list(X_test.shape),
        })
        # Free all split arrays — they're saved to disk; loaded lazily later
        del X_train, X_val, X_test, y_train, y_val, y_test, scaler, label_enc
        del X_seq, y_labels
        if preprocessed_arrays is not None:
            preprocessed_arrays.clear()
            preprocessed_arrays = None
        pm.end_stage("split_save")
        gc.collect()
        logger.info("Data split done.")

    # =================================================================
    # Load split data lazily — only what's needed per stage
    # =================================================================
    # Training stages only need X_train, y_train, X_val, y_val, scaler, label_enc
    # Test data (X_test, y_test) is loaded on-demand for evaluation.
    logger.info("Loading split arrays from: %s", preprocessed_dir)
    X_train, X_val, y_train, y_val, scaler, label_enc = (
        load_split_data_train(preprocessed_dir)
    )

    n_classes = _load_n_classes(preprocessed_dir, args.dataset)
    logger.info(
        "Loaded — train: %s, val: %s, n_classes: %d",
        X_train.shape, X_val.shape, n_classes,
    )

    # =================================================================
    # STAGE 4: Hyperparameter tuning (optional)
    # =================================================================
    if "tuning" in stages_to_run:
        if args.tune:
            pm.start_stage("tuning")
            logger.info("━━━ Stage 4/9: Hyperparameter tuning ━━━")

            from src.tuning.tuner import run_tuning

            best_cfg = run_tuning(
                X_train, y_train, X_val, y_val,
                config, args.dataset, n_classes,
                output_dir=str(config_dir),
                seed=args.seed,
            )
            # Merge best config into main config
            if best_cfg:
                for key in ("lstm_units", "dropout", "dense_units", "learning_rate"):
                    if key in best_cfg:
                        config["training"][key] = best_cfg[key]

            rm.stage_complete("tuning", metadata={"best_config": best_cfg})
            pm.end_stage("tuning")
            gc.collect()
            logger.info("Tuning done.")
        else:
            logger.info("Tuning requested but --tune not set — skipping.")
            rm.stage_complete("tuning")

    # =================================================================
    # STAGE 5: Baseline models (RF, SVM, LR)
    # =================================================================
    if "baselines" in stages_to_run:
        if args.skip_baselines:
            logger.info("Skipping baselines (--skip-baselines).")
            rm.stage_complete("baselines")
        else:
            pm.start_stage("baselines")
            logger.info("━━━ Stage 5/9: Baseline models ━━━")

            # Auto-sampling: cap at 200K rows for baselines (Req 5-6)
            import numpy as np

            MAX_BASELINE_SAMPLES = 200_000
            X_bl = X_train
            y_bl = y_train
            if X_train.shape[0] > MAX_BASELINE_SAMPLES:
                rng = np.random.RandomState(args.seed)
                idx = rng.choice(
                    X_train.shape[0], MAX_BASELINE_SAMPLES, replace=False,
                )
                idx.sort()
                X_bl = X_train[idx]
                y_bl = y_train[idx]
                logger.info(
                    "Auto-sampled baselines: %s → %s rows",
                    f"{X_train.shape[0]:,}", f"{MAX_BASELINE_SAMPLES:,}",
                )

            fitted = train_all_baselines(X_bl, y_bl, config.get("baselines", {}))
            baselines_dir.mkdir(parents=True, exist_ok=True)
            save_all_baselines(fitted, baselines_dir)

            # Evaluate each baseline
            from src.utils.serialization import load_baseline_model

            # Load test data on-demand for baseline evaluation
            X_test, y_test = load_split_data_test(preprocessed_dir)

            results: Dict[str, Any] = {}
            for name in ["random_forest", "svm", "logistic_regression"]:
                try:
                    model = load_baseline_model(name, baselines_dir)
                    y_pred, y_prob = predict_baseline(model, X_test, name)
                    metrics = compute_metrics(y_test, y_pred, y_prob, dataset=args.dataset, model_name=name)
                    results[name] = {
                        "accuracy": metrics["accuracy"],
                        "f1_macro": metrics["f1_macro"],
                        "f1_weighted": metrics["f1_weighted"],
                        "training_samples": int(X_bl.shape[0]),
                    }
                    logger.info(
                        "%s — accuracy: %.4f, F1-macro: %.4f",
                        name, metrics["accuracy"], metrics["f1_macro"],
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Baseline %s failed: %s", name, exc)

            import json as _json

            with open(baselines_dir / "baseline_results.json", "w") as f:
                _json.dump(results, f, indent=2)

            rm.stage_complete("baselines", metadata={
                "training_samples": int(X_bl.shape[0]),
                "models": list(results.keys()),
            })
            # Free test data and baseline copies from memory
            del X_test, y_test, X_bl, y_bl, fitted
            pm.end_stage("baselines")
            gc.collect()
            logger.info("Baselines done.")

    # =================================================================
    # STAGE 6: LSTM training
    # =================================================================
    if "lstm_train" in stages_to_run:
        pm.start_stage("lstm_train")
        logger.info("━━━ Stage 6/9: LSTM training ━━━")

        # Build model
        model = create_model(config, n_classes)

        # Check for existing checkpoint to resume from
        resume_ckpt_path: Optional[str] = None
        initial_epoch = 0

        if args.resume:
            keras_ckpt = final_dir / "lstm_model.keras"
            h5_ckpt = final_dir / "lstm_model.h5"
            if keras_ckpt.exists():
                resume_ckpt_path = str(keras_ckpt)
            elif h5_ckpt.exists():
                resume_ckpt_path = str(h5_ckpt)

            if resume_ckpt_path:
                logger.info("Resuming LSTM from checkpoint: %s", resume_ckpt_path)
                model = create_model(config, n_classes)
                model.load_weights(resume_ckpt_path)

                # Determine initial_epoch from saved history
                history_csv = final_dir / "training_history.csv"
                if history_csv.exists():
                    import csv as _csv

                    with open(history_csv, "r") as f:
                        reader = _csv.DictReader(f)
                        rows = list(reader)
                    initial_epoch = len(rows)
                    logger.info(
                        "Resuming from epoch %d / %d",
                        initial_epoch, epochs,
                    )

        run_full_training(
            X_train, X_val,
            y_train, y_val,
            n_classes=n_classes,
            dataset=args.dataset,
            config=config,
            output_dir=str(final_dir),
            resume=resume_ckpt_path is not None,
        )

        rm.stage_complete("lstm_train", metadata={
            "epochs": epochs,
            "resumed_from": initial_epoch,
        })
        pm.end_stage("lstm_train")
        gc.collect()
        logger.info("LSTM training done.")

    # =================================================================
    # STAGE 7: Evaluation
    # =================================================================
    if "evaluation" in stages_to_run:
        pm.start_stage("evaluation")
        logger.info("━━━ Stage 7/9: Evaluation ━━━")

        # Load test data on-demand for evaluation
        X_test, y_test = load_split_data_test(preprocessed_dir)

        # Load trained LSTM
        from src.utils.serialization import load_keras_model as load_trained_model

        keras_path = final_dir / "lstm_model.keras"
        h5_path = final_dir / "lstm_model.h5"
        if keras_path.exists():
            lstm_model = load_trained_model(str(keras_path))
        elif h5_path.exists():
            lstm_model = load_trained_model(str(h5_path))
        else:
            logger.error("No trained LSTM found at %s", final_dir)
            sys.exit(1)

        y_pred = lstm_model.predict(X_test, verbose=0).argmax(axis=1)
        n_classes = int(max(y_test.max() + 1, y_pred.max() + 1))

        # Classification report
        report_result = generate_classification_report(
            y_test, y_pred, dataset=args.dataset,
            output_dir=str(tables_dir),
        )
        report_csv = report_result.get("csv_path", "")

        # Confusion matrix
        plot_confusion_matrix(
            y_test, y_pred, dataset=args.dataset,
            output_path=str(tables_dir),
        )

        # Per-class metrics (via compute_metrics)
        full_metrics = compute_metrics(
            y_test, y_pred, dataset=args.dataset, model_name="LSTM",
        )

        logger.info("Evaluation reports saved to: %s", tables_dir)

        rm.stage_complete("evaluation", metadata={
            "report_csv": report_csv,
        })
        # Free test data and model from memory
        del X_test, y_test, lstm_model, y_pred
        pm.end_stage("evaluation")
        gc.collect()
        logger.info("Evaluation done.")

    # =================================================================
    # STAGE 8: Visualization
    # =================================================================
    if "visualization" in stages_to_run:
        pm.start_stage("visualization")
        logger.info("━━━ Stage 8/9: Visualization ━━━")

        figures_dir.mkdir(parents=True, exist_ok=True)

        # --- 8a. Training history curves ---
        history_csv = final_dir / "training_history.csv"
        if history_csv.exists():
            import csv as _csv

            with open(history_csv, "r") as fh:
                reader = _csv.DictReader(fh)
                rows = list(reader)
            if rows:
                history_dict: Dict[str, List[float]] = {}
                for key in rows[0]:
                    if key in ("epoch",):
                        continue
                    history_dict[key] = [float(r[key]) for r in rows]
                plot_training_curves(
                    history_dict,
                    model_name="LSTM",
                    dataset=args.dataset,
                    output_dir=figures_dir,
                )
            else:
                logger.warning("Training history CSV is empty: %s", history_csv)

        # --- 8b. ROC curves ---
        from src.utils.serialization import load_keras_model as load_trained_model

        X_test_viz, y_test_viz = load_split_data_test(preprocessed_dir)

        keras_path = final_dir / "lstm_model.keras"
        h5_path = final_dir / "lstm_model.h5"
        lstm_model = None
        if keras_path.exists():
            lstm_model = load_trained_model(str(keras_path))
        elif h5_path.exists():
            lstm_model = load_trained_model(str(h5_path))

        if lstm_model is not None:
            y_pred_proba = lstm_model.predict(X_test_viz, verbose=0)

            class_names = list(label_enc.classes_)
            plot_roc_curves(
                y_test_viz, y_pred_proba,
                class_names=class_names,
                dataset=args.dataset,
                model_name="LSTM",
                output_path=figures_dir / "roc_curve.png",
            )

            del lstm_model, y_pred_proba

        del X_test_viz, y_test_viz

        rm.stage_complete("visualization")
        pm.end_stage("visualization")
        gc.collect()
        logger.info("Visualization done.")

    # =================================================================
    # STAGE 9: Export
    # =================================================================
    if "export" in stages_to_run:
        pm.start_stage("export")
        logger.info("━━━ Stage 9/9: Export ━━━")

        exported_dir = dirs["exported"]
        exported_dir.mkdir(parents=True, exist_ok=True)

        # Copy final model
        import shutil

        for fname in ["lstm_model.keras", "lstm_model.h5"]:
            src = final_dir / fname
            dst = exported_dir / fname
            if src.exists():
                shutil.copy2(src, dst)
                logger.info("Exported: %s → %s", src, dst)

        for fname in ["scaler.pkl", "label_encoder.pkl", "feature_names.pkl", "metadata.json"]:
            src = preprocessed_dir / fname
            dst = exported_dir / fname
            if src.exists():
                shutil.copy2(src, dst)
                logger.info("Exported: %s → %s", src, dst)

        # Chapter 4 ZIP
        try:
            export_chapter4_zip(str(exported_dir))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chapter 4 ZIP export failed: %s", exc)

        rm.stage_complete("export")
        pm.end_stage("export")
        gc.collect()
        logger.info("Export done.")

    # ── Summary ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Pipeline completed for dataset: %s", args.dataset)
    logger.info("Output directory: %s", dirs["root"])
    logger.info("=" * 60)

    # Print progress summary
    pm.print_summary()


# ─── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
