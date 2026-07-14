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
import numpy as np
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
from src.data.split import split_and_save, split_and_save_2d
from src.evaluation.classification_report import generate_classification_report
from src.evaluation.confusion_matrix import plot_confusion_matrix
from src.evaluation.metrics import compute_metrics
from src.models.baseline_models import (
    train_all_baselines,
    save_all_baselines,
    load_all_baselines,
    predict_baseline,
)
from src.training.trainer import run_full_training
from src.config import reload_config
from src.utils.helpers import set_global_seed as set_seed
from src.utils.logger import get_logger
from src.utils.paths import get_dataset_output_dirs
from src.utils.serialization import (
    save_processed_arrays,
    load_processed_arrays,
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

    # Stage range control (for per-cell Colab execution)
    STAGE_LIST = [
        "preprocessing", "split_save", "scale_sequences",
        "tuning", "baselines", "lstm_train",
        "evaluation", "visualization", "export",
    ]
    p.add_argument(
        "--start-stage",
        type=str,
        default=None,
        choices=STAGE_LIST,
        help="Start pipeline from this stage (skip earlier stages).",
    )
    p.add_argument(
        "--end-stage",
        type=str,
        default=None,
        choices=STAGE_LIST,
        help="Stop pipeline after this stage (inclusive).",
    )

    # Drive backup
    p.add_argument(
        "--zip-drive",
        action="store_true",
        help="After pipeline completes, zip outputs to Drive and free local disk.",
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
        y = np.load(y_path)
        return int(len(set(y.tolist())))

    raise FileNotFoundError(
        f"Cannot determine n_classes from {preprocessed_dir}. "
        "Run preprocessing first or provide a valid preprocessed directory."
    )


def _setup_output_dirs(output_base: str, dataset: str) -> Dict[str, Path]:
    """Create and return the per-dataset output directory structure.

    ``get_dataset_output_dirs`` provides the core keys.  This helper
    adds the aliases that ``main()`` expects and creates every
    subdirectory.
    """
    dirs = get_dataset_output_dirs(dataset, output_base)

    # Aliases expected by main() — map friendly names to real paths
    dirs["preprocessed"] = dirs["root"] / "preprocessed"
    dirs["baselines"]    = dirs["models_baselines"]
    dirs["final"]        = dirs["models_final"]
    dirs["config"]       = dirs["root"] / "config"

    # Create every directory
    for d in dirs.values():
        if isinstance(d, Path):
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

    # ── Colab keepalive — prints heartbeat every 5 min ────────────
    import threading

    _keepalive_stop = threading.Event()

    def _keepalive():
        while not _keepalive_stop.wait(300):
            logger.info("[KEEPALIVE] still running ...")

    _ka_thread = threading.Thread(target=_keepalive, daemon=True)
    _ka_thread.start()

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
    config = reload_config(str(config_path))
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
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.model.learning_rate = args.learning_rate
    if args.sequence_length is not None:
        config.sequence.window_size = args.sequence_length

    epochs = config.training.epochs

    # Determine which stages to run
    stages_to_run = plan["to_run"] if plan else list(CheckpointManager.STAGES)

    # Apply --start-stage / --end-stage filters
    if args.start_stage is not None:
        start_idx = CheckpointManager.STAGES.index(args.start_stage)
        stages_to_run = [s for s in stages_to_run if CheckpointManager.STAGES.index(s) >= start_idx]
        logger.info("Starting from stage: %s (index %d)", args.start_stage, start_idx)
    if args.end_stage is not None:
        end_idx = CheckpointManager.STAGES.index(args.end_stage)
        stages_to_run = [s for s in stages_to_run if CheckpointManager.STAGES.index(s) <= end_idx]
        logger.info("Stopping at stage: %s (index %d)", args.end_stage, end_idx)

    # ── Determine if we need to load preprocessed data ──────────────
    # When resuming, some stages may be skipped but later stages need
    # the preprocessed arrays.  We load them lazily.
    preprocessed_arrays: Optional[Dict[str, Any]] = None

    def _ensure_preprocessed() -> Dict[str, Any]:
        """Load preprocessed arrays (X_seq, y_labels) if not already loaded."""
        nonlocal preprocessed_arrays
        if preprocessed_arrays is not None and "X_seq" in preprocessed_arrays:
            return preprocessed_arrays
        logger.info("Loading preprocessed arrays from: %s", preprocessed_dir)
        X_seq = np.load(str(preprocessed_dir / "X_sequences.npy"))
        y_labels = np.load(str(preprocessed_dir / "y_labels.npy"))

        # Try to recover n_classes from metadata
        import json as _json
        meta_path = preprocessed_dir / "metadata.json"
        n_classes = int(np.max(y_labels)) + 1
        if meta_path.exists():
            with open(meta_path) as f:
                meta = _json.load(f)
            n_classes = meta.get("n_classes", n_classes)

        preprocessed_arrays = {
            "X_seq": X_seq,
            "y_labels": y_labels,
            "n_classes": n_classes,
        }
        logger.info("Loaded — X: %s  y: %s  classes: %d", X_seq.shape, y_labels.shape, n_classes)
        return preprocessed_arrays

    # =================================================================
    # STAGE 1: Preprocessing (leakage-safe: no scaling yet)
    # =================================================================
    if "preprocessing" in stages_to_run:
        pm.start_stage("preprocessing")
        logger.info("━━━ Stage 1/9: Preprocessing (2D, no scaling) ━━━")

        logger.info("Loading raw dataset: %s ...", args.dataset)
        df_raw, df_test_unused = load_dataset(args.dataset)
        logger.info("Raw shape: %s", df_raw.shape)

        # Sub-sample early to avoid OOM during sequence building
        if args.subsample is not None and 0 < args.subsample < 1.0:
            n_total = len(df_raw)
            n_keep = int(n_total * args.subsample)
            rng = np.random.RandomState(args.seed)
            indices = rng.choice(n_total, size=n_keep, replace=False)
            indices.sort()
            df_raw = df_raw.iloc[indices].reset_index(drop=True)
            logger.info(
                "Sub-sampled raw DataFrame %s -> %s rows (%.1f%%)",
                f"{n_total:,}", f"{n_keep:,}", args.subsample * 100,
            )
            del indices
            gc.collect()

        # skip_scaling=True: scaler fitted later on train split only
        X_2d, y_labels, _scaler_unused, feature_names, metadata = preprocess_dataset(
            df_raw, dataset=args.dataset, skip_scaling=True,
        )

        # Build a LabelEncoder from metadata for downstream compatibility
        from sklearn.preprocessing import LabelEncoder as _LE
        label_enc = _LE()
        label_enc.classes_ = np.array(metadata["class_names"])

        n_classes = int(metadata["n_classes"])
        logger.info(
            "After preprocessing — 2D array: %s, classes: %d",
            X_2d.shape, n_classes,
        )

        # Persist 2D unscaled arrays
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(preprocessed_dir / "X_2d.npy"), X_2d)
        np.save(str(preprocessed_dir / "y_labels.npy"), y_labels)
        import pickle as _pkl
        with open(preprocessed_dir / "label_encoder.pkl", "wb") as f:
            _pkl.dump(label_enc, f)
        with open(preprocessed_dir / "feature_names.pkl", "wb") as f:
            _pkl.dump(feature_names, f)
        preprocessed_arrays = {
            "X_2d": X_2d,
            "y_labels": y_labels,
            "label_enc": label_enc,
            "n_classes": n_classes,
        }

        import json as _json
        with open(preprocessed_dir / "metadata.json", "w") as f:
            _json.dump({
                "dataset": args.dataset,
                "n_classes": n_classes,
                "n_samples": int(X_2d.shape[0]),
                "n_features": int(X_2d.shape[1]),
                "class_names": label_enc.classes_.tolist(),
                "skip_scaling": True,
                "note": "2D unscaled array; scaler fitted in stage 3",
            }, f, indent=2)

        rm.stage_complete("preprocessing", metadata={
            "n_samples": int(X_2d.shape[0]),
            "n_classes": n_classes,
        })
        del df_raw, label_enc, feature_names
        pm.end_stage("preprocessing")
        gc.collect()
        logger.info("Preprocessing done.")

    # =================================================================
    # STAGE 2: Train/Val/Test split (2D, before scaling)
    # =================================================================
    if "split_save" in stages_to_run:
        pm.start_stage("split_save")
        logger.info("━━━ Stage 2/9: Splitting 2D data ━━━")

        X_2d = np.load(str(preprocessed_dir / "X_2d.npy"))
        y_labels = np.load(str(preprocessed_dir / "y_labels.npy"))
        logger.info("Loaded 2D data: X=%s  y=%s", X_2d.shape, y_labels.shape)

        X_train, X_val, X_test, y_train, y_val, y_test = split_and_save_2d(
            X=X_2d, y=y_labels,
            output_dir=preprocessed_dir,
            random_state=args.seed,
            dataset=args.dataset,
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
        del X_2d, y_labels, X_train, X_val, X_test, y_train, y_val, y_test
        if preprocessed_arrays is not None:
            preprocessed_arrays.clear()
            preprocessed_arrays = None
        pm.end_stage("split_save")
        gc.collect()
        logger.info("Data split done.")

    # =================================================================
    # STAGE 3: Scale per-split + build 3D sequences (leakage-safe)
    # =================================================================
    if "scale_sequences" in stages_to_run:
        pm.start_stage("scale_sequences")
        logger.info("━━━ Stage 3/9: Scale + sequences (per-split) ━━━")

        from src.data.preprocessing import fit_scaler, apply_scaler
        from src.utils.constants import (
            X_TRAIN_NPY, X_VAL_NPY, X_TEST_NPY,
            Y_TRAIN_NPY, Y_VAL_NPY, Y_TEST_NPY,
        )
        window_size = config.window_size
        step_size = config.sequence.step_size
        label_position = config.sequence.label_position
        logger.info(
            "Sliding window: size=%d  step=%d  label=%s",
            window_size, step_size, label_position,
        )

        def _scale_and_build(split_label, x_npy, y_npy, scaler_path):
            """Load 2D split, fit/apply scaler, build sequences, save 3D."""
            X_2d = np.load(str(preprocessed_dir / x_npy))
            y_1d = np.load(str(preprocessed_dir / y_npy))
            logger.info(
                "[%s] Loaded 2D: X=%s y=%s", split_label, X_2d.shape, y_1d.shape,
            )

            if split_label == "train":
                scaler = fit_scaler(
                    pd.DataFrame(X_2d),
                    feature_range=(0, 1),
                )
                import pickle as _pkl
                with open(preprocessed_dir / scaler_path, "wb") as f:
                    _pkl.dump(scaler, f)
                logger.info("[%s] Scaler fitted on training data.", split_label)
            else:
                import pickle as _pkl
                with open(preprocessed_dir / scaler_path, "rb") as f:
                    scaler = _pkl.load(f)
                logger.info("[%s] Loaded training scaler.", split_label)

            X_scaled = apply_scaler(pd.DataFrame(X_2d), scaler)

            X_seq, y_seq = build_sequences(
                X_scaled, y_1d,
                window_size=window_size,
                step_size=step_size,
                label_position=label_position,
            )
            logger.info(
                "[%s] 3D sequences: X=%s y=%s",
                split_label, X_seq.shape, y_seq.shape,
            )
            return X_seq, y_seq

        # Process train (fits scaler), then val/test (load same scaler)
        X_train_seq, y_train_seq = _scale_and_build(
            "train", X_TRAIN_NPY, Y_TRAIN_NPY, "scaler.pkl",
        )
        X_val_seq, y_val_seq = _scale_and_build(
            "val", X_VAL_NPY, Y_VAL_NPY, "scaler.pkl",
        )
        X_test_seq, y_test_seq = _scale_and_build(
            "test", X_TEST_NPY, Y_TEST_NPY, "scaler.pkl",
        )

        # Overwrite split files with 3D sequences
        np.save(str(preprocessed_dir / X_TRAIN_NPY), X_train_seq)
        np.save(str(preprocessed_dir / Y_TRAIN_NPY), y_train_seq)
        np.save(str(preprocessed_dir / X_VAL_NPY), X_val_seq)
        np.save(str(preprocessed_dir / Y_VAL_NPY), y_val_seq)
        np.save(str(preprocessed_dir / X_TEST_NPY), X_test_seq)
        np.save(str(preprocessed_dir / Y_TEST_NPY), y_test_seq)

        # Clean up 2D intermediates
        for f in ["X_2d.npy"]:
            p = preprocessed_dir / f
            if p.exists():
                p.unlink()

        # Update metadata
        import json as _json
        meta_path = preprocessed_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = _json.load(f)
        else:
            meta = {}
        meta["skip_scaling"] = False
        meta["sequence_length"] = window_size
        meta["n_sequences_train"] = int(X_train_seq.shape[0])
        meta["n_sequences_val"] = int(X_val_seq.shape[0])
        meta["n_sequences_test"] = int(X_test_seq.shape[0])
        meta["sequence_shape_3d_train"] = list(X_train_seq.shape)
        with open(meta_path, "w") as f:
            _json.dump(meta, f, indent=2)

        rm.stage_complete("scale_sequences", metadata={
            "train_shape": list(X_train_seq.shape),
            "val_shape": list(X_val_seq.shape),
            "test_shape": list(X_test_seq.shape),
        })
        del X_train_seq, X_val_seq, X_test_seq, y_train_seq, y_val_seq, y_test_seq
        pm.end_stage("scale_sequences")
        gc.collect()
        logger.info("Scale + sequences done.")

    # =================================================================
    # Load split data lazily — only when a stage that needs it starts.
    # This block MUST run BEFORE all stage execution blocks so that
    # stages_to_run is finalized (with any auto-prepended prereqs)
    # before any stage checks membership in it.
    # =================================================================
    stages_needing_splits = {"tuning", "baselines", "lstm_train", "evaluation", "visualization", "export"}

    X_train = X_val = y_train = y_val = scaler = label_enc = None
    n_classes = None
    splits_loaded = False

    def _ensure_splits_loaded():
        """Load split arrays on first demand (after prepended stages have run)."""
        nonlocal X_train, X_val, y_train, y_val, scaler, label_enc, n_classes, splits_loaded
        if splits_loaded:
            return
        logger.info("Loading split arrays from: %s", preprocessed_dir)
        X_train, X_val, y_train, y_val, scaler, label_enc = (
            load_split_data_train(preprocessed_dir)
        )
        n_classes = _load_n_classes(preprocessed_dir, args.dataset)
        splits_loaded = True
        logger.info(
            "Loaded — train: %s, val: %s, n_classes: %d",
            X_train.shape, X_val.shape, n_classes,
        )

    # Auto-detect missing prerequisites: if user starts mid-pipeline
    # (e.g. --start-stage lstm_train) but earlier stages haven't been
    # run, prepend them automatically instead of crashing.
    needs_splits = stages_needing_splits & set(stages_to_run)
    if needs_splits:
        root_done_dir = dirs["root"] / ".checkpoints"
        preproc_marker = root_done_dir / "preprocessing.done"
        split_marker = root_done_dir / "split_save.done"
        scale_marker = root_done_dir / "scale_sequences.done"
        x_train_exists = (preprocessed_dir / "X_train.npy").exists()

        missing = []
        if not preproc_marker.exists():
            missing.append("preprocessing")
        if not split_marker.exists():
            missing.append("split_save")
        if not scale_marker.exists():
            missing.append("scale_sequences")
        if not x_train_exists:
            if "scale_sequences" not in missing:
                missing.append("scale_sequences")

        if missing:
            logger.warning(
                "Prerequisite stages missing: %s — auto-prepending to pipeline.",
                ", ".join(missing),
            )
            prereq_stages = []
            for stage_name in CheckpointManager.STAGES:
                if stage_name in missing and stage_name not in stages_to_run:
                    prereq_stages.append(stage_name)
            stages_to_run = prereq_stages + [
                s for s in stages_to_run if s not in prereq_stages
            ]
            logger.info("Expanded stages_to_run: %s", stages_to_run)
    else:
        try:
            n_classes = _load_n_classes(preprocessed_dir, args.dataset)
        except FileNotFoundError:
            logger.info("n_classes not yet available (preprocessing will determine it)")
        if n_classes is not None:
            logger.info("No training stages requested. n_classes=%d", n_classes)
        else:
            logger.info("No training stages requested.")

    # =================================================================
    # STAGE 4: Hyperparameter tuning (optional)
    # =================================================================
    if "tuning" in stages_to_run:
        if args.tune:
            _ensure_splits_loaded()
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
                        if key == "learning_rate":
                            config.model.learning_rate = best_cfg[key]
                        else:
                            setattr(config.model, key, best_cfg[key])

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
            _ensure_splits_loaded()
            pm.start_stage("baselines")
            logger.info("━━━ Stage 5/9: Baseline models ━━━")

            # Check if models already exist (resume after partial run)
            from src.utils.serialization import load_baseline_model

            existing_models = []
            for name in ["random_forest", "svm", "logistic_regression"]:
                if (baselines_dir / f"{name}.pkl").exists():
                    existing_models.append(name)

            X_bl = None  # set below for metadata

            if len(existing_models) == 3:
                logger.info("All 3 baseline models exist — skipping training.")
                fitted = {n: load_baseline_model(n, baselines_dir) for n in existing_models}
            else:
                # Auto-sampling: cap at 200K rows for baselines (Req 5-6)
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

                fitted = train_all_baselines(X_bl, y_bl, config.raw.get("baselines", {}))
                baselines_dir.mkdir(parents=True, exist_ok=True)
                save_all_baselines(fitted, baselines_dir)

            # Evaluate each baseline on test set
            # Load test data on-demand for baseline evaluation
            logger.info("Loading test split for baseline evaluation ...")
            X_test, y_test = load_split_data_test(preprocessed_dir)

            results: Dict[str, Any] = {}
            for name in ["random_forest", "svm", "logistic_regression"]:
                try:
                    logger.info("Evaluating %s on test set (%s samples) ...", name, f"{X_test.shape[0]:,}")
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
                "training_samples": int(X_bl.shape[0]) if X_bl is not None else 200000,
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
        _ensure_splits_loaded()
        pm.start_stage("lstm_train")
        logger.info("━━━ Stage 6/9: LSTM training ━━━")

        # run_full_training() handles checkpoint resume internally via
        # train_lstm() which loads weights from config.training.model_checkpoint.filepath
        run_full_training(
            X_train, X_val,
            y_train, y_val,
            n_classes=n_classes,
            dataset=args.dataset,
            config=config,
            output_dir=str(dirs["root"]),
            resume=args.resume,
        )

        rm.stage_complete("lstm_train", metadata={
            "epochs": epochs,
            "resumed": args.resume,
        })
        pm.end_stage("lstm_train")
        gc.collect()
        logger.info("LSTM training done.")

    # =================================================================
    # STAGE 7: Evaluation
    # =================================================================
    if "evaluation" in stages_to_run:
        _ensure_splits_loaded()
        pm.start_stage("evaluation")
        logger.info("━━━ Stage 7/9: Evaluation ━━━")

        # Load test data on-demand for evaluation
        X_test, y_test = load_split_data_test(preprocessed_dir)

        # Load trained LSTM
        from src.utils.serialization import load_keras_model as load_trained_model

        keras_path = final_dir / "lstm_ids_model.keras"
        h5_path = final_dir / "lstm_ids_model.h5"
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
            output_path=tables_dir / "confusion_matrix.png",
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
        _ensure_splits_loaded()
        pm.start_stage("visualization")
        logger.info("━━━ Stage 8/9: Visualization ━━━")

        figures_dir.mkdir(parents=True, exist_ok=True)

        # --- 8a. Training history curves ---
        history_csv = dirs["root"] / "reports" / "logs" / "training_history.csv"
        if not history_csv.exists():
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

        keras_path = final_dir / "lstm_ids_model.keras"
        h5_path = final_dir / "lstm_ids_model.h5"
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
        _ensure_splits_loaded()
        pm.start_stage("export")
        logger.info("━━━ Stage 9/9: Export ━━━")

        exported_dir = dirs["exported"]
        exported_dir.mkdir(parents=True, exist_ok=True)

        # Copy final model
        import shutil

        for fname in ["lstm_ids_model.keras", "lstm_ids_model.h5"]:
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
            export_chapter4_zip(
                figures_dir=figures_dir,
                tables_dir=tables_dir,
                output_dir=exported_dir,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chapter 4 ZIP export failed: %s", exc)

        rm.stage_complete("export")
        pm.end_stage("export")
        gc.collect()
        logger.info("Export done.")

    # ── Summary ─────────────────────────────────────────────────────
    _keepalive_stop.set()
    logger.info("=" * 60)
    logger.info("Pipeline completed for dataset: %s", args.dataset)
    logger.info("Output directory: %s", dirs["root"])
    logger.info("=" * 60)

    # Print progress summary
    pm.print_summary()

    # ── Zip outputs to Drive ──────────────────────────────────────
    if args.zip_drive:
        import shutil
        import zipfile

        drive_root = Path("/content/drive/MyDrive/lstm_ids_results")
        drive_root.mkdir(parents=True, exist_ok=True)
        zip_path = drive_root / f"{args.dataset}.zip"
        src_dir = dirs["root"]

        logger.info("Zipping outputs to: %s", zip_path)
        logger.info("Source: %s", src_dir)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            file_count = 0
            for fpath in sorted(src_dir.rglob("*")):
                if fpath.is_file():
                    arcname = fpath.relative_to(src_dir.parent)
                    zf.write(fpath, arcname)
                    file_count += 1
                    if file_count % 50 == 0:
                        logger.info("  zipped %d files ...", file_count)

        zip_size_gb = zip_path.stat().st_size / (1024 ** 3)
        logger.info("ZIP saved: %s (%.2f GB, %d files)", zip_path, zip_size_gb, file_count)

        # Free local disk
        shutil.rmtree(src_dir)
        logger.info("Local output removed: %s", src_dir)
        gc.collect()


# ─── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
