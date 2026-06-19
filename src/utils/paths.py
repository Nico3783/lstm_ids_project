
# src/utils/paths.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Centralised path management for the entire project.
#          Resolves every directory and file path relative to
#          the project root, creates directories on demand,
#          and exposes convenience accessors used by all other
#          modules.  No other module should hard-code a path
#          string — they import from here instead.

import os
from pathlib import Path
from typing import Optional

from src.utils.constants import (
    # Data file names
    MERGED_DATASET_CSV,
    CLEANED_DATASET_CSV,
    ENCODED_DATASET_CSV,
    SCALED_DATASET_CSV,
    X_TRAIN_NPY, X_VAL_NPY, X_TEST_NPY,
    Y_TRAIN_NPY, Y_VAL_NPY, Y_TEST_NPY,
    FEATURE_NAMES_PKL,
    LABEL_ENCODER_PKL,
    SCALER_PKL,
    METADATA_JSON,
    # Model file names
    BEST_MODEL_KERAS,
    BEST_MODEL_H5,
    FINAL_MODEL_KERAS,
    FINAL_MODEL_H5,
    MODEL_METADATA_JSON,
    RANDOM_FOREST_PKL,
    SVM_PKL,
    LOGISTIC_REGRESSION_PKL,
    BASELINE_RESULTS_JSON,
    # Report file names
    FIG_CLASS_DISTRIBUTION,
    FIG_CORRELATION_HEATMAP,
    FIG_PREPROCESSING_PIPELINE,
    FIG_LSTM_ARCHITECTURE,
    FIG_TRAINING_ACCURACY,
    FIG_TRAINING_LOSS,
    FIG_CONFUSION_MATRIX,
    FIG_ROC_CURVE,
    FIG_MODEL_COMPARISON,
    FIG_PRECISION_RECALL,
    FIG_FEATURE_IMPORTANCE,
    TABLE_DATASET_SUMMARY,
    TABLE_HYPERPARAMETERS,
    TABLE_BASELINE_METRICS,
    TABLE_FINAL_METRICS,
    METRICS_CLASSIFICATION_REPORT,
    METRICS_EVALUATION_RESULTS,
    METRICS_ROC_AUC_SCORES,
    LOG_TRAINING,
    LOG_TRAINING_HISTORY,
    LOG_PIPELINE,
    OUT_TEST_PREDICTIONS,
    OUT_NEW_PREDICTIONS,
)


# Project Root Resolution

def get_project_root() -> Path:
    """
    Return the absolute path to the project root directory.

    The root is determined as the directory that contains
    ``config.yaml``.  The search starts from this file's
    location (``src/utils/``) and walks upward until a
    directory containing ``config.yaml`` is found.

    Returns
    -------
    Path
        Absolute project root path.

    Raises
    ------
    FileNotFoundError
        If ``config.yaml`` cannot be located within 6 levels
        of the current file.
    """
    current = Path(__file__).resolve().parent
    for _ in range(6):
        if (current / "config.yaml").exists():
            return current
        current = current.parent
    raise FileNotFoundError(
        "Could not locate project root (config.yaml not found). "
        "Ensure you are running from within the lstm_ids_project/ directory."
    )


# Resolve root once at import time — all path constants below
# are absolute paths derived from this root.
PROJECT_ROOT: Path = get_project_root()


# Directory Paths

# --- Data directories ---
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
SAMPLE_DATA_DIR: Path = DATA_DIR / "sample"

# Dataset-specific raw directories
NSL_KDD_RAW_DIR: Path = RAW_DATA_DIR / "nsl_kdd"
CICIDS2017_RAW_DIR: Path = RAW_DATA_DIR / "cicids2017"
UNSW_NB15_RAW_DIR: Path = RAW_DATA_DIR / "unsw_nb15"

# --- Model directories ---
MODELS_DIR: Path = PROJECT_ROOT / "models"
CHECKPOINTS_DIR: Path = MODELS_DIR / "checkpoints"
FINAL_MODEL_DIR: Path = MODELS_DIR / "final"
BASELINES_DIR: Path = MODELS_DIR / "baselines"

# --- Report directories ---
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
TABLES_DIR: Path = REPORTS_DIR / "tables"
METRICS_DIR: Path = REPORTS_DIR / "metrics"
LOGS_DIR: Path = REPORTS_DIR / "logs"

# --- Output directories ---
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR: Path = OUTPUTS_DIR / "predictions"
EXPORTED_DIR: Path = OUTPUTS_DIR / "exported"

# --- Documentation directories ---
DOCS_DIR: Path = PROJECT_ROOT / "docs"
ARCHITECTURE_DOCS_DIR: Path = DOCS_DIR / "architecture"
METHODOLOGY_DOCS_DIR: Path = DOCS_DIR / "methodology"
SCREENSHOTS_DIR: Path = DOCS_DIR / "screenshots"

# --- Notebook directory ---
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

# --- Source directory ---
SRC_DIR: Path = PROJECT_ROOT / "src"

# --- Test directory ---
TESTS_DIR: Path = PROJECT_ROOT / "tests"

# --- Scripts directory ---
SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts"

# TensorBoard log directory
TENSORBOARD_LOG_DIR: Path = LOGS_DIR / "tensorboard"


# File Paths — Raw Dataset Files

# NSL-KDD
NSL_KDD_TRAIN_FILE: Path = NSL_KDD_RAW_DIR / "KDDTrain+.txt"
NSL_KDD_TEST_FILE: Path = NSL_KDD_RAW_DIR / "KDDTest+.txt"
NSL_KDD_TRAIN_20PCT_FILE: Path = NSL_KDD_RAW_DIR / "KDDTrain+_20Percent.txt"
NSL_KDD_FIELD_NAMES_FILE: Path = NSL_KDD_RAW_DIR / "field_names.csv"

# UNSW-NB15
UNSW_NB15_TRAIN_FILE: Path = UNSW_NB15_RAW_DIR / "UNSW_NB15_training-set.csv"
UNSW_NB15_TEST_FILE: Path = UNSW_NB15_RAW_DIR / "UNSW_NB15_testing-set.csv"
UNSW_NB15_FEATURES_FILE: Path = UNSW_NB15_RAW_DIR / "UNSW-NB15_features.csv"
UNSW_NB15_GT_FILE: Path = UNSW_NB15_RAW_DIR / "UNSW-NB15_GT.csv"

# Sample data
SAMPLE_INPUT_FILE: Path = SAMPLE_DATA_DIR / "sample_input.csv"
SAMPLE_PREDICTIONS_FILE: Path = SAMPLE_DATA_DIR / "sample_predictions.csv"


# File Paths — Interim Data

MERGED_DATASET_PATH: Path = INTERIM_DATA_DIR / MERGED_DATASET_CSV
CLEANED_DATASET_PATH: Path = INTERIM_DATA_DIR / CLEANED_DATASET_CSV
ENCODED_DATASET_PATH: Path = INTERIM_DATA_DIR / ENCODED_DATASET_CSV
SCALED_DATASET_PATH: Path = INTERIM_DATA_DIR / SCALED_DATASET_CSV


# File Paths — Processed Arrays and Artifacts

X_TRAIN_PATH: Path = PROCESSED_DATA_DIR / X_TRAIN_NPY
X_VAL_PATH: Path = PROCESSED_DATA_DIR / X_VAL_NPY
X_TEST_PATH: Path = PROCESSED_DATA_DIR / X_TEST_NPY
Y_TRAIN_PATH: Path = PROCESSED_DATA_DIR / Y_TRAIN_NPY
Y_VAL_PATH: Path = PROCESSED_DATA_DIR / Y_VAL_NPY
Y_TEST_PATH: Path = PROCESSED_DATA_DIR / Y_TEST_NPY
FEATURE_NAMES_PATH: Path = PROCESSED_DATA_DIR / FEATURE_NAMES_PKL
LABEL_ENCODER_PROCESSED_PATH: Path = PROCESSED_DATA_DIR / LABEL_ENCODER_PKL
SCALER_PROCESSED_PATH: Path = PROCESSED_DATA_DIR / SCALER_PKL
METADATA_PATH: Path = PROCESSED_DATA_DIR / METADATA_JSON


# File Paths — Models

BEST_MODEL_KERAS_PATH: Path = CHECKPOINTS_DIR / BEST_MODEL_KERAS
BEST_MODEL_H5_PATH: Path = CHECKPOINTS_DIR / BEST_MODEL_H5
FINAL_MODEL_KERAS_PATH: Path = FINAL_MODEL_DIR / FINAL_MODEL_KERAS
FINAL_MODEL_H5_PATH: Path = FINAL_MODEL_DIR / FINAL_MODEL_H5
MODEL_METADATA_PATH: Path = FINAL_MODEL_DIR / MODEL_METADATA_JSON
LABEL_ENCODER_FINAL_PATH: Path = FINAL_MODEL_DIR / LABEL_ENCODER_PKL
SCALER_FINAL_PATH: Path = FINAL_MODEL_DIR / SCALER_PKL
FEATURE_NAMES_FINAL_PATH: Path = FINAL_MODEL_DIR / FEATURE_NAMES_PKL

# Baseline models
RANDOM_FOREST_PATH: Path = BASELINES_DIR / RANDOM_FOREST_PKL
SVM_PATH: Path = BASELINES_DIR / SVM_PKL
LOGISTIC_REGRESSION_PATH: Path = BASELINES_DIR / LOGISTIC_REGRESSION_PKL
BASELINE_RESULTS_PATH: Path = BASELINES_DIR / BASELINE_RESULTS_JSON


# File Paths — Report Figures

FIG_CLASS_DISTRIBUTION_PATH: Path = FIGURES_DIR / FIG_CLASS_DISTRIBUTION
FIG_CORRELATION_HEATMAP_PATH: Path = FIGURES_DIR / FIG_CORRELATION_HEATMAP
FIG_PREPROCESSING_PIPELINE_PATH: Path = FIGURES_DIR / FIG_PREPROCESSING_PIPELINE
FIG_LSTM_ARCHITECTURE_PATH: Path = FIGURES_DIR / FIG_LSTM_ARCHITECTURE
FIG_TRAINING_ACCURACY_PATH: Path = FIGURES_DIR / FIG_TRAINING_ACCURACY
FIG_TRAINING_LOSS_PATH: Path = FIGURES_DIR / FIG_TRAINING_LOSS
FIG_CONFUSION_MATRIX_PATH: Path = FIGURES_DIR / FIG_CONFUSION_MATRIX
FIG_ROC_CURVE_PATH: Path = FIGURES_DIR / FIG_ROC_CURVE
FIG_MODEL_COMPARISON_PATH: Path = FIGURES_DIR / FIG_MODEL_COMPARISON
FIG_PRECISION_RECALL_PATH: Path = FIGURES_DIR / FIG_PRECISION_RECALL
FIG_FEATURE_IMPORTANCE_PATH: Path = FIGURES_DIR / FIG_FEATURE_IMPORTANCE


# File Paths — Report Tables

TABLE_DATASET_SUMMARY_PATH: Path = TABLES_DIR / TABLE_DATASET_SUMMARY
TABLE_HYPERPARAMETERS_PATH: Path = TABLES_DIR / TABLE_HYPERPARAMETERS
TABLE_BASELINE_METRICS_PATH: Path = TABLES_DIR / TABLE_BASELINE_METRICS
TABLE_FINAL_METRICS_PATH: Path = TABLES_DIR / TABLE_FINAL_METRICS


# File Paths — Report Metrics

METRICS_CLASSIFICATION_REPORT_PATH: Path = (
    METRICS_DIR / METRICS_CLASSIFICATION_REPORT
)
METRICS_EVALUATION_RESULTS_PATH: Path = METRICS_DIR / METRICS_EVALUATION_RESULTS
METRICS_ROC_AUC_SCORES_PATH: Path = METRICS_DIR / METRICS_ROC_AUC_SCORES


# File Paths — Logs

TRAINING_LOG_PATH: Path = LOGS_DIR / LOG_TRAINING
TRAINING_HISTORY_CSV_PATH: Path = LOGS_DIR / LOG_TRAINING_HISTORY
PIPELINE_LOG_PATH: Path = LOGS_DIR / LOG_PIPELINE


# File Paths — Outputs

TEST_PREDICTIONS_PATH: Path = PREDICTIONS_DIR / OUT_TEST_PREDICTIONS
NEW_PREDICTIONS_PATH: Path = PREDICTIONS_DIR / OUT_NEW_PREDICTIONS

# Exported ZIP archives for Chapter 4
CHAPTER4_FIGURES_ZIP: Path = EXPORTED_DIR / "chapter4_figures.zip"
CHAPTER4_TABLES_ZIP: Path = EXPORTED_DIR / "chapter4_tables.zip"
COMPLETE_RESULTS_ZIP: Path = EXPORTED_DIR / "complete_results.zip"

# Config file
CONFIG_FILE: Path = PROJECT_ROOT / "config.yaml"


# Directory Creation Utilities

# All directories that must exist before the pipeline runs.
ALL_DIRECTORIES = [
    DATA_DIR,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    SAMPLE_DATA_DIR,
    NSL_KDD_RAW_DIR,
    CICIDS2017_RAW_DIR,
    UNSW_NB15_RAW_DIR,
    MODELS_DIR,
    CHECKPOINTS_DIR,
    FINAL_MODEL_DIR,
    BASELINES_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    METRICS_DIR,
    LOGS_DIR,
    TENSORBOARD_LOG_DIR,
    OUTPUTS_DIR,
    PREDICTIONS_DIR,
    EXPORTED_DIR,
    DOCS_DIR,
    ARCHITECTURE_DOCS_DIR,
    METHODOLOGY_DOCS_DIR,
    SCREENSHOTS_DIR,
    NOTEBOOKS_DIR,
    TESTS_DIR,
    SCRIPTS_DIR,
]


def create_project_directories() -> None:
    """
    Create every project directory listed in ``ALL_DIRECTORIES``
    if it does not already exist.

    This function is called once at the start of
    ``run_pipeline.py`` and ``setup.sh`` to ensure the full
    directory tree is in place before any file I/O occurs.

    Returns
    -------
    None
    """
    for directory in ALL_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path) -> Path:
    """
    Ensure that the parent directory of *path* exists,
    creating it recursively if necessary.

    Parameters
    ----------
    path : Path
        File or directory path whose parent must exist.

    Returns
    -------
    Path
        The same *path* that was passed in, unchanged.

    Examples
    --------
    >>> from src.utils.paths import ensure_dir, FIGURES_DIR
    >>> output_path = ensure_dir(FIGURES_DIR / "my_plot.png")
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_raw_data_dir(dataset: str) -> Path:
    """
    Return the raw data directory for the specified dataset.

    Parameters
    ----------
    dataset : str
        Dataset identifier — one of ``nsl_kdd``, ``cicids2017``,
        ``unsw_nb15``.

    Returns
    -------
    Path
        Absolute path to the dataset's raw directory.

    Raises
    ------
    ValueError
        If *dataset* is not a recognised identifier.
    """
    mapping = {
        "nsl_kdd": NSL_KDD_RAW_DIR,
        "cicids2017": CICIDS2017_RAW_DIR,
        "unsw_nb15": UNSW_NB15_RAW_DIR,
    }
    if dataset not in mapping:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Choose from: {list(mapping.keys())}"
        )
    return mapping[dataset]


def get_checkpoint_path(epoch: Optional[int] = None) -> Path:
    """
    Return the path for a training checkpoint file.

    Parameters
    ----------
    epoch : int, optional
        Epoch number to embed in the filename.  When *None*
        returns the best-model path used by ModelCheckpoint.

    Returns
    -------
    Path
        Checkpoint file path.

    Examples
    --------
    >>> get_checkpoint_path()
    PosixPath('.../models/checkpoints/best_model.keras')
    >>> get_checkpoint_path(5)
    PosixPath('.../models/checkpoints/checkpoint_epoch_05.keras')
    """
    if epoch is None:
        return BEST_MODEL_KERAS_PATH
    return CHECKPOINTS_DIR / f"checkpoint_epoch_{epoch:02d}.keras"


def path_exists(path: Path) -> bool:
    """
    Return True if *path* exists on disk, False otherwise.

    Parameters
    ----------
    path : Path
        Path to check.

    Returns
    -------
    bool
    """
    return path.exists()


def assert_file_exists(path: Path, description: str = "") -> None:
    """
    Raise ``FileNotFoundError`` if *path* does not exist.

    Used to validate dataset files before processing begins,
    providing a clear error message rather than a cryptic
    downstream exception.

    Parameters
    ----------
    path : Path
        File path to validate.
    description : str, optional
        Human-readable description of the file, included in
        the error message to aid debugging.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    if not path.exists():
        label = f" ({description})" if description else ""
        raise FileNotFoundError(
            f"Required file not found{label}: {path}\n"
            "Please ensure the dataset has been downloaded and placed "
            "in the correct directory. Run:\n"
            "  python -m src.data.download --dataset <name>\n"
            "or follow the manual instructions in README.md."
        )


def get_all_paths_summary() -> dict:
    """
    Return a dictionary mapping descriptive labels to their
    absolute path strings.  Used by ``run_pipeline.py`` to
    log the active path configuration at startup.

    Returns
    -------
    dict
        {label: str(path)} for all key project paths.
    """
    return {
        "Project root": str(PROJECT_ROOT),
        "Raw data": str(RAW_DATA_DIR),
        "Processed data": str(PROCESSED_DATA_DIR),
        "Models (checkpoints)": str(CHECKPOINTS_DIR),
        "Models (final)": str(FINAL_MODEL_DIR),
        "Models (baselines)": str(BASELINES_DIR),
        "Report figures": str(FIGURES_DIR),
        "Report tables": str(TABLES_DIR),
        "Report metrics": str(METRICS_DIR),
        "Logs": str(LOGS_DIR),
        "Predictions": str(PREDICTIONS_DIR),
        "Config file": str(CONFIG_FILE),
    }