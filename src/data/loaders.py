
# src/data/loaders.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Raw dataset loaders for NSL-KDD, CICIDS2017, and
#          UNSW-NB15.  Each loader reads the raw CSV/TXT files
#          from disk, assigns correct column names, normalises
#          column headers, handles common file quirks, merges
#          train and test splits where appropriate, and
#          returns a clean pandas DataFrame ready for the
#          validation and preprocessing stages.
#
#          Design principles:
#          - No preprocessing transformations here — loaders
#            only read, rename, and merge.
#          - Every loader validates the loaded DataFrame using
#            src.data.validators before returning it.
#          - All file paths come from src.utils.paths —
#            no hard-coded strings anywhere.
#
#          Aligned with Chapter 3, Section 3.5.1 and 3.5.2.

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.validators import (
    validate_dataframe,
    ValidationReport,
)
from src.utils.constants import (
    NSL_KDD_COLUMNS,
    NSL_KDD_TARGET_COLUMN,
    NSL_KDD_DIFFICULTY_COLUMN,
    NSL_KDD_ATTACK_TO_CATEGORY,
    NSL_KDD_CATEGORY_TO_INT,
    CICIDS2017_TARGET_COLUMN,
    CICIDS2017_BENIGN_LABEL,
    UNSW_NB15_TARGET_COLUMN,
    UNSW_NB15_BINARY_LABEL_COLUMN,
    SUPPORTED_DATASETS,
)
from src.utils.logger import get_logger
from src.utils.paths import (
    NSL_KDD_TRAIN_FILE,
    NSL_KDD_TEST_FILE,
    NSL_KDD_TRAIN_20PCT_FILE,
    NSL_KDD_FIELD_NAMES_FILE,
    CICIDS2017_RAW_DIR,
    UNSW_NB15_TRAIN_FILE,
    UNSW_NB15_TEST_FILE,
    assert_file_exists,
)

logger = get_logger(__name__)


# NSL-KDD Loader

def load_nsl_kdd(
    train_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
    use_20pct_train: bool = False,
    merge: bool = True,
    validate: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the NSL-KDD dataset from raw TXT files.

    NSL-KDD files are headerless CSVs with 43 columns:
    41 network traffic features, 1 attack-type label, and
    1 difficulty score.  Column names are assigned from the
    ``NSL_KDD_COLUMNS`` constant (Chapter 3, Section 3.3.1).

    The attack-type label is a raw string (e.g. ``"neptune"``,
    ``"portsweep"``).  This loader preserves the raw label —
    the preprocessing module maps it to a 5-class integer.

    Parameters
    ----------
    train_path : Path, optional
        Path to the training file.  Defaults to
        ``data/raw/nsl_kdd/KDDTrain+.txt``.
    test_path : Path, optional
        Path to the test file.  Defaults to
        ``data/raw/nsl_kdd/KDDTest+.txt``.
    use_20pct_train : bool
        Use the 20 % training subset instead of the full
        training set.  Useful for rapid prototyping.
    merge : bool
        If True, return (full_df, test_df) where full_df is
        the concatenation of train and test sets (used for
        full preprocessing before re-splitting).
        If False, return (train_df, test_df) separately.
    validate : bool
        Run validation checks after loading.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (train_or_merged_df, test_df).
        When *merge* is True the first element is the merged
        full dataset; the second is the original test split
        retained for reference.

    Raises
    ------
    FileNotFoundError
        If either dataset file is missing.
    """
    # Resolve paths
    if use_20pct_train:
        train_file = train_path or NSL_KDD_TRAIN_20PCT_FILE
        logger.info("Using 20%% NSL-KDD training subset.")
    else:
        train_file = train_path or NSL_KDD_TRAIN_FILE

    test_file = test_path or NSL_KDD_TEST_FILE

    assert_file_exists(train_file, "NSL-KDD training file")
    assert_file_exists(test_file, "NSL-KDD test file")

    logger.info("Loading NSL-KDD training data: %s", train_file)
    train_df = _read_nsl_kdd_file(train_file)
    logger.info(
        "NSL-KDD train loaded: %d rows × %d cols.",
        len(train_df), len(train_df.columns),
    )

    logger.info("Loading NSL-KDD test data: %s", test_file)
    test_df = _read_nsl_kdd_file(test_file)
    logger.info(
        "NSL-KDD test loaded: %d rows × %d cols.",
        len(test_df), len(test_df.columns),
    )

    # Ensure field_names.csv exists (generate if absent)
    _ensure_nsl_kdd_field_names()

    if validate:
        logger.info("Running validation on NSL-KDD splits ...")
        validate_dataframe(train_df, "nsl_kdd", split="train")
        validate_dataframe(test_df, "nsl_kdd", split="test")

    if merge:
        # Tag original split for traceability
        train_df["_split"] = "train"
        test_df["_split"]  = "test"
        merged = pd.concat(
            [train_df, test_df], axis=0, ignore_index=True
        )
        logger.info(
            "NSL-KDD merged dataset: %d rows × %d cols.",
            len(merged), len(merged.columns),
        )
        return merged, test_df
    else:
        return train_df, test_df


def _read_nsl_kdd_file(path: Path) -> pd.DataFrame:
    """
    Read a single NSL-KDD TXT file into a DataFrame.

    NSL-KDD files have no header row.  Column names come from
    ``NSL_KDD_COLUMNS`` which lists all 43 columns in order.

    Parameters
    ----------
    path : Path
        Path to the NSL-KDD TXT file.

    Returns
    -------
    pd.DataFrame
        43-column DataFrame with correct dtypes.
    """
    try:
        df = pd.read_csv(
            path,
            header=None,
            names=NSL_KDD_COLUMNS,
            dtype=str,           # Read all as string first
            na_values=["", " ", "NA", "NaN", "null"],
            low_memory=False,
        )
    except Exception as exc:
        raise IOError(
            f"Failed to read NSL-KDD file '{path}': {exc}"
        ) from exc

    # Convert numeric columns — errors become NaN for later
    # imputation rather than crashing the loader
    numeric_cols = [
        c for c in NSL_KDD_COLUMNS
        if c not in [
            "protocol_type", "service", "flag",
            NSL_KDD_TARGET_COLUMN, NSL_KDD_DIFFICULTY_COLUMN,
        ]
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalise label strings — lowercase and strip whitespace
    if NSL_KDD_TARGET_COLUMN in df.columns:
        df[NSL_KDD_TARGET_COLUMN] = (
            df[NSL_KDD_TARGET_COLUMN]
            .astype(str)
            .str.lower()
            .str.strip()
            .str.rstrip(".")   # Some files have trailing dot
        )

    # Normalise categorical feature values
    for col in ["protocol_type", "service", "flag"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.lower().str.strip()
            )

    # Convert difficulty to numeric
    if NSL_KDD_DIFFICULTY_COLUMN in df.columns:
        df[NSL_KDD_DIFFICULTY_COLUMN] = pd.to_numeric(
            df[NSL_KDD_DIFFICULTY_COLUMN], errors="coerce"
        )

    return df


def _ensure_nsl_kdd_field_names() -> None:
    """
    Write ``field_names.csv`` if it does not already exist.
    This file is used by notebooks and EDA scripts.
    """
    from src.utils.constants import NSL_KDD_COLUMNS
    from src.data.download import NSL_KDD_FIELD_NAMES_CONTENT

    if not NSL_KDD_FIELD_NAMES_FILE.exists():
        NSL_KDD_FIELD_NAMES_FILE.parent.mkdir(
            parents=True, exist_ok=True
        )
        NSL_KDD_FIELD_NAMES_FILE.write_text(
            NSL_KDD_FIELD_NAMES_CONTENT, encoding="utf-8"
        )
        logger.info(
            "field_names.csv auto-generated: %s",
            NSL_KDD_FIELD_NAMES_FILE,
        )


def get_nsl_kdd_summary(df: pd.DataFrame) -> Dict:
    """
    Return a summary statistics dictionary for a loaded
    NSL-KDD DataFrame.  Populates the Dataset Summary table
    required for Chapter 4.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded NSL-KDD DataFrame (before preprocessing).

    Returns
    -------
    dict
        Summary statistics including shape, class distribution,
        missing values, and categorical cardinalities.
    """
    label_dist = (
        df[NSL_KDD_TARGET_COLUMN].value_counts().to_dict()
        if NSL_KDD_TARGET_COLUMN in df.columns
        else {}
    )
    missing_pct = float(
        df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
    )
    return {
        "dataset": "NSL-KDD",
        "n_samples": len(df),
        "n_features_raw": 41,
        "n_classes": 5,
        "attack_type_distribution": label_dist,
        "n_unique_labels": len(label_dist),
        "missing_value_pct": round(missing_pct, 4),
        "protocol_type_cardinality": (
            df["protocol_type"].nunique()
            if "protocol_type" in df.columns
            else None
        ),
        "service_cardinality": (
            df["service"].nunique()
            if "service" in df.columns
            else None
        ),
        "flag_cardinality": (
            df["flag"].nunique()
            if "flag" in df.columns
            else None
        ),
    }


# CICIDS2017 Loader

def load_cicids2017(
    data_dir: Optional[Path] = None,
    files: Optional[List[str]] = None,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Load and merge all CICIDS2017 CSV files into a single
    DataFrame.

    CICIDS2017 consists of 8 daily CSV files, each containing
    80 bidirectional flow features plus a ``Label`` column.
    All 8 files are concatenated into a single DataFrame.

    Column names are stripped of leading/trailing whitespace
    (a known quirk of CICIDS2017 files — the label column is
    named ``" Label"`` with a leading space).

    Parameters
    ----------
    data_dir : Path, optional
        Directory containing the CSV files.  Defaults to
        ``data/raw/cicids2017/``.
    files : list of str, optional
        Specific filenames to load.  Defaults to all 8 daily
        files defined in the project config.
    validate : bool
        Run validation checks after loading.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with all flow records.

    Raises
    ------
    FileNotFoundError
        If *data_dir* does not exist.
    ValueError
        If no files could be loaded successfully.
    """
    from src.config import get_config

    data_dir = data_dir or CICIDS2017_RAW_DIR

    if not data_dir.exists():
        raise FileNotFoundError(
            f"CICIDS2017 data directory not found: {data_dir}\n"
            "Run: python -m src.data.download --dataset cicids2017"
        )

    if files is None:
        cfg = get_config()
        files = cfg.raw.get("dataset", {}).get(
            "cicids2017", {}
        ).get("files", _DEFAULT_CICIDS2017_FILES)

    logger.info(
        "Loading CICIDS2017 from %s (%d files) ...",
        data_dir, len(files),
    )

    frames: List[pd.DataFrame] = []
    for fname in files:
        fpath = data_dir / fname
        if not fpath.exists():
            logger.warning(
                "  [SKIP] File not found: %s", fpath
            )
            continue
        try:
            df_part = _read_cicids2017_file(fpath)
            frames.append(df_part)
            logger.info(
                "  [OK]   %s — %d rows loaded.",
                fname, len(df_part),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "  [FAIL] Could not read %s: %s", fname, exc
            )

    if not frames:
        raise ValueError(
            "No CICIDS2017 files could be loaded. "
            "Check that CSV files are present in:\n"
            f"  {data_dir}"
        )

    merged = pd.concat(frames, axis=0, ignore_index=True)
    logger.info(
        "CICIDS2017 merged: %d rows × %d cols from %d files.",
        len(merged), len(merged.columns), len(frames),
    )

    if validate:
        validate_dataframe(merged, "cicids2017", split="full")

    return merged


def _read_cicids2017_file(path: Path) -> pd.DataFrame:
    """
    Read a single CICIDS2017 CSV file.

    Strips whitespace from column names and the Label column
    values, replaces Inf values with NaN (handled in
    preprocessing), and coerces numeric columns.

    Parameters
    ----------
    path : Path

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(
        path,
        encoding="utf-8",
        low_memory=False,
        na_values=["", " ", "NA", "nan", "NaN", "Infinity",
                   "infinity", "-Infinity"],
    )

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Replace inf / -inf values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Normalise the label column
    target = CICIDS2017_TARGET_COLUMN.strip()
    if target in df.columns:
        df[target] = (
            df[target].astype(str).str.strip()
        )

    # Coerce all non-label columns to numeric where possible
    non_label_cols = [c for c in df.columns if c != target]
    for col in non_label_cols:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


_DEFAULT_CICIDS2017_FILES: List[str] = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv",
]


def get_cicids2017_summary(df: pd.DataFrame) -> Dict:
    """
    Return summary statistics for a loaded CICIDS2017
    DataFrame for the Chapter 4 Dataset Summary table.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict
    """
    target = CICIDS2017_TARGET_COLUMN.strip()
    label_dist = (
        df[target].value_counts().to_dict()
        if target in df.columns
        else {}
    )
    missing_pct = float(
        df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
    )
    inf_count = int(
        df.select_dtypes(include=[np.number])
        .isin([np.inf, -np.inf])
        .sum()
        .sum()
    )
    return {
        "dataset": "CICIDS2017",
        "n_samples": len(df),
        "n_features_raw": 80,
        "n_classes_raw": len(label_dist),
        "label_distribution": label_dist,
        "missing_value_pct": round(missing_pct, 4),
        "infinite_value_count": inf_count,
    }


# UNSW-NB15 Loader

def load_unsw_nb15(
    train_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
    merge: bool = True,
    validate: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the UNSW-NB15 dataset from CSV files.

    UNSW-NB15 has a header row.  The target column is
    ``attack_cat`` (9 attack families + Normal) and a binary
    ``label`` column (0 = normal, 1 = attack).

    Parameters
    ----------
    train_path : Path, optional
        Path to the training CSV.  Defaults to
        ``data/raw/unsw_nb15/UNSW_NB15_training-set.csv``.
    test_path : Path, optional
        Path to the test CSV.  Defaults to
        ``data/raw/unsw_nb15/UNSW_NB15_testing-set.csv``.
    merge : bool
        If True, return (merged_df, test_df).
    validate : bool
        Run validation checks after loading.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)

    Raises
    ------
    FileNotFoundError
        If either file is missing.
    """
    train_file = train_path or UNSW_NB15_TRAIN_FILE
    test_file  = test_path  or UNSW_NB15_TEST_FILE

    assert_file_exists(train_file, "UNSW-NB15 training file")
    assert_file_exists(test_file,  "UNSW-NB15 test file")

    logger.info("Loading UNSW-NB15 training data: %s", train_file)
    train_df = _read_unsw_nb15_file(train_file)
    logger.info(
        "UNSW-NB15 train loaded: %d rows × %d cols.",
        len(train_df), len(train_df.columns),
    )

    logger.info("Loading UNSW-NB15 test data: %s", test_file)
    test_df = _read_unsw_nb15_file(test_file)
    logger.info(
        "UNSW-NB15 test loaded: %d rows × %d cols.",
        len(test_df), len(test_df.columns),
    )

    if validate:
        validate_dataframe(train_df, "unsw_nb15", split="train")
        validate_dataframe(test_df,  "unsw_nb15", split="test")

    if merge:
        train_df["_split"] = "train"
        test_df["_split"]  = "test"
        merged = pd.concat(
            [train_df, test_df], axis=0, ignore_index=True
        )
        logger.info(
            "UNSW-NB15 merged: %d rows × %d cols.",
            len(merged), len(merged.columns),
        )
        return merged, test_df
    else:
        return train_df, test_df


def _read_unsw_nb15_file(path: Path) -> pd.DataFrame:
    """
    Read a single UNSW-NB15 CSV file.

    Parameters
    ----------
    path : Path

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(
        path,
        encoding="utf-8",
        low_memory=False,
        na_values=["", " ", "NA", "nan", "NaN", "-"],
    )

    # Normalise column names: lowercase, strip spaces
    df.columns = (
        df.columns.str.lower().str.strip().str.replace(" ", "_")
    )

    # Replace inf values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Normalise attack_cat label
    if UNSW_NB15_TARGET_COLUMN in df.columns:
        df[UNSW_NB15_TARGET_COLUMN] = (
            df[UNSW_NB15_TARGET_COLUMN]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        # Standardise "normal" representations
        df[UNSW_NB15_TARGET_COLUMN] = df[
            UNSW_NB15_TARGET_COLUMN
        ].replace({"nan": "normal", "": "normal"})

    return df


def get_unsw_nb15_summary(df: pd.DataFrame) -> Dict:
    """
    Return summary statistics for a loaded UNSW-NB15
    DataFrame.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict
    """
    label_dist = (
        df[UNSW_NB15_TARGET_COLUMN].value_counts().to_dict()
        if UNSW_NB15_TARGET_COLUMN in df.columns
        else {}
    )
    missing_pct = float(
        df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
    )
    return {
        "dataset": "UNSW-NB15",
        "n_samples": len(df),
        "n_features_raw": 49,
        "n_attack_families": 9,
        "n_classes": 10,
        "attack_category_distribution": label_dist,
        "missing_value_pct": round(missing_pct, 4),
    }


# Unified Loader Dispatcher

def load_dataset(
    dataset: str,
    merge: bool = True,
    validate: bool = True,
    use_20pct_train: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load the specified dataset and return a (main, test)
    DataFrame tuple.

    This is the single entry point used by the preprocessing
    pipeline and all higher-level modules.  Callers do not
    need to know which loader handles which dataset.

    Parameters
    ----------
    dataset : str
        Dataset identifier — ``nsl_kdd``, ``cicids2017``,
        or ``unsw_nb15``.
    merge : bool
        Merge train + test before returning (NSL-KDD and
        UNSW-NB15 only).  CICIDS2017 does not have a
        pre-defined split and is always returned as a single
        merged DataFrame.
    validate : bool
        Run validation checks after loading.
    use_20pct_train : bool
        Use the NSL-KDD 20 % training subset (NSL-KDD only).

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame or None)
        - First element: full / merged DataFrame.
        - Second element: original test DataFrame, or None
          for CICIDS2017.

    Raises
    ------
    ValueError
        If *dataset* is not recognised.
    FileNotFoundError
        If dataset files are missing.
    """
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Supported: {SUPPORTED_DATASETS}"
        )

    logger.info("=" * 60)
    logger.info("LOADING DATASET: %s", dataset.upper())
    logger.info("=" * 60)

    if dataset == "nsl_kdd":
        main_df, test_df = load_nsl_kdd(
            merge=merge,
            validate=validate,
            use_20pct_train=use_20pct_train,
        )
        return main_df, test_df

    elif dataset == "cicids2017":
        main_df = load_cicids2017(validate=validate)
        return main_df, None

    elif dataset == "unsw_nb15":
        main_df, test_df = load_unsw_nb15(
            merge=merge,
            validate=validate,
        )
        return main_df, test_df

    # Should never reach here given the check above
    raise ValueError(f"Unhandled dataset: {dataset}")


def get_dataset_summary(
    df: pd.DataFrame,
    dataset: str,
) -> Dict:
    """
    Return dataset-specific summary statistics for the
    Chapter 4 Dataset Summary table.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded dataset DataFrame.
    dataset : str
        Dataset identifier.

    Returns
    -------
    dict
        Summary statistics dictionary.

    Raises
    ------
    ValueError
        If *dataset* is not recognised.
    """
    summarisers = {
        "nsl_kdd":    get_nsl_kdd_summary,
        "cicids2017": get_cicids2017_summary,
        "unsw_nb15":  get_unsw_nb15_summary,
    }
    if dataset not in summarisers:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Supported: {SUPPORTED_DATASETS}"
        )
    summary = summarisers[dataset](df)
    logger.info("Dataset summary for %s:", dataset.upper())
    for key, val in summary.items():
        if not isinstance(val, dict):
            logger.info("  %-35s %s", key, val)
    return summary


# Convenience Loaders (shorthand for the most common usage)

def load_nsl_kdd_train_test() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load NSL-KDD as separate (train, test) DataFrames without
    merging.  Convenience wrapper for notebooks and unit tests.

    Returns
    -------
    tuple of (train_df, test_df)
    """
    return load_nsl_kdd(merge=False, validate=True)


def load_nsl_kdd_merged() -> pd.DataFrame:
    """
    Load NSL-KDD as a single merged DataFrame.  The ``_split``
    column records the origin of each row (``"train"``/``"test"``).

    Returns
    -------
    pd.DataFrame
    """
    merged, _ = load_nsl_kdd(merge=True, validate=True)
    return merged