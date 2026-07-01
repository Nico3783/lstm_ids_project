
# src/data/preprocessing.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Complete data preprocessing pipeline transforming
#          raw loaded DataFrames into clean, encoded, and
#          scaled feature matrices ready for sequence building.
#
#          Pipeline stages (Chapter 3, Section 3.5.2):
#            1. Drop irrelevant columns (difficulty, _split)
#            2. Handle missing values — mean (continuous),
#               mode (categorical)
#            3. Handle infinite values — replace with NaN
#            4. Remove duplicate rows
#            5. Map NSL-KDD raw attack types → 5-class labels
#            6. One-hot encode categorical features
#            7. Label encode the target column → integers
#            8. Min-Max scale numeric features [0, 1]
#            9. Optionally apply SMOTE for class balancing
#           10. Save all artifacts (scaler, encoder, names)
#
#          The scaler and label encoder are fitted on the
#          training portion only and applied to val/test to
#          prevent data leakage (Chapter 3, Section 3.5.2).

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from src.utils.constants import (
    NSL_KDD_TARGET_COLUMN,
    NSL_KDD_DIFFICULTY_COLUMN,
    NSL_KDD_CATEGORICAL_FEATURES,
    NSL_KDD_ATTACK_TO_CATEGORY,
    NSL_KDD_CATEGORY_TO_INT,
    NSL_KDD_CLASS_NAMES,
    CICIDS2017_TARGET_COLUMN,
    CICIDS2017_BENIGN_LABEL,
    UNSW_NB15_TARGET_COLUMN,
    UNSW_NB15_BINARY_LABEL_COLUMN,
    UNSW_NB15_CATEGORICAL_FEATURES,
    SUPPORTED_DATASETS,
)
from src.utils.logger import get_logger
from src.utils.paths import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    ensure_dir,
)
from src.utils.serialization import save_preprocessing_artifacts

logger = get_logger(__name__)


# Step 1 — Drop Irrelevant Columns

def drop_irrelevant_columns(
    df: pd.DataFrame,
    dataset: str = "nsl_kdd",
) -> pd.DataFrame:
    """
    Remove columns that carry no predictive information and
    should not enter the feature matrix.

    For NSL-KDD: drops the ``difficulty`` column (a meta-label
    recording how hard the record was to classify — not a
    network feature) and the ``_split`` tracking column added
    by the loader.

    Parameters
    ----------
    df : pd.DataFrame
    dataset : str

    Returns
    -------
    pd.DataFrame
        DataFrame with irrelevant columns removed.
    """
    cols_to_drop: List[str] = ["_split"]

    if dataset == "nsl_kdd":
        cols_to_drop.append(NSL_KDD_DIFFICULTY_COLUMN)

    existing = [c for c in cols_to_drop if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
        logger.info(
            "Dropped irrelevant columns: %s", existing
        )
    return df


# Step 2 & 3 — Handle Missing and Infinite Values

def handle_missing_and_infinite(
    df: pd.DataFrame,
    strategy_continuous: str = "mean",
    strategy_categorical: str = "mode",
    dataset: str = "nsl_kdd",
) -> pd.DataFrame:
    """
    Replace infinite values with NaN, then impute missing
    values using the specified strategies.

    Chapter 3, Section 3.5.2:
    - Continuous features: mean imputation
    - Categorical features: mode imputation
    - Infinite values (common in CICIDS2017 flow rate
      features) are replaced with NaN before imputation.

    Parameters
    ----------
    df : pd.DataFrame
    strategy_continuous : str
        ``"mean"`` or ``"median"`` for numeric columns.
    strategy_categorical : str
        ``"mode"`` for categorical columns.
    dataset : str

    Returns
    -------
    pd.DataFrame
        DataFrame with no missing or infinite values.
    """
    # Replace inf / -inf → NaN
    n_inf = int(
        df.select_dtypes(include=[np.number])
        .isin([np.inf, -np.inf])
        .sum()
        .sum()
    )
    if n_inf > 0:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        logger.info(
            "Replaced %d infinite values with NaN.", n_inf
        )

    n_missing_before = int(df.isnull().sum().sum())
    if n_missing_before == 0:
        logger.info("No missing values — imputation skipped.")
        return df

    logger.info(
        "Imputing %d missing values "
        "(continuous: %s, categorical: %s) ...",
        n_missing_before, strategy_continuous, strategy_categorical,
    )

    # Numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].isnull().any():
            if strategy_continuous == "mean":
                fill_val = df[col].mean()
            elif strategy_continuous == "median":
                fill_val = df[col].median()
            else:
                fill_val = df[col].mean()
            df[col].fillna(fill_val, inplace=True)

    # Categorical / object columns
    cat_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    for col in cat_cols:
        if df[col].isnull().any():
            fill_val = df[col].mode()
            fill_val = fill_val.iloc[0] if len(fill_val) > 0 else "unknown"
            df[col].fillna(fill_val, inplace=True)

    n_missing_after = int(df.isnull().sum().sum())
    logger.info(
        "Imputation complete. Missing values: %d → %d.",
        n_missing_before, n_missing_after,
    )
    return df


# Step 4 — Remove Duplicates

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate rows from the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    n_before = len(df)
    df = df.drop_duplicates()
    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.info(
            "Removed %d duplicate rows (%d → %d).",
            n_removed, n_before, len(df),
        )
    else:
        logger.info("No duplicate rows found.")
    return df.reset_index(drop=True)


# Step 5 — Label Mapping (NSL-KDD specific)

def map_nsl_kdd_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map NSL-KDD raw attack-type strings to 5-class integer
    labels as specified in Chapter 3, Section 3.5.2:

        normal → 0
        dos    → 1
        probe  → 2
        r2l    → 3
        u2r    → 4

    This two-step mapping (raw string → category string →
    integer) matches the methodology described in the report.
    Records with unrecognised attack types are dropped with a
    warning so the model is never trained on ambiguous labels.

    Parameters
    ----------
    df : pd.DataFrame
        NSL-KDD DataFrame with raw ``label`` column values
        such as ``"neptune"``, ``"normal"``, ``"portsweep"``.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``label`` column replaced by integer
        class codes (0–4).
    """
    if NSL_KDD_TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{NSL_KDD_TARGET_COLUMN}' not found."
        )

    # Step 1: raw attack type → category string
    df["label_category"] = (
        df[NSL_KDD_TARGET_COLUMN]
        .str.lower()
        .str.strip()
        .map(NSL_KDD_ATTACK_TO_CATEGORY)
    )

    n_unknown = df["label_category"].isnull().sum()
    if n_unknown > 0:
        unknown_vals = (
            df.loc[df["label_category"].isnull(), NSL_KDD_TARGET_COLUMN]
            .unique()
            .tolist()
        )
        logger.warning(
            "Dropping %d records with unrecognised attack types: %s",
            n_unknown, unknown_vals,
        )
        df = df.dropna(subset=["label_category"])

    # Step 2: category string → integer
    df[NSL_KDD_TARGET_COLUMN] = (
        df["label_category"].map(NSL_KDD_CATEGORY_TO_INT).astype(int)
    )
    df.drop(columns=["label_category"], inplace=True)

    dist = df[NSL_KDD_TARGET_COLUMN].value_counts().sort_index()
    logger.info("NSL-KDD label mapping complete.")
    logger.info("  Class distribution after mapping:")
    for cls_int, count in dist.items():
        cls_name = NSL_KDD_CLASS_NAMES[int(cls_int)]
        logger.info("    %d (%s): %d", cls_int, cls_name, count)

    return df


def map_cicids2017_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map CICIDS2017 string labels to integer codes.

    BENIGN → 0, attack categories → 1 … N in sorted order.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        With ``Label`` column replaced by integer codes.
    """
    target = CICIDS2017_TARGET_COLUMN.strip()
    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found in DataFrame."
        )

    categories = sorted(df[target].unique().tolist())

    # Place BENIGN at index 0
    if CICIDS2017_BENIGN_LABEL in categories:
        categories.remove(CICIDS2017_BENIGN_LABEL)
        categories = [CICIDS2017_BENIGN_LABEL] + categories

    mapping = {cat: idx for idx, cat in enumerate(categories)}
    df[target] = df[target].map(mapping)

    logger.info(
        "CICIDS2017 label mapping complete — %d classes.",
        len(mapping),
    )
    return df


def map_unsw_nb15_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map UNSW-NB15 attack category strings to integer codes.

    'normal' → 0, attack families → 1 … N alphabetically.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        With ``attack_cat`` replaced by integer codes.
    """
    if UNSW_NB15_TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{UNSW_NB15_TARGET_COLUMN}' not found."
        )

    df[UNSW_NB15_TARGET_COLUMN] = (
        df[UNSW_NB15_TARGET_COLUMN]
        .astype(str)
        .str.lower()
        .str.strip()
    )
    categories = sorted(df[UNSW_NB15_TARGET_COLUMN].unique().tolist())

    if "normal" in categories:
        categories.remove("normal")
        categories = ["normal"] + categories

    mapping = {cat: idx for idx, cat in enumerate(categories)}
    df[UNSW_NB15_TARGET_COLUMN] = (
        df[UNSW_NB15_TARGET_COLUMN].map(mapping)
    )

    # Drop the binary label column — we use the multi-class target
    if UNSW_NB15_BINARY_LABEL_COLUMN in df.columns:
        df.drop(columns=[UNSW_NB15_BINARY_LABEL_COLUMN], inplace=True)

    logger.info(
        "UNSW-NB15 label mapping complete — %d classes.",
        len(mapping),
    )
    return df


# Step 6 — One-Hot Encoding

def encode_categorical_features(
    df: pd.DataFrame,
    dataset: str = "nsl_kdd",
    drop_first: bool = False,
) -> pd.DataFrame:
    """
    One-hot encode categorical features as described in
    Chapter 3, Section 3.5.2.

    For NSL-KDD: encodes ``protocol_type``, ``service``,
    and ``flag`` columns — preserving all dummy categories
    (drop_first=False) so ordinal assumptions are not
    incorrectly imposed.

    Parameters
    ----------
    df : pd.DataFrame
    dataset : str
    drop_first : bool
        Drop first dummy column per category (default False).

    Returns
    -------
    pd.DataFrame
        DataFrame with categorical columns replaced by
        one-hot encoded binary columns.
    """
    if dataset == "nsl_kdd":
        cat_cols = [
            c for c in NSL_KDD_CATEGORICAL_FEATURES
            if c in df.columns
        ]
    elif dataset == "unsw_nb15":
        cat_cols = [
            c for c in UNSW_NB15_CATEGORICAL_FEATURES
            if c in df.columns
        ]
    elif dataset == "cicids2017":
        # CICIDS2017 features are all numeric — no encoding needed
        cat_cols = []
    else:
        cat_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        # Exclude the target column
        target_candidates = [
            NSL_KDD_TARGET_COLUMN,
            CICIDS2017_TARGET_COLUMN.strip(),
            UNSW_NB15_TARGET_COLUMN,
        ]
        cat_cols = [c for c in cat_cols if c not in target_candidates]

    if not cat_cols:
        logger.info(
            "No categorical features to encode for '%s'.", dataset
        )
        return df

    n_before = len(df.columns)
    df = pd.get_dummies(
        df,
        columns=cat_cols,
        drop_first=drop_first,
        dtype=float,
    )
    n_after = len(df.columns)
    logger.info(
        "One-hot encoding: %d categorical columns → %d new columns "
        "(%d total, was %d).",
        len(cat_cols), n_after - n_before + len(cat_cols),
        n_after, n_before,
    )
    return df


# Step 7 — Separate Features and Target

def separate_features_target(
    df: pd.DataFrame,
    dataset: str = "nsl_kdd",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split the DataFrame into feature matrix X and target
    vector y.

    Parameters
    ----------
    df : pd.DataFrame
    dataset : str

    Returns
    -------
    tuple of (X: pd.DataFrame, y: pd.Series)

    Raises
    ------
    ValueError
        If the target column is not present.
    """
    target_col = _get_target_col(dataset)

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found after preprocessing. "
            f"Available: {list(df.columns)}"
        )

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    logger.info(
        "Features / target separated — X: %s, y: %s, classes: %s",
        X.shape, y.shape, sorted(y.unique().tolist()),
    )
    return X, y


# Step 8 — Min-Max Scaling

def fit_scaler(
    X_train: pd.DataFrame,
    feature_range: Tuple[float, float] = (0.0, 1.0),
) -> MinMaxScaler:
    """
    Fit a MinMaxScaler on the training feature matrix.

    The scaler is fitted exclusively on the training set and
    is then applied to validation and test sets to prevent
    data leakage (Chapter 3, Section 3.5.2).

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    feature_range : tuple
        Output range, default [0, 1].

    Returns
    -------
    MinMaxScaler
        Fitted scaler instance.
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler.fit(X_train)
    logger.info(
        "MinMaxScaler fitted on training data — "
        "feature range: %s.", feature_range,
    )
    return scaler


def apply_scaler(
    X: pd.DataFrame,
    scaler: MinMaxScaler,
) -> np.ndarray:
    """
    Apply a fitted MinMaxScaler to a feature matrix.

    Parameters
    ----------
    X : pd.DataFrame
    scaler : MinMaxScaler

    Returns
    -------
    np.ndarray
        Scaled feature matrix as float32.
    """
    scaled = scaler.transform(X).astype(np.float32)
    logger.debug(
        "Scaler applied — shape %s, range [%.4f, %.4f].",
        scaled.shape, scaled.min(), scaled.max(),
    )
    return scaled


# Step 9 — Optional SMOTE

def apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique)
    to balance class distribution in the training set.

    SMOTE is optional in this project; the primary class
    balancing strategy is inverse-frequency class weighting
    (Chapter 3, Section 3.5.4).  This function is called
    only when ``use_smote: true`` in config.yaml.

    Parameters
    ----------
    X : np.ndarray
        2-D training feature array (samples, features).
    y : np.ndarray
        1-D training label array.
    random_state : int

    Returns
    -------
    tuple of (X_resampled, y_resampled)

    Raises
    ------
    ImportError
        If ``imbalanced-learn`` is not installed.
    """
    try:
        from imblearn.over_sampling import SMOTE  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "imbalanced-learn is required for SMOTE. "
            "Install it with: pip install imbalanced-learn"
        ) from exc

    logger.info(
        "Applying SMOTE — before: %d samples, distribution: %s",
        len(y),
        dict(zip(*np.unique(y, return_counts=True))),
    )

    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)

    logger.info(
        "SMOTE complete — after: %d samples, distribution: %s",
        len(y_res),
        dict(zip(*np.unique(y_res, return_counts=True))),
    )
    return X_res, y_res


# Save Interim DataFrames

def save_interim(
    df: pd.DataFrame,
    stage: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Save an intermediate DataFrame to the interim data
    directory for inspection, debugging, and Chapter 3
    documentation.

    Parameters
    ----------
    df : pd.DataFrame
    stage : str
        Pipeline stage name — one of ``"merged"``,
        ``"cleaned"``, ``"encoded"``, ``"scaled"``.
    output_dir : Path, optional
        Override directory.  Defaults to ``data/interim/``.

    Returns
    -------
    Path
        Path to the saved CSV.
    """
    from src.utils.constants import (
        MERGED_DATASET_CSV, CLEANED_DATASET_CSV,
        ENCODED_DATASET_CSV, SCALED_DATASET_CSV,
    )

    filenames = {
        "merged":  MERGED_DATASET_CSV,
        "cleaned": CLEANED_DATASET_CSV,
        "encoded": ENCODED_DATASET_CSV,
        "scaled":  SCALED_DATASET_CSV,
    }
    fname = filenames.get(stage, f"{stage}_dataset.csv")
    out_dir = output_dir or INTERIM_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / fname
    df.to_csv(out_path, index=False)
    logger.info(
        "Interim data saved [%s]: %s (%d rows)",
        stage, out_path, len(df),
    )
    return out_path


# Full Preprocessing Pipeline

def preprocess_dataset(
    df: pd.DataFrame,
    dataset: str = "nsl_kdd",
    strategy_continuous: str = "mean",
    strategy_categorical: str = "mode",
    drop_first: bool = False,
    feature_range: Tuple[float, float] = (0.0, 1.0),
    save_interim_files: bool = True,
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, List[str], Dict]:
    """
    Execute the complete preprocessing pipeline on the full
    (merged) dataset and return scaled numpy arrays.

    This function is called by the main pipeline BEFORE the
    train/val/test split so the split module receives a
    clean, encoded, scaled array.

    .. warning::
        When using this function the scaler is fitted on the
        entire dataset.  For strict leakage prevention, use
        ``preprocess_train_val_test()`` which fits the scaler
        on the training split only.

    Pipeline
    --------
    1.  Drop irrelevant columns
    2.  Handle missing / infinite values
    3.  Remove duplicates
    4.  Map labels to integers
    5.  One-hot encode categorical features
    6.  Separate X and y
    7.  Fit and apply MinMax scaler
    8.  Save preprocessing artifacts

    Parameters
    ----------
    df : pd.DataFrame
        Loaded merged dataset (output of ``load_dataset()``).
    dataset : str
    strategy_continuous : str
    strategy_categorical : str
    drop_first : bool
    feature_range : tuple
    save_interim_files : bool
        Save each intermediate stage as CSV for inspection.
    artifacts_dir : Path, optional
        Directory to save scaler + metadata.

    Returns
    -------
    tuple of
        (X_scaled: np.ndarray,
         y: np.ndarray,
         scaler: MinMaxScaler,
         feature_names: list of str,
         metadata: dict)
    """
    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE — %s", dataset.upper())
    logger.info("=" * 60)

    # Step 1 — Drop irrelevant columns
    df = drop_irrelevant_columns(df, dataset=dataset)

    if save_interim_files:
        save_interim(df, "merged")

    # Step 2 & 3 — Missing / infinite values
    df = handle_missing_and_infinite(
        df,
        strategy_continuous=strategy_continuous,
        strategy_categorical=strategy_categorical,
        dataset=dataset,
    )

    # Step 4 — Remove duplicates
    df = remove_duplicates(df)

    if save_interim_files:
        save_interim(df, "cleaned")

    # Step 5 — Map labels
    df = _map_labels(df, dataset=dataset)

    # Step 6 — One-hot encode
    df = encode_categorical_features(
        df, dataset=dataset, drop_first=drop_first
    )

    if save_interim_files:
        save_interim(df, "encoded")

    # Step 7 — Separate X and y
    X_df, y = separate_features_target(df, dataset=dataset)
    feature_names = X_df.columns.tolist()

    logger.info(
        "Feature matrix: %d samples × %d features.",
        len(X_df), len(feature_names),
    )

    # Step 8 — Scale
    scaler = fit_scaler(X_df, feature_range=feature_range)
    X_scaled = apply_scaler(X_df, scaler)
    y_array = y.values.astype(np.int64)

    if save_interim_files:
        scaled_df = pd.DataFrame(
            X_scaled, columns=feature_names
        )
        scaled_df["label"] = y_array
        save_interim(scaled_df, "scaled")

    # Metadata snapshot
    n_classes = int(np.max(y_array)) + 1
    metadata = {
        "dataset": dataset,
        "n_samples": int(len(X_scaled)),
        "n_features": int(len(feature_names)),
        "n_classes": n_classes,
        "class_names": (
            NSL_KDD_CLASS_NAMES if dataset == "nsl_kdd"
            else [str(i) for i in range(n_classes)]
        ),
        "feature_range": list(feature_range),
        "missing_strategy_continuous": strategy_continuous,
        "missing_strategy_categorical": strategy_categorical,
        "encoding": "onehot",
        "scaling": "minmax",
        "class_distribution": {
            int(k): int(v)
            for k, v in zip(
                *np.unique(y_array, return_counts=True)
            )
        },
    }

    # Save artifacts if directory specified
    if artifacts_dir is not None:
        # Create a dummy LabelEncoder to satisfy the artifact
        # bundle — encoding was done via integer mapping
        le = LabelEncoder()
        le.classes_ = np.array(metadata["class_names"])
        save_preprocessing_artifacts(
            scaler=scaler,
            label_encoder=le,
            feature_names=feature_names,
            metadata=metadata,
            output_dir=artifacts_dir,
        )

    logger.info(
        "Preprocessing complete — X: %s, y: %s, classes: %d.",
        X_scaled.shape, y_array.shape, n_classes,
    )
    return X_scaled, y_array, scaler, feature_names, metadata


def preprocess_train_val_test(
    X_train_df: pd.DataFrame,
    X_val_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    feature_range: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Fit MinMaxScaler on the training split only and apply to
    all three splits — the leakage-safe approach described in
    Chapter 3, Section 3.5.2.

    Parameters
    ----------
    X_train_df, X_val_df, X_test_df : pd.DataFrame
        Feature-only DataFrames (no target column) for each
        split.
    feature_range : tuple

    Returns
    -------
    tuple of (X_train_scaled, X_val_scaled, X_test_scaled,
              fitted_scaler)
    """
    logger.info(
        "Fitting scaler on training split only "
        "(leakage-safe approach) ..."
    )
    scaler = fit_scaler(X_train_df, feature_range=feature_range)

    X_train_scaled = apply_scaler(X_train_df, scaler)
    X_val_scaled   = apply_scaler(X_val_df,   scaler)
    X_test_scaled  = apply_scaler(X_test_df,  scaler)

    logger.info(
        "Scaling applied — train %s | val %s | test %s.",
        X_train_scaled.shape,
        X_val_scaled.shape,
        X_test_scaled.shape,
    )
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


# New Data Preprocessing (Inference)

def preprocess_new_data(
    df: pd.DataFrame,
    dataset: str,
    scaler: MinMaxScaler,
    feature_names: List[str],
    strategy_continuous: str = "mean",
    strategy_categorical: str = "mode",
    drop_first: bool = False,
) -> np.ndarray:
    """
    Apply the saved preprocessing pipeline to new, unseen
    data for inference.

    Uses the already-fitted *scaler* and the *feature_names*
    list from training to ensure exact column alignment.

    Parameters
    ----------
    df : pd.DataFrame
        Raw new data in the same format as the training data.
    dataset : str
    scaler : MinMaxScaler
        Fitted scaler from training.
    feature_names : list of str
        Ordered list of feature names from training.
    strategy_continuous : str
    strategy_categorical : str
    drop_first : bool

    Returns
    -------
    np.ndarray
        Scaled 2-D feature array (n_samples, n_features).
    """
    logger.info(
        "Preprocessing new data for inference "
        "(%d rows) ...", len(df),
    )

    df = drop_irrelevant_columns(df, dataset=dataset)
    df = handle_missing_and_infinite(
        df,
        strategy_continuous=strategy_continuous,
        strategy_categorical=strategy_categorical,
        dataset=dataset,
    )
    df = encode_categorical_features(
        df, dataset=dataset, drop_first=drop_first
    )

    # Drop target column if present
    target_col = _get_target_col(dataset)
    if target_col in df.columns:
        df = df.drop(columns=[target_col])

    # Align columns with training feature names
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0   # Add missing dummy columns as zero

    df = df[feature_names]   # Reorder to match training order

    X_scaled = apply_scaler(df, scaler)
    logger.info(
        "New data preprocessed — shape: %s.", X_scaled.shape
    )
    return X_scaled


# Internal Helpers

def _map_labels(
    df: pd.DataFrame,
    dataset: str,
) -> pd.DataFrame:
    """Dispatch to the correct label mapper."""
    if dataset == "nsl_kdd":
        return map_nsl_kdd_labels(df)
    elif dataset == "cicids2017":
        return map_cicids2017_labels(df)
    elif dataset == "unsw_nb15":
        return map_unsw_nb15_labels(df)
    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Supported: {SUPPORTED_DATASETS}"
        )


def _get_target_col(dataset: str) -> str:
    """Return the target column name for the given dataset."""
    mapping = {
        "nsl_kdd":    NSL_KDD_TARGET_COLUMN,
        "cicids2017": CICIDS2017_TARGET_COLUMN.strip(),
        "unsw_nb15":  UNSW_NB15_TARGET_COLUMN,
    }
    return mapping.get(dataset, "label")