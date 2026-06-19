
# src/data/validators.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Data quality validation layer that runs immediately
#          after raw data is loaded and before any preprocessing
#          transformation is applied.
#
#          Validators catch structural problems — wrong column
#          counts, missing target columns, unsupported dtypes,
#          extreme missing-value ratios, and label integrity
#          issues — early in the pipeline so that errors are
#          reported with clear, actionable messages rather than
#          cryptic downstream exceptions.
#
#          Each validator returns a ValidationReport dataclass
#          that records every check performed, its pass/fail
#          status, and descriptive messages.  The report is
#          logged and optionally saved to JSON so it can be
#          included in Chapter 3 documentation as evidence of
#          rigorous data quality assessment.
#
#          Aligned with Chapter 3, Section 3.5.2 —
#          Data Preprocessing Pipeline (Data Cleaning step).

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.constants import (
    NSL_KDD_COLUMNS,
    NSL_KDD_CATEGORICAL_FEATURES,
    NSL_KDD_NUMERICAL_FEATURES,
    NSL_KDD_TARGET_COLUMN,
    NSL_KDD_ATTACK_TO_CATEGORY,
    NSL_KDD_CATEGORY_TO_INT,
    CICIDS2017_TARGET_COLUMN,
    CICIDS2017_BENIGN_LABEL,
    UNSW_NB15_TARGET_COLUMN,
    UNSW_NB15_BINARY_LABEL_COLUMN,
    SUPPORTED_DATASETS,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Validation Report Data Structures

@dataclass
class CheckResult:
    """
    Result of a single validation check.

    Attributes
    ----------
    name : str
        Short identifier for the check, e.g. ``"column_count"``.
    passed : bool
        True if the check succeeded.
    message : str
        Human-readable description of the outcome.
    severity : str
        One of ``"error"`` (pipeline should halt),
        ``"warning"`` (pipeline may continue with caution),
        or ``"info"`` (informational only).
    details : dict
        Optional extra information (counts, lists, etc.)
    """
    name: str
    passed: bool
    message: str
    severity: str = "error"
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """
    Aggregated report from running all validation checks on
    a loaded DataFrame.

    Attributes
    ----------
    dataset : str
        Dataset identifier (``nsl_kdd``, etc.).
    split : str
        Which split was validated (``"train"``, ``"test"``,
        ``"full"``).
    n_rows : int
        Number of rows in the validated DataFrame.
    n_cols : int
        Number of columns in the validated DataFrame.
    checks : list of CheckResult
        All individual check results.
    passed : bool
        True only if every ``"error"``-severity check passed.
    """
    dataset: str
    split: str
    n_rows: int
    n_cols: int
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if no error-severity check failed."""
        return all(
            c.passed for c in self.checks if c.severity == "error"
        )

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    @property
    def errors(self) -> List[CheckResult]:
        return [
            c for c in self.checks
            if not c.passed and c.severity == "error"
        ]

    @property
    def warnings(self) -> List[CheckResult]:
        return [
            c for c in self.checks
            if not c.passed and c.severity == "warning"
        ]

    def summary(self) -> str:
        """Return a multi-line human-readable summary string."""
        lines = [
            f"Validation Report — {self.dataset} ({self.split})",
            f"  Rows: {self.n_rows:,}  |  Columns: {self.n_cols}",
            f"  Checks: {len(self.checks)} total, "
            f"{self.n_passed} passed, {self.n_failed} failed",
            f"  Overall: {'PASSED ✓' if self.passed else 'FAILED ✗'}",
        ]
        if self.errors:
            lines.append("  ERRORS:")
            for e in self.errors:
                lines.append(f"    ✗ [{e.name}] {e.message}")
        if self.warnings:
            lines.append("  WARNINGS:")
            for w in self.warnings:
                lines.append(f"    ⚠ [{w.name}] {w.message}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the report to a JSON-compatible dict."""
        return {
            "dataset": self.dataset,
            "split": self.split,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "overall_passed": self.passed,
            "n_checks": len(self.checks),
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "severity": c.severity,
                    "message": c.message,
                    "details": c.details,
                }
                for c in self.checks
            ],
        }

    def save(self, path: Path) -> None:
        """
        Save the report as a JSON file.

        Parameters
        ----------
        path : Path
            Destination file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=4)
        logger.info("Validation report saved: %s", path)


# Generic Checks (shared across datasets)

def _check_not_empty(df: pd.DataFrame) -> CheckResult:
    """Check that the DataFrame is not empty."""
    passed = len(df) > 0
    return CheckResult(
        name="not_empty",
        passed=passed,
        severity="error",
        message=(
            f"DataFrame has {len(df):,} rows."
            if passed
            else "DataFrame is empty — no rows loaded."
        ),
        details={"n_rows": len(df)},
    )


def _check_column_count(
    df: pd.DataFrame,
    expected: int,
    tolerance: int = 5,
) -> CheckResult:
    """
    Check that the DataFrame column count is close to expected.

    A tolerance of ±5 accommodates one-hot encoding expansion
    that may have already been applied.
    """
    actual = len(df.columns)
    passed = abs(actual - expected) <= tolerance
    return CheckResult(
        name="column_count",
        passed=passed,
        severity="warning",
        message=(
            f"Column count: {actual} "
            f"(expected ~{expected}, tolerance ±{tolerance})."
        ),
        details={"actual": actual, "expected": expected},
    )


def _check_target_column_exists(
    df: pd.DataFrame,
    target_col: str,
) -> CheckResult:
    """Check that the target/label column is present."""
    passed = target_col in df.columns
    return CheckResult(
        name="target_column_exists",
        passed=passed,
        severity="error",
        message=(
            f"Target column '{target_col}' found."
            if passed
            else f"Target column '{target_col}' NOT FOUND in DataFrame. "
                 f"Available columns: {list(df.columns[:10])} ..."
        ),
        details={"target_col": target_col},
    )


def _check_missing_values(
    df: pd.DataFrame,
    max_missing_ratio: float = 0.50,
) -> CheckResult:
    """
    Check per-column missing value ratios.

    Raises a warning if any column exceeds *max_missing_ratio*
    of missing values (50 % by default).  The CICIDS2017
    dataset is known to contain NaN and Inf values in flow
    feature columns (Chapter 3, Section 3.5.2).
    """
    total_cells = len(df) * len(df.columns)
    missing_per_col = df.isnull().sum()
    total_missing = int(missing_per_col.sum())
    overall_ratio = total_missing / total_cells if total_cells > 0 else 0.0

    problematic = missing_per_col[
        missing_per_col / len(df) > max_missing_ratio
    ]

    passed = len(problematic) == 0
    return CheckResult(
        name="missing_values",
        passed=passed,
        severity="warning",
        message=(
            f"Missing values: {total_missing:,} cells "
            f"({overall_ratio:.2%} of total). "
            + (
                f"{len(problematic)} columns exceed "
                f"{max_missing_ratio:.0%} missing threshold."
                if not passed
                else "No column exceeds missing threshold."
            )
        ),
        details={
            "total_missing": total_missing,
            "overall_missing_ratio": round(overall_ratio, 4),
            "problematic_columns": problematic.to_dict(),
        },
    )


def _check_infinite_values(df: pd.DataFrame) -> CheckResult:
    """
    Check for infinite values in numeric columns.

    Inf values are common in CICIDS2017 flow rate features
    (e.g. Flow Bytes/s when duration is zero).
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts: Dict[str, int] = {}
    for col in numeric_cols:
        n_inf = int(np.isinf(df[col]).sum())
        if n_inf > 0:
            inf_counts[col] = n_inf

    total_inf = sum(inf_counts.values())
    passed = total_inf == 0
    return CheckResult(
        name="infinite_values",
        passed=passed,
        severity="warning",
        message=(
            f"Infinite values found: {total_inf:,} cells "
            f"across {len(inf_counts)} columns."
            if not passed
            else "No infinite values detected."
        ),
        details={
            "total_infinite": total_inf,
            "affected_columns": inf_counts,
        },
    )


def _check_duplicate_rows(
    df: pd.DataFrame,
    max_dup_ratio: float = 0.30,
) -> CheckResult:
    """
    Check for duplicate rows.

    A warning is raised if duplicates exceed *max_dup_ratio*
    of total rows (30 % by default).
    """
    n_dups = int(df.duplicated().sum())
    dup_ratio = n_dups / len(df) if len(df) > 0 else 0.0
    passed = dup_ratio <= max_dup_ratio
    return CheckResult(
        name="duplicate_rows",
        passed=passed,
        severity="warning",
        message=(
            f"Duplicate rows: {n_dups:,} ({dup_ratio:.2%} of total)."
            + (
                f" Exceeds threshold of {max_dup_ratio:.0%}."
                if not passed
                else ""
            )
        ),
        details={
            "n_duplicates": n_dups,
            "duplicate_ratio": round(dup_ratio, 4),
        },
    )


def _check_class_distribution(
    df: pd.DataFrame,
    target_col: str,
    min_samples_per_class: int = 10,
) -> CheckResult:
    """
    Check that every class has at least *min_samples_per_class*
    samples.  Classes with very few samples may cause issues
    during stratified splitting and model evaluation.
    """
    if target_col not in df.columns:
        return CheckResult(
            name="class_distribution",
            passed=False,
            severity="error",
            message=f"Cannot check class distribution: "
                    f"target column '{target_col}' not found.",
        )

    dist = df[target_col].value_counts().to_dict()
    sparse_classes = {
        k: v for k, v in dist.items() if v < min_samples_per_class
    }
    passed = len(sparse_classes) == 0
    return CheckResult(
        name="class_distribution",
        passed=passed,
        severity="warning",
        message=(
            f"Class distribution: {len(dist)} classes. "
            + (
                f"Sparse classes (<{min_samples_per_class} samples): "
                f"{sparse_classes}."
                if not passed
                else f"All classes have ≥{min_samples_per_class} samples."
            )
        ),
        details={
            "distribution": {str(k): int(v) for k, v in dist.items()},
            "sparse_classes": {
                str(k): int(v) for k, v in sparse_classes.items()
            },
            "n_classes": len(dist),
        },
    )


def _check_numeric_dtypes(
    df: pd.DataFrame,
    expected_numeric_cols: List[str],
) -> CheckResult:
    """
    Check that expected numeric columns have numeric dtypes.

    Non-numeric dtypes in feature columns would cause silent
    errors during scaling and LSTM training.
    """
    non_numeric: List[str] = []
    for col in expected_numeric_cols:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric.append(col)

    passed = len(non_numeric) == 0
    return CheckResult(
        name="numeric_dtypes",
        passed=passed,
        severity="warning",
        message=(
            f"{len(non_numeric)} expected-numeric columns have "
            f"non-numeric dtypes: {non_numeric[:10]}."
            if not passed
            else "All expected numeric columns have numeric dtypes."
        ),
        details={"non_numeric_columns": non_numeric},
    )


def _check_feature_variance(
    df: pd.DataFrame,
    numeric_cols: List[str],
    min_variance: float = 0.0,
) -> CheckResult:
    """
    Check for zero-variance (constant) numeric columns.

    Constant features carry no information and can cause
    numerical instability in scaling operations.
    """
    cols_present = [c for c in numeric_cols if c in df.columns]
    if not cols_present:
        return CheckResult(
            name="feature_variance",
            passed=True,
            severity="info",
            message="No numeric columns to check for variance.",
        )

    variances = df[cols_present].var(numeric_only=True)
    zero_var_cols = variances[variances <= min_variance].index.tolist()
    passed = len(zero_var_cols) == 0

    return CheckResult(
        name="feature_variance",
        passed=passed,
        severity="warning",
        message=(
            f"{len(zero_var_cols)} zero-variance (constant) feature "
            f"columns detected: {zero_var_cols[:10]}."
            if not passed
            else "No zero-variance feature columns detected."
        ),
        details={
            "zero_variance_columns": zero_var_cols,
            "n_zero_variance": len(zero_var_cols),
        },
    )


def _check_label_values(
    df: pd.DataFrame,
    target_col: str,
    valid_labels: Optional[List[Any]] = None,
) -> CheckResult:
    """
    Check that all values in the target column are among
    *valid_labels*.  Unknown labels would be silently dropped
    or mapped incorrectly by the label encoder.
    """
    if target_col not in df.columns:
        return CheckResult(
            name="label_values",
            passed=False,
            severity="error",
            message=f"Target column '{target_col}' not found.",
        )

    if valid_labels is None:
        return CheckResult(
            name="label_values",
            passed=True,
            severity="info",
            message="Label value check skipped — no valid_labels provided.",
        )

    unique_labels = set(df[target_col].dropna().unique().tolist())
    valid_set = set(valid_labels)
    unknown = unique_labels - valid_set
    passed = len(unknown) == 0

    return CheckResult(
        name="label_values",
        passed=passed,
        severity="warning",
        message=(
            f"Unknown label values found: {list(unknown)[:20]}. "
            f"These will be dropped during preprocessing."
            if not passed
            else f"All {len(unique_labels)} label values are recognised."
        ),
        details={
            "unique_labels_found": list(unique_labels),
            "unknown_labels": list(unknown),
        },
    )


def _check_feature_range(
    df: pd.DataFrame,
    numeric_cols: List[str],
    expected_min: float = 0.0,
    expected_max: float = 1.0,
    tolerance: float = 0.01,
) -> CheckResult:
    """
    Check whether numeric columns fall within the expected
    scaled range [0, 1] (used AFTER scaling to confirm the
    MinMaxScaler applied correctly).
    """
    cols_present = [c for c in numeric_cols if c in df.columns]
    if not cols_present:
        return CheckResult(
            name="feature_range",
            passed=True,
            severity="info",
            message="No numeric columns to check range.",
        )

    out_of_range: List[str] = []
    for col in cols_present:
        col_min = df[col].min()
        col_max = df[col].max()
        if (
            col_min < expected_min - tolerance
            or col_max > expected_max + tolerance
        ):
            out_of_range.append(col)

    passed = len(out_of_range) == 0
    return CheckResult(
        name="feature_range",
        passed=passed,
        severity="warning",
        message=(
            f"{len(out_of_range)} columns outside expected "
            f"range [{expected_min}, {expected_max}]: "
            f"{out_of_range[:10]}."
            if not passed
            else f"All numeric columns within [{expected_min}, {expected_max}]."
        ),
        details={"out_of_range_columns": out_of_range},
    )


# Dataset-Specific Validators

def validate_nsl_kdd_dataframe(
    df: pd.DataFrame,
    split: str = "unknown",
) -> ValidationReport:
    """
    Run all validation checks on a loaded NSL-KDD DataFrame.

    Expected schema: 43 columns (41 features + label +
    difficulty) as defined in Chapter 3, Section 3.3.1.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded NSL-KDD DataFrame (before preprocessing).
    split : str
        Split label for the report (``"train"`` or ``"test"``).

    Returns
    -------
    ValidationReport
    """
    logger.info(
        "Validating NSL-KDD %s split (%d rows × %d cols) ...",
        split, len(df), len(df.columns),
    )

    checks = [
        _check_not_empty(df),
        _check_column_count(df, expected=43, tolerance=2),
        _check_target_column_exists(df, NSL_KDD_TARGET_COLUMN),
        _check_missing_values(df, max_missing_ratio=0.50),
        _check_infinite_values(df),
        _check_duplicate_rows(df, max_dup_ratio=0.30),
        _check_class_distribution(
            df,
            target_col=NSL_KDD_TARGET_COLUMN,
            min_samples_per_class=5,
        ),
        _check_numeric_dtypes(df, NSL_KDD_NUMERICAL_FEATURES),
        _check_feature_variance(df, NSL_KDD_NUMERICAL_FEATURES),
        _check_label_values(
            df,
            target_col=NSL_KDD_TARGET_COLUMN,
            valid_labels=list(NSL_KDD_ATTACK_TO_CATEGORY.keys()),
        ),
    ]

    report = ValidationReport(
        dataset="nsl_kdd",
        split=split,
        n_rows=len(df),
        n_cols=len(df.columns),
        checks=checks,
    )
    _log_report(report)
    return report


def validate_cicids2017_dataframe(
    df: pd.DataFrame,
    split: str = "full",
) -> ValidationReport:
    """
    Run all validation checks on a loaded CICIDS2017 DataFrame.

    Expected schema: ~80 bidirectional flow feature columns
    plus the ``Label`` column (Chapter 3, Section 3.3.1).

    Parameters
    ----------
    df : pd.DataFrame
        Loaded CICIDS2017 DataFrame.
    split : str
        Split label for the report.

    Returns
    -------
    ValidationReport
    """
    logger.info(
        "Validating CICIDS2017 %s split (%d rows × %d cols) ...",
        split, len(df), len(df.columns),
    )

    # Strip whitespace from column names (common in CICIDS2017)
    df.columns = df.columns.str.strip()
    target_col = CICIDS2017_TARGET_COLUMN.strip()

    numeric_cols = [
        c for c in df.columns
        if c != target_col
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    checks = [
        _check_not_empty(df),
        _check_column_count(df, expected=80, tolerance=5),
        _check_target_column_exists(df, target_col),
        _check_missing_values(df, max_missing_ratio=0.50),
        _check_infinite_values(df),
        _check_duplicate_rows(df, max_dup_ratio=0.40),
        _check_class_distribution(
            df, target_col=target_col, min_samples_per_class=10
        ),
        _check_feature_variance(df, numeric_cols),
    ]

    report = ValidationReport(
        dataset="cicids2017",
        split=split,
        n_rows=len(df),
        n_cols=len(df.columns),
        checks=checks,
    )
    _log_report(report)
    return report


def validate_unsw_nb15_dataframe(
    df: pd.DataFrame,
    split: str = "unknown",
) -> ValidationReport:
    """
    Run all validation checks on a loaded UNSW-NB15 DataFrame.

    Expected schema: 49 features + attack_cat + label columns
    (Chapter 3, Section 3.3.1).

    Parameters
    ----------
    df : pd.DataFrame
        Loaded UNSW-NB15 DataFrame.
    split : str
        Split label for the report.

    Returns
    -------
    ValidationReport
    """
    logger.info(
        "Validating UNSW-NB15 %s split (%d rows × %d cols) ...",
        split, len(df), len(df.columns),
    )

    from src.utils.constants import UNSW_NB15_CATEGORICAL_FEATURES
    numeric_cols = [
        c for c in df.columns
        if c not in UNSW_NB15_CATEGORICAL_FEATURES
        + [UNSW_NB15_TARGET_COLUMN, UNSW_NB15_BINARY_LABEL_COLUMN]
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    checks = [
        _check_not_empty(df),
        _check_column_count(df, expected=49, tolerance=5),
        _check_target_column_exists(df, UNSW_NB15_TARGET_COLUMN),
        _check_missing_values(df, max_missing_ratio=0.50),
        _check_infinite_values(df),
        _check_duplicate_rows(df, max_dup_ratio=0.30),
        _check_class_distribution(
            df,
            target_col=UNSW_NB15_TARGET_COLUMN,
            min_samples_per_class=10,
        ),
        _check_feature_variance(df, numeric_cols),
    ]

    report = ValidationReport(
        dataset="unsw_nb15",
        split=split,
        n_rows=len(df),
        n_cols=len(df.columns),
        checks=checks,
    )
    _log_report(report)
    return report


def validate_processed_arrays(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int = 10,
    split: str = "unknown",
    dataset: str = "nsl_kdd",
) -> ValidationReport:
    """
    Validate the final processed NumPy arrays produced by the
    sequence builder, immediately before they are saved or
    passed to the training module.

    Checks
    ------
    - X is 3-D: (samples, window_size, n_features)
    - y is 1-D: (samples,)
    - Sample counts match between X and y
    - X window dimension matches configured window_size
    - X values are in [0, 1] after MinMax scaling
    - y contains only integer class labels ≥ 0

    Parameters
    ----------
    X : np.ndarray
        Feature array of shape (n_samples, window_size, n_features).
    y : np.ndarray
        Label array of shape (n_samples,).
    window_size : int
        Expected sequence length (timesteps dimension).
    split : str
        Split label for the report.
    dataset : str
        Dataset identifier.

    Returns
    -------
    ValidationReport
    """
    logger.info(
        "Validating processed arrays — %s %s split "
        "(X: %s, y: %s) ...",
        dataset, split, X.shape, y.shape,
    )

    checks: List[CheckResult] = []

    # -- X dimensionality --
    x_3d = X.ndim == 3
    checks.append(CheckResult(
        name="X_ndim",
        passed=x_3d,
        severity="error",
        message=(
            f"X has correct 3-D shape: {X.shape}."
            if x_3d
            else f"X must be 3-D (samples, timesteps, features), "
                 f"got shape {X.shape} ({X.ndim}-D)."
        ),
        details={"shape": list(X.shape), "ndim": X.ndim},
    ))

    # -- y dimensionality --
    y_1d = y.ndim == 1
    checks.append(CheckResult(
        name="y_ndim",
        passed=y_1d,
        severity="error",
        message=(
            f"y has correct 1-D shape: {y.shape}."
            if y_1d
            else f"y must be 1-D (samples,), got shape {y.shape}."
        ),
        details={"shape": list(y.shape), "ndim": y.ndim},
    ))

    # -- Sample count match --
    counts_match = X.shape[0] == y.shape[0]
    checks.append(CheckResult(
        name="sample_count_match",
        passed=counts_match,
        severity="error",
        message=(
            f"X and y sample counts match: {X.shape[0]:,}."
            if counts_match
            else f"X has {X.shape[0]:,} samples but "
                 f"y has {y.shape[0]:,} — mismatch."
        ),
        details={
            "X_samples": int(X.shape[0]),
            "y_samples": int(y.shape[0]),
        },
    ))

    # -- Window size --
    if x_3d:
        win_ok = X.shape[1] == window_size
        checks.append(CheckResult(
            name="window_size",
            passed=win_ok,
            severity="error",
            message=(
                f"Window size correct: {X.shape[1]} timesteps."
                if win_ok
                else f"Window size mismatch: X has {X.shape[1]} "
                     f"timesteps, expected {window_size}."
            ),
            details={
                "actual_window": int(X.shape[1]),
                "expected_window": window_size,
            },
        ))

    # -- Feature values in [0, 1] --
    if x_3d and X.size > 0:
        x_min = float(X.min())
        x_max = float(X.max())
        range_ok = x_min >= -0.01 and x_max <= 1.01
        checks.append(CheckResult(
            name="feature_value_range",
            passed=range_ok,
            severity="warning",
            message=(
                f"Feature values in expected range "
                f"[{x_min:.4f}, {x_max:.4f}]."
                if range_ok
                else f"Feature values outside [0, 1]: "
                     f"min={x_min:.4f}, max={x_max:.4f}. "
                     f"Check MinMax scaling was applied."
            ),
            details={"x_min": x_min, "x_max": x_max},
        ))

    # -- Label values are non-negative integers --
    if y.size > 0:
        labels_valid = bool(
            np.issubdtype(y.dtype, np.integer) and y.min() >= 0
        )
        unique_labels = sorted(np.unique(y).tolist())
        checks.append(CheckResult(
            name="label_integrity",
            passed=labels_valid,
            severity="error",
            message=(
                f"Labels are valid non-negative integers. "
                f"Unique: {unique_labels}."
                if labels_valid
                else f"Labels must be non-negative integers. "
                     f"dtype={y.dtype}, min={y.min()}, "
                     f"unique={unique_labels[:10]}."
            ),
            details={
                "dtype": str(y.dtype),
                "unique_labels": unique_labels,
                "n_classes": len(unique_labels),
            },
        ))

    # -- Not empty --
    not_empty = X.shape[0] > 0
    checks.append(CheckResult(
        name="arrays_not_empty",
        passed=not_empty,
        severity="error",
        message=(
            f"Arrays are non-empty: {X.shape[0]:,} samples."
            if not_empty
            else "Arrays are empty — no samples produced."
        ),
        details={"n_samples": int(X.shape[0]) if X.ndim > 0 else 0},
    ))

    report = ValidationReport(
        dataset=dataset,
        split=split,
        n_rows=int(X.shape[0]) if X.ndim > 0 else 0,
        n_cols=int(X.shape[-1]) if X.ndim > 0 else 0,
        checks=checks,
    )
    _log_report(report)
    return report


# Dispatcher

def validate_dataframe(
    df: pd.DataFrame,
    dataset: str,
    split: str = "unknown",
) -> ValidationReport:
    """
    Dispatch to the correct dataset-specific validator.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded dataset DataFrame.
    dataset : str
        Dataset identifier — ``nsl_kdd``, ``cicids2017``,
        or ``unsw_nb15``.
    split : str
        Split label for the report.

    Returns
    -------
    ValidationReport

    Raises
    ------
    ValueError
        If *dataset* is not a recognised identifier.
    """
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Supported: {SUPPORTED_DATASETS}"
        )
    validators = {
        "nsl_kdd":    validate_nsl_kdd_dataframe,
        "cicids2017": validate_cicids2017_dataframe,
        "unsw_nb15":  validate_unsw_nb15_dataframe,
    }
    return validators[dataset](df, split=split)


def assert_validation_passed(
    report: ValidationReport,
    halt_on_warning: bool = False,
) -> None:
    """
    Raise a ``ValueError`` if the validation report contains
    any failing error-severity checks.

    Parameters
    ----------
    report : ValidationReport
        Report to inspect.
    halt_on_warning : bool
        If True, also raise on warning-severity failures.

    Raises
    ------
    ValueError
        If the report contains error (or warning) failures.
    """
    if not report.passed:
        error_msgs = "\n".join(
            f"  [{e.name}] {e.message}" for e in report.errors
        )
        raise ValueError(
            f"Data validation FAILED for "
            f"{report.dataset} ({report.split}):\n{error_msgs}"
        )

    if halt_on_warning and report.warnings:
        warn_msgs = "\n".join(
            f"  [{w.name}] {w.message}" for w in report.warnings
        )
        raise ValueError(
            f"Data validation produced warnings for "
            f"{report.dataset} ({report.split}):\n{warn_msgs}"
        )


# Internal Helpers

def _log_report(report: ValidationReport) -> None:
    """Log the summary of a validation report."""
    summary = report.summary()
    if report.passed:
        logger.info(summary)
    else:
        logger.error(summary)