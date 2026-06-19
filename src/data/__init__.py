
# src/data/__init__.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Marks src/data/ as a Python package and exposes
#          the public API of all data pipeline submodules —
#          download, validation, loading, EDA, preprocessing,
#          feature engineering, sequence building, and
#          splitting — under a single import namespace.

# --- Acquisition & Validation ---
from src.data.download import (
    acquire_dataset,
    acquire_all_datasets,
    validate_dataset,
    check_dataset_availability,
    download_nsl_kdd,
    validate_nsl_kdd,
    validate_cicids2017,
    validate_unsw_nb15,
    print_cicids2017_instructions,
    print_unsw_nb15_instructions,
    print_nsl_kdd_manual_instructions,
)

# --- Data Quality Validators ---
from src.data.validators import (
    validate_dataframe,
    validate_nsl_kdd_dataframe,
    validate_cicids2017_dataframe,
    validate_unsw_nb15_dataframe,
    validate_processed_arrays,
    assert_validation_passed,
    ValidationReport,
    CheckResult,
)

# --- Loaders ---
from src.data.loaders import (
    load_dataset,
    load_nsl_kdd,
    load_cicids2017,
    load_unsw_nb15,
    get_dataset_summary,
    load_nsl_kdd_train_test,
    load_nsl_kdd_merged,
)

# --- Exploratory Data Analysis ---
from src.data.exploratory import (
    run_eda,
    describe_dataset,
    plot_class_distribution,
    plot_missing_values,
    plot_correlation_heatmap,
    plot_feature_distributions,
    generate_dataset_summary_table,
)

# --- Preprocessing ---
from src.data.preprocessing import (
    preprocess_dataset,
    preprocess_train_val_test,
    preprocess_new_data,
    drop_irrelevant_columns,
    handle_missing_and_infinite,
    remove_duplicates,
    map_nsl_kdd_labels,
    encode_categorical_features,
    separate_features_target,
    fit_scaler,
    apply_scaler,
    apply_smote,
)

# --- Feature Engineering ---
from src.data.feature_engineering import (
    remove_zero_variance_features,
    remove_highly_correlated_features,
    compute_permutation_importance,
    plot_feature_importance,
    get_feature_engineering_summary,
)

# --- Sequence Builder ---
from src.data.sequence_builder import (
    build_sequences,
    build_sequences_chunked,
    sequence_generator,
    rebuild_sequences_from_flat,
    shuffle_sequences,
    get_sequence_stats,
    estimate_sequence_count,
    estimate_memory_mb,
)

# --- Train / Val / Test Split ---
from src.data.split import (
    split_sequences,
    split_and_save,
    compute_split_class_weights,
    get_split_summary,
    save_split_summary,
    stratified_kfold_splits,
)

__all__ = [
    # download
    "acquire_dataset", "acquire_all_datasets",
    "validate_dataset", "check_dataset_availability",
    "download_nsl_kdd",
    "validate_nsl_kdd", "validate_cicids2017", "validate_unsw_nb15",
    "print_cicids2017_instructions", "print_unsw_nb15_instructions",
    "print_nsl_kdd_manual_instructions",
    # validators
    "validate_dataframe",
    "validate_nsl_kdd_dataframe", "validate_cicids2017_dataframe",
    "validate_unsw_nb15_dataframe", "validate_processed_arrays",
    "assert_validation_passed", "ValidationReport", "CheckResult",
    # loaders
    "load_dataset", "load_nsl_kdd", "load_cicids2017", "load_unsw_nb15",
    "get_dataset_summary", "load_nsl_kdd_train_test", "load_nsl_kdd_merged",
    # eda
    "run_eda", "describe_dataset",
    "plot_class_distribution", "plot_missing_values",
    "plot_correlation_heatmap", "plot_feature_distributions",
    "generate_dataset_summary_table",
    # preprocessing
    "preprocess_dataset", "preprocess_train_val_test",
    "preprocess_new_data", "drop_irrelevant_columns",
    "handle_missing_and_infinite", "remove_duplicates",
    "map_nsl_kdd_labels", "encode_categorical_features",
    "separate_features_target", "fit_scaler", "apply_scaler", "apply_smote",
    # feature engineering
    "remove_zero_variance_features", "remove_highly_correlated_features",
    "compute_permutation_importance", "plot_feature_importance",
    "get_feature_engineering_summary",
    # sequence builder
    "build_sequences", "build_sequences_chunked",
    "sequence_generator", "rebuild_sequences_from_flat",
    "shuffle_sequences", "get_sequence_stats",
    "estimate_sequence_count", "estimate_memory_mb",
    # split
    "split_sequences", "split_and_save",
    "compute_split_class_weights", "get_split_summary",
    "save_split_summary", "stratified_kfold_splits",
]