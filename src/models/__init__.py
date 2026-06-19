
# src/models/__init__.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Marks src/models/ as a Python package and exposes
#          the primary model creation and utility functions.

from src.models.lstm_model import (
    build_lstm_model,
    build_rnn_model,
    get_model_summary_string,
    log_model_summary,
    save_model_summary,
    get_model_config_dict,
)

from src.models.baseline_models import (
    build_random_forest,
    build_svm,
    build_logistic_regression,
    train_baseline,
    train_all_baselines,
    predict_baseline,
    save_all_baselines,
    load_all_baselines,
)

from src.models.model_factory import create_model

from src.models.model_utils import (
    count_parameters,
    build_model_metadata,
    save_model_metadata,
    plot_lstm_architecture,
    list_checkpoints,
    get_best_checkpoint,
    cleanup_old_checkpoints,
)

__all__ = [
    # lstm_model
    "build_lstm_model", "build_rnn_model",
    "get_model_summary_string", "log_model_summary",
    "save_model_summary", "get_model_config_dict",
    # baseline_models
    "build_random_forest", "build_svm", "build_logistic_regression",
    "train_baseline", "train_all_baselines",
    "predict_baseline", "save_all_baselines", "load_all_baselines",
    # model_factory
    "create_model",
    # model_utils
    "count_parameters", "build_model_metadata", "save_model_metadata",
    "plot_lstm_architecture", "list_checkpoints",
    "get_best_checkpoint", "cleanup_old_checkpoints",
]