
# src/training/__init__.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Marks src/training/ as a Python package and
#          exposes the primary training functions and classes.

from src.training.callbacks import (
    build_callbacks,
    build_callbacks_from_config,
    EpochProgressLogger,
)

from src.training.class_weights import (
    get_class_weights,
    weights_to_sample_weights,
)

from src.training.trainer import (
    TrainingResult,
    train_lstm,
    train_baselines,
    run_full_training,
)

from src.training.hyperparameter_tuning import (
    TrialResult,
    generate_grid_search_space,
    generate_random_search_space,
    run_grid_search,
    run_random_search,
    run_trial,
)

__all__ = [
    # callbacks
    "build_callbacks",
    "build_callbacks_from_config",
    "EpochProgressLogger",
    # class weights
    "get_class_weights",
    "weights_to_sample_weights",
    # trainer
    "TrainingResult",
    "train_lstm",
    "train_baselines",
    "run_full_training",
    # hyperparameter tuning
    "TrialResult",
    "generate_grid_search_space",
    "generate_random_search_space",
    "run_grid_search",
    "run_random_search",
    "run_trial",
]