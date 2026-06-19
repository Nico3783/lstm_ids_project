
# src/config/__init__.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Marks src/config/ as a Python package and exposes
#          the primary configuration accessor so every module
#          can obtain the live config object with:
#
#              from src.config import get_config, AppConfig

from src.config.settings import AppConfig, get_config, reload_config, override_dataset

__all__ = [
    "AppConfig",
    "get_config",
    "reload_config",
    "override_dataset",
]