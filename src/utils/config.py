# src/utils/config.py

"""Lightweight YAML config loader returning plain dicts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML configuration file and return it as a nested dict.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed configuration.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    yaml.YAMLError
        If the YAML content is malformed.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if raw is None:
        raise ValueError(f"Config file is empty: {path}")

    return raw
