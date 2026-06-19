# src/utils/logger.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Centralised logging configuration for all modules.
#          Provides a single factory function that returns a
#          consistently formatted logger writing to both the
#          console and a rotating log file. Every module in
#          the project calls get_logger(__name__) to obtain
#          its logger — ensuring uniform timestamps, levels,
#          and output paths across the entire pipeline.

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from src.utils.constants import LOG_FORMAT, LOG_DATE_FORMAT, LOG_LEVEL


# Module-level registry so each named logger is only
# configured once — prevents duplicate handler attachment
# when a module is imported multiple times.
_CONFIGURED_LOGGERS: set = set()


def get_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = LOG_LEVEL,
    max_bytes: int = 10 * 1024 * 1024,   # 10 MB per file
    backup_count: int = 5,
) -> logging.Logger:
    """
    Return a named logger configured with a console handler and,
    optionally, a rotating file handler.

    If the logger with *name* has already been configured in this
    process, the existing logger is returned unchanged so that
    calling get_logger multiple times from the same module does
    not attach duplicate handlers.

    Parameters
    ----------
    name : str
        Logger name — use ``__name__`` in every calling module.
    log_file : str, optional
        Absolute or relative path to the log file.  Parent
        directories are created automatically.  When *None* the
        logger writes to the console only.
    level : str
        Logging level string — one of DEBUG, INFO, WARNING,
        ERROR, CRITICAL.  Defaults to the project constant
        LOG_LEVEL ("INFO").
    max_bytes : int
        Maximum size of a single log file before rotation.
        Defaults to 10 MB.
    backup_count : int
        Number of rotated backup files to retain.

    Returns
    -------
    logging.Logger
        Fully configured logger instance.

    Examples
    --------
    >>> from src.utils.logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Pipeline started.")
    """
    logger = logging.getLogger(name)

    # Return early if this logger was already fully configured.
    if name in _CONFIGURED_LOGGERS:
        return logger

    # Resolve numeric log level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Prevent propagation to the root logger to avoid duplicate
    # output when multiple modules are active simultaneously.
    logger.propagate = False

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # Console handler — always attached
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating file handler — attached only when log_file given
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            filename=str(log_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _CONFIGURED_LOGGERS.add(name)
    return logger


def get_pipeline_logger(log_dir: str = "reports/logs") -> logging.Logger:
    """
    Convenience factory that returns the project-wide pipeline
    logger writing to ``reports/logs/pipeline.log``.

    This logger is used by ``run_pipeline.py`` and top-level
    CLI scripts to record end-to-end pipeline progress.

    Parameters
    ----------
    log_dir : str
        Directory where ``pipeline.log`` will be created.

    Returns
    -------
    logging.Logger
        Configured pipeline logger.
    """
    log_file = os.path.join(log_dir, "pipeline.log")
    return get_logger("pipeline", log_file=log_file)


def get_training_logger(log_dir: str = "reports/logs") -> logging.Logger:
    """
    Convenience factory that returns the training logger writing
    to ``reports/logs/training.log``.

    Parameters
    ----------
    log_dir : str
        Directory where ``training.log`` will be created.

    Returns
    -------
    logging.Logger
        Configured training logger.
    """
    log_file = os.path.join(log_dir, "training.log")
    return get_logger("training", log_file=log_file)


def set_global_log_level(level: str) -> None:
    """
    Update the log level of every logger that has been created
    through this module.  Useful for switching to DEBUG mode
    at runtime (e.g. ``--verbose`` CLI flag).

    Parameters
    ----------
    level : str
        New log level string — DEBUG | INFO | WARNING | ERROR.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    for name in _CONFIGURED_LOGGERS:
        log = logging.getLogger(name)
        log.setLevel(numeric_level)
        for handler in log.handlers:
            handler.setLevel(numeric_level)


def log_section_header(logger: logging.Logger, title: str) -> None:
    """
    Emit a clearly visible section divider to the log.  Used by
    pipeline scripts to visually separate major stages in the
    log output and in Chapter 4 screenshots.

    Parameters
    ----------
    logger : logging.Logger
        Target logger.
    title : str
        Section title to display inside the divider.

    Examples
    --------
    >>> log_section_header(logger, "DATA PREPROCESSING")
    # Emits:
    # DATA PREPROCESSING
    """
    border = "=" * 60
    logger.info(border)
    logger.info(title.upper())
    logger.info(border)


def log_dict(logger: logging.Logger, data: dict, title: str = "") -> None:
    """
    Log the key-value pairs of a dictionary in a readable table
    format.  Used to display hyperparameters, dataset statistics,
    and evaluation metrics in the log output.

    Parameters
    ----------
    logger : logging.Logger
        Target logger.
    data : dict
        Dictionary to display.
    title : str, optional
        Optional heading printed before the table.
    """
    if title:
        logger.info(title)
    for key, value in data.items():
        logger.info("  %-35s %s", str(key), str(value))