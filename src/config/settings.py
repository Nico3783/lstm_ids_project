
# src/config/settings.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Parse ``config.yaml`` into a structured, typed
#          configuration object (AppConfig) and expose a
#          singleton ``get_config()`` accessor.
#
#          Every module that needs a hyperparameter, path
#          setting, or dataset option calls get_config() and
#          reads from the AppConfig dataclass rather than
#          accessing the YAML dict directly. This provides:
#            - Auto-completion in IDEs
#            - Type checking
#            - A single point of validation for all settings
#            - The ability to override settings at runtime
#              (e.g. from CLI flags or unit tests)

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Path to the project root config.yaml — resolved relative to
# this file's location (src/config/settings.py → ../../config.yaml)
_DEFAULT_CONFIG_PATH: Path = (
    Path(__file__).resolve().parent.parent.parent / "config.yaml"
)


# Nested Configuration Dataclasses

@dataclass
class SequenceConfig:
    """Sequence construction parameters (Chapter 3, Sec 3.5.2)."""
    window_size: int = 10
    step_size: int = 1
    label_position: str = "last"


@dataclass
class SplitConfig:
    """Train / validation / test split ratios (Chapter 3, Sec 3.5.2)."""
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    stratified: bool = True


@dataclass
class LSTMLayerConfig:
    """Configuration for a single LSTM layer."""
    units: int = 128
    return_sequences: bool = True
    dropout: float = 0.2
    activation: str = "tanh"
    recurrent_activation: str = "sigmoid"


@dataclass
class DenseLayerConfig:
    """Configuration for a Dense hidden layer."""
    units: int = 32
    activation: str = "relu"
    l2_regularization: float = 0.001
    batch_normalization: bool = True


@dataclass
class ModelConfig:
    """
    LSTM model architecture and compilation settings.
    Chapter 3, Section 3.5.3.
    """
    type: str = "lstm"
    lstm_layers: List[LSTMLayerConfig] = field(default_factory=lambda: [
        LSTMLayerConfig(units=128, return_sequences=True,  dropout=0.2),
        LSTMLayerConfig(units=64,  return_sequences=False, dropout=0.2),
    ])
    dense_layers: List[DenseLayerConfig] = field(default_factory=lambda: [
        DenseLayerConfig(units=32, activation="relu",
                         l2_regularization=0.001, batch_normalization=True),
    ])
    output_activation: str = "softmax"
    optimizer: str = "adam"
    learning_rate: float = 0.001
    loss: str = "categorical_crossentropy"
    metrics: List[str] = field(
        default_factory=lambda: ["accuracy", "Precision", "Recall"]
    )
    # Dataset-specific model overrides — when present for the
    # active dataset, these override the base architecture.
    # Example: model_overrides.cicids2017.lstm_layers → replaces
    # the default lstm_layers when active_dataset == "cicids2017".
    model_overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    monitor: str = "val_loss"
    patience: int = 10
    restore_best_weights: bool = True
    min_delta: float = 0.0001


@dataclass
class ModelCheckpointConfig:
    enabled: bool = True
    monitor: str = "val_loss"
    save_best_only: bool = True
    filepath: str = "models/checkpoints/best_model.keras"


@dataclass
class ReduceLRConfig:
    enabled: bool = True
    monitor: str = "val_loss"
    factor: float = 0.5
    patience: int = 5
    min_lr: float = 1e-6


@dataclass
class TrainingConfig:
    """
    Training loop configuration.
    Chapter 3, Section 3.5.4.
    """
    epochs: int = 100
    batch_size: int = 64
    early_stopping: EarlyStoppingConfig = field(
        default_factory=EarlyStoppingConfig
    )
    model_checkpoint: ModelCheckpointConfig = field(
        default_factory=ModelCheckpointConfig
    )
    reduce_lr: ReduceLRConfig = field(default_factory=ReduceLRConfig)
    tensorboard_log_dir: str = "reports/logs/tensorboard"
    csv_logger_filepath: str = "reports/logs/training_history.csv"


@dataclass
class HyperparameterTuningConfig:
    """
    Grid search configuration.
    Chapter 3, Section 3.5.4.
    """
    enabled: bool = False
    method: str = "grid_search"
    n_trials: int = 50
    n_lstm_layers: List[int] = field(default_factory=lambda: [1, 2, 3])
    lstm_units: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    dropout_rate: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.5]
    )
    learning_rate: List[float] = field(
        default_factory=lambda: [0.01, 0.001, 0.0001]
    )
    batch_size: List[int] = field(default_factory=lambda: [32, 64, 128])
    objective: str = "val_accuracy"
    direction: str = "maximize"


@dataclass
class PreprocessingConfig:
    """
    Data cleaning, encoding, and scaling settings.
    Chapter 3, Section 3.5.2.
    """
    handle_missing: bool = True
    missing_strategy_continuous: str = "mean"
    missing_strategy_categorical: str = "mode"
    remove_duplicates: bool = True
    handle_infinite: bool = True
    encoding_method: str = "onehot"
    drop_first: bool = False
    scaler: str = "minmax"
    scaler_feature_range: Tuple[float, float] = (0.0, 1.0)
    use_class_weights: bool = True
    use_smote: bool = False


@dataclass
class EvaluationConfig:
    """Evaluation metrics and reporting settings."""
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "precision_macro", "precision_weighted",
        "recall_macro", "recall_weighted",
        "f1_macro", "f1_weighted", "roc_auc", "confusion_matrix",
    ])
    averaging: List[str] = field(default_factory=lambda: ["macro", "weighted"])
    roc_curve: bool = True
    precision_recall_curve: bool = True
    permutation_importance: bool = True
    n_permutation_repeats: int = 10


@dataclass
class VisualizationConfig:
    """Plot styling and output settings."""
    dpi: int = 300
    figure_size: Tuple[int, int] = (12, 8)
    style: str = "seaborn-v0_8-whitegrid"
    color_palette: str = "husl"
    save_format: str = "png"
    font_size: int = 12
    title_font_size: int = 14


@dataclass
class DatasetConfig:
    """Active dataset selection and per-dataset parameters."""
    active: str = "nsl_kdd"
    nsl_kdd_train_file: str = "data/raw/nsl_kdd/KDDTrain+.txt"
    nsl_kdd_test_file: str = "data/raw/nsl_kdd/KDDTest+.txt"
    nsl_kdd_train_20pct_file: str = "data/raw/nsl_kdd/KDDTrain+_20Percent.txt"
    nsl_kdd_field_names_file: str = "data/raw/nsl_kdd/field_names.csv"


@dataclass
class ProjectConfig:
    """Top-level project metadata."""
    name: str = "Deep Learning IDS Using LSTM"
    version: str = "1.0.0"
    author: str = "FUTA Cybersecurity Final Year Project"
    institution: str = "Federal University of Technology, Akure"
    seed: int = 42


# Root Application Configuration

@dataclass
class AppConfig:
    """
    Root configuration object holding all sub-configs.

    Obtain the singleton instance through ``get_config()``.
    Do not instantiate this class directly in application code.

    Attributes
    ----------
    project : ProjectConfig
    dataset : DatasetConfig
    preprocessing : PreprocessingConfig
    sequence : SequenceConfig
    split : SplitConfig
    model : ModelConfig
    training : TrainingConfig
    hyperparameter_tuning : HyperparameterTuningConfig
    evaluation : EvaluationConfig
    visualization : VisualizationConfig
    raw : dict
        The raw parsed YAML dictionary, available for any
        setting not yet promoted to a typed field.
    """
    project: ProjectConfig = field(default_factory=ProjectConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(
        default_factory=PreprocessingConfig
    )
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hyperparameter_tuning: HyperparameterTuningConfig = field(
        default_factory=HyperparameterTuningConfig
    )
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    visualization: VisualizationConfig = field(
        default_factory=VisualizationConfig
    )
    raw: Dict[str, Any] = field(default_factory=dict)

    # Convenience property accessors

    @property
    def active_dataset(self) -> str:
        """Return the currently selected dataset identifier."""
        return self.dataset.active

    @property
    def window_size(self) -> int:
        """Sliding window size for sequence construction."""
        return self.sequence.window_size

    @property
    def n_classes(self) -> int:
        """
        Return the number of target classes for the active dataset.
        NSL-KDD: 5, CICIDS2017: dynamic, UNSW-NB15: 10.
        """
        from src.utils.constants import (
            NSL_KDD_NUM_CLASSES,
            UNSW_NB15_ATTACK_CATEGORIES,
        )
        mapping = {
            "nsl_kdd": NSL_KDD_NUM_CLASSES,
            "cicids2017": 15,
            "unsw_nb15": len(UNSW_NB15_ATTACK_CATEGORIES),
        }
        return mapping.get(self.active_dataset, 5)

    @property
    def seed(self) -> int:
        """Global random seed."""
        return self.project.seed


# YAML → AppConfig Parser

def _parse_config(raw: Dict[str, Any]) -> AppConfig:
    """
    Convert a raw YAML dictionary into a typed ``AppConfig``.

    Parameters
    ----------
    raw : dict
        Parsed YAML content from ``config.yaml``.

    Returns
    -------
    AppConfig
    """
    cfg = AppConfig(raw=raw)

    # --- Project ---
    p = raw.get("project", {})
    cfg.project = ProjectConfig(
        name=p.get("name", cfg.project.name),
        version=p.get("version", cfg.project.version),
        author=p.get("author", cfg.project.author),
        institution=p.get("institution", cfg.project.institution),
        seed=p.get("seed", cfg.project.seed),
    )

    # --- Dataset ---
    d = raw.get("dataset", {})
    nsl = d.get("nsl_kdd", {})
    cfg.dataset = DatasetConfig(
        active=d.get("active", cfg.dataset.active),
        nsl_kdd_train_file=nsl.get("train_file", cfg.dataset.nsl_kdd_train_file),
        nsl_kdd_test_file=nsl.get("test_file", cfg.dataset.nsl_kdd_test_file),
        nsl_kdd_train_20pct_file=nsl.get(
            "train_20pct_file", cfg.dataset.nsl_kdd_train_20pct_file
        ),
        nsl_kdd_field_names_file=nsl.get(
            "field_names_file", cfg.dataset.nsl_kdd_field_names_file
        ),
    )

    # --- Preprocessing ---
    pr = raw.get("preprocessing", {})
    fr = pr.get("scaler_feature_range", [0.0, 1.0])
    cfg.preprocessing = PreprocessingConfig(
        handle_missing=pr.get("handle_missing", True),
        missing_strategy_continuous=pr.get(
            "missing_strategy_continuous", "mean"
        ),
        missing_strategy_categorical=pr.get(
            "missing_strategy_categorical", "mode"
        ),
        remove_duplicates=pr.get("remove_duplicates", True),
        handle_infinite=pr.get("handle_infinite", True),
        encoding_method=pr.get("encoding_method", "onehot"),
        drop_first=pr.get("drop_first", False),
        scaler=pr.get("scaler", "minmax"),
        scaler_feature_range=(float(fr[0]), float(fr[1])),
        use_class_weights=pr.get("use_class_weights", True),
        use_smote=pr.get("use_smote", False),
    )

    # --- Sequence ---
    sq = raw.get("sequence", {})
    cfg.sequence = SequenceConfig(
        window_size=sq.get("window_size", 10),
        step_size=sq.get("step_size", 1),
        label_position=sq.get("label_position", "last"),
    )

    # --- Split ---
    sp = raw.get("split", {})
    cfg.split = SplitConfig(
        train_ratio=sp.get("train_ratio", 0.70),
        val_ratio=sp.get("val_ratio", 0.15),
        test_ratio=sp.get("test_ratio", 0.15),
        stratified=sp.get("stratified", True),
    )

    # --- Model ---
    m = raw.get("model", {})
    arch = m.get("architecture", {})
    lstm_raw = arch.get("lstm_layers", [])
    dense_raw = arch.get("dense_layers", [])

    lstm_layers = [
        LSTMLayerConfig(
            units=l.get("units", 128),
            return_sequences=l.get("return_sequences", True),
            dropout=l.get("dropout", 0.2),
            activation=l.get("activation", "tanh"),
            recurrent_activation=l.get("recurrent_activation", "sigmoid"),
        )
        for l in lstm_raw
    ] or cfg.model.lstm_layers

    dense_layers = [
        DenseLayerConfig(
            units=d_layer.get("units", 32),
            activation=d_layer.get("activation", "relu"),
            l2_regularization=d_layer.get("l2_regularization", 0.001),
            batch_normalization=d_layer.get("batch_normalization", True),
        )
        for d_layer in dense_raw
    ] or cfg.model.dense_layers

    cfg.model = ModelConfig(
        type=m.get("type", "lstm"),
        lstm_layers=lstm_layers,
        dense_layers=dense_layers,
        output_activation=arch.get("output_activation", "softmax"),
        optimizer=m.get("optimizer", "adam"),
        learning_rate=m.get("learning_rate", 0.001),
        loss=m.get("loss", "categorical_crossentropy"),
        metrics=m.get("metrics", ["accuracy", "Precision", "Recall"]),
    )

    # --- Training ---
    tr = raw.get("training", {})
    es = tr.get("early_stopping", {})
    mc = tr.get("model_checkpoint", {})
    rl = tr.get("reduce_lr", {})
    tb = tr.get("tensorboard", {})
    cl = tr.get("csv_logger", {})

    cfg.training = TrainingConfig(
        epochs=tr.get("epochs", 100),
        batch_size=tr.get("batch_size", 64),
        early_stopping=EarlyStoppingConfig(
            enabled=es.get("enabled", True),
            monitor=es.get("monitor", "val_loss"),
            patience=es.get("patience", 10),
            restore_best_weights=es.get("restore_best_weights", True),
            min_delta=es.get("min_delta", 0.0001),
        ),
        model_checkpoint=ModelCheckpointConfig(
            enabled=mc.get("enabled", True),
            monitor=mc.get("monitor", "val_loss"),
            save_best_only=mc.get("save_best_only", True),
            filepath=mc.get("filepath", "models/checkpoints/best_model.keras"),
        ),
        reduce_lr=ReduceLRConfig(
            enabled=rl.get("enabled", True),
            monitor=rl.get("monitor", "val_loss"),
            factor=rl.get("factor", 0.5),
            patience=rl.get("patience", 5),
            min_lr=rl.get("min_lr", 1e-6),
        ),
        tensorboard_log_dir=tb.get("log_dir", "reports/logs/tensorboard"),
        csv_logger_filepath=cl.get(
            "filepath", "reports/logs/training_history.csv"
        ),
    )

    # --- Hyperparameter Tuning ---
    ht = raw.get("hyperparameter_tuning", {})
    ss = ht.get("search_space", {})
    cfg.hyperparameter_tuning = HyperparameterTuningConfig(
        enabled=ht.get("enabled", False),
        method=ht.get("method", "grid_search"),
        n_trials=ht.get("n_trials", 50),
        n_lstm_layers=ss.get("n_lstm_layers", [1, 2, 3]),
        lstm_units=ss.get("lstm_units", [32, 64, 128, 256]),
        dropout_rate=ss.get("dropout_rate", [0.1, 0.2, 0.3, 0.5]),
        learning_rate=ss.get("learning_rate", [0.01, 0.001, 0.0001]),
        batch_size=ss.get("batch_size", [32, 64, 128]),
        objective=ht.get("objective", "val_accuracy"),
        direction=ht.get("direction", "maximize"),
    )

    # --- Evaluation ---
    ev = raw.get("evaluation", {})
    cfg.evaluation = EvaluationConfig(
        metrics=ev.get("metrics", cfg.evaluation.metrics),
        averaging=ev.get("averaging", ["macro", "weighted"]),
        roc_curve=ev.get("roc_curve", True),
        precision_recall_curve=ev.get("precision_recall_curve", True),
        permutation_importance=ev.get("permutation_importance", True),
        n_permutation_repeats=ev.get("n_permutation_repeats", 10),
    )

    # --- Visualization ---
    vi = raw.get("visualization", {})
    fig_size = vi.get("figure_size", [12, 8])
    cfg.visualization = VisualizationConfig(
        dpi=vi.get("dpi", 300),
        figure_size=(int(fig_size[0]), int(fig_size[1])),
        style=vi.get("style", "seaborn-v0_8-whitegrid"),
        color_palette=vi.get("color_palette", "husl"),
        save_format=vi.get("save_format", "png"),
        font_size=vi.get("font_size", 12),
        title_font_size=vi.get("title_font_size", 14),
    )

    return cfg


# Singleton Management

_CONFIG_INSTANCE: Optional[AppConfig] = None


def get_config(
    config_path: Optional[Union[str, Path]] = None,
) -> AppConfig:
    """
    Return the singleton ``AppConfig`` instance.

    On the first call the YAML file is parsed and the
    singleton is cached.  Subsequent calls return the cached
    instance, so the YAML file is read only once per process.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to the YAML config file.  Defaults to the
        project-root ``config.yaml`` located two directories
        above this file.

    Returns
    -------
    AppConfig
        Fully populated configuration object.

    Examples
    --------
    >>> from src.config import get_config
    >>> cfg = get_config()
    >>> cfg.training.batch_size
    64
    >>> cfg.sequence.window_size
    10
    >>> cfg.model.learning_rate
    0.001
    """
    global _CONFIG_INSTANCE
    if _CONFIG_INSTANCE is None:
        _CONFIG_INSTANCE = _load_config(config_path)
    return _CONFIG_INSTANCE


def reload_config(
    config_path: Optional[Union[str, Path]] = None,
) -> AppConfig:
    """
    Force a fresh parse of the YAML file, replacing the
    cached singleton.

    Useful in unit tests and when the config file is modified
    between pipeline stages.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to the YAML config file.

    Returns
    -------
    AppConfig
        Freshly loaded configuration object.
    """
    global _CONFIG_INSTANCE
    _CONFIG_INSTANCE = None
    return get_config(config_path)


def _load_config(
    config_path: Optional[Union[str, Path]] = None,
) -> AppConfig:
    """
    Parse ``config.yaml`` and return a populated ``AppConfig``.

    Parameters
    ----------
    config_path : str or Path, optional
        Explicit path override.  Falls back to the environment
        variable ``IDS_CONFIG_PATH`` then the default project-
        root ``config.yaml``.

    Returns
    -------
    AppConfig

    Raises
    ------
    FileNotFoundError
        If the config file cannot be located.
    yaml.YAMLError
        If the YAML content is malformed.
    """
    # Resolve path: argument > env var > default
    if config_path is not None:
        path = Path(config_path)
    elif "IDS_CONFIG_PATH" in os.environ:
        path = Path(os.environ["IDS_CONFIG_PATH"])
    else:
        path = _DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}\n"
            "Ensure config.yaml is present in the project root directory."
        )

    logger.info("Loading configuration from: %s", path)

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if raw is None:
        raise ValueError(
            f"Configuration file is empty or invalid: {path}"
        )

    config = _parse_config(raw)

    logger.info(
        "Configuration loaded — dataset: '%s' | window: %d | "
        "epochs: %d | batch: %d | lr: %s",
        config.active_dataset,
        config.window_size,
        config.training.epochs,
        config.training.batch_size,
        config.model.learning_rate,
    )
    return config


# Runtime Override Helpers

def override_dataset(dataset: str) -> AppConfig:
    """
    Switch the active dataset on the live config singleton
    without reloading the YAML file.

    Called by CLI scripts that accept a ``--dataset`` argument.

    Parameters
    ----------
    dataset : str
        Dataset identifier: ``nsl_kdd`` | ``cicids2017`` |
        ``unsw_nb15``.

    Returns
    -------
    AppConfig
        The updated singleton.

    Raises
    ------
    ValueError
        If *dataset* is not a recognised identifier.
    """
    from src.utils.constants import SUPPORTED_DATASETS

    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Supported: {SUPPORTED_DATASETS}"
        )
    cfg = get_config()
    cfg.dataset.active = dataset
    logger.info("Active dataset overridden to: '%s'", dataset)
    return cfg


def override_hyperparameters(
    learning_rate: Optional[float] = None,
    batch_size: Optional[int] = None,
    epochs: Optional[int] = None,
    window_size: Optional[int] = None,
) -> AppConfig:
    """
    Override specific training hyperparameters on the live
    singleton at runtime.

    Used by hyperparameter tuning loops to inject candidate
    values without modifying ``config.yaml``.

    Parameters
    ----------
    learning_rate : float, optional
    batch_size : int, optional
    epochs : int, optional
    window_size : int, optional

    Returns
    -------
    AppConfig
        The updated singleton.
    """
    cfg = get_config()
    if learning_rate is not None:
        cfg.model.learning_rate = learning_rate
        logger.debug("LR overridden to %s", learning_rate)
    if batch_size is not None:
        cfg.training.batch_size = batch_size
        logger.debug("Batch size overridden to %d", batch_size)
    if epochs is not None:
        cfg.training.epochs = epochs
        logger.debug("Epochs overridden to %d", epochs)
    if window_size is not None:
        cfg.sequence.window_size = window_size
        logger.debug("Window size overridden to %d", window_size)
    return cfg
