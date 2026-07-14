
# src/utils/constants.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Project-wide constants — dataset field names,
#          label mappings, feature lists, and fixed values
#          used across all modules. All values are derived
#          directly from Chapter 3 of the project report.

from typing import Dict, List, Tuple

# Reproducibility
RANDOM_SEED: int = 42

# NSL-KDD Dataset Constants
# Chapter 3, Section 3.3.1
# The 41 original features + label + difficulty columns

NSL_KDD_COLUMNS: List[str] = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty",
]

# Categorical features requiring one-hot encoding (Chapter 3, Sec 3.5.2)
NSL_KDD_CATEGORICAL_FEATURES: List[str] = [
    "protocol_type",
    "service",
    "flag",
]

# Numerical features (continuous) — all except categorical + target
NSL_KDD_NUMERICAL_FEATURES: List[str] = [
    col for col in NSL_KDD_COLUMNS
    if col not in NSL_KDD_CATEGORICAL_FEATURES + ["label", "difficulty"]
]

# Target column
NSL_KDD_TARGET_COLUMN: str = "label"
NSL_KDD_DIFFICULTY_COLUMN: str = "difficulty"

# Attack-type to 5-class category mapping
# Chapter 3, Section 3.5.2 — Label Encoding
# 0=Normal, 1=DoS, 2=Probe, 3=R2L, 4=U2R
NSL_KDD_ATTACK_TO_CATEGORY: Dict[str, str] = {
    "normal": "normal",
    # DoS attacks
    "back": "dos",
    "land": "dos",
    "neptune": "dos",
    "pod": "dos",
    "smurf": "dos",
    "teardrop": "dos",
    "apache2": "dos",
    "mailbomb": "dos",
    "processtable": "dos",
    "udpstorm": "dos",
    # Probe attacks
    "ipsweep": "probe",
    "nmap": "probe",
    "portsweep": "probe",
    "satan": "probe",
    "mscan": "probe",
    "saint": "probe",
    # R2L attacks
    "ftp_write": "r2l",
    "guess_passwd": "r2l",
    "imap": "r2l",
    "multihop": "r2l",
    "phf": "r2l",
    "spy": "r2l",
    "warezclient": "r2l",
    "warezmaster": "r2l",
    "sendmail": "r2l",
    "named": "r2l",
    "snmpgetattack": "r2l",
    "snmpguess": "r2l",
    "xlock": "r2l",
    "xsnoop": "r2l",
    "httptunnel": "r2l",
    # U2R attacks
    "buffer_overflow": "u2r",
    "loadmodule": "u2r",
    "perl": "u2r",
    "rootkit": "u2r",
    "ps": "u2r",
    "sqlattack": "u2r",
    "xterm": "u2r",
    "worm": "u2r",
}

# Category to integer label mapping (Chapter 3, Section 3.5.2)
NSL_KDD_CATEGORY_TO_INT: Dict[str, int] = {
    "normal": 0,
    "dos": 1,
    "probe": 2,
    "r2l": 3,
    "u2r": 4,
}

# Integer to category label mapping (inverse)
NSL_KDD_INT_TO_CATEGORY: Dict[int, str] = {
    v: k for k, v in NSL_KDD_CATEGORY_TO_INT.items()
}

# Human-readable class names for plots and reports
NSL_KDD_CLASS_NAMES: List[str] = ["Normal", "DoS", "Probe", "R2L", "U2R"]

# Number of classes in NSL-KDD
NSL_KDD_NUM_CLASSES: int = 5

# Number of features after preprocessing + one-hot encoding (Chapter 3)
NSL_KDD_NUM_FEATURES: int = 41

# CICIDS2017 Dataset Constants
# Chapter 3, Section 3.3.1
# 80 bidirectional flow features

CICIDS2017_TARGET_COLUMN: str = " Label"      # Leading space in raw files
CICIDS2017_BENIGN_LABEL: str = "BENIGN"
CICIDS2017_NUM_FEATURES: int = 80

# Attack categories present in CICIDS2017
CICIDS2017_ATTACK_CATEGORIES: List[str] = [
    "BENIGN",
    "DoS Hulk",
    "PortScan",
    "DDoS",
    "DoS GoldenEye",
    "FTP-Patator",
    "SSH-Patator",
    "DoS slowloris",
    "DoS Slowhttptest",
    "Bot",
    "Web Attack – Brute Force",
    "Web Attack – XSS",
    "Infiltration",
    "Web Attack – Sql Injection",
    "Heartbleed",
]

# Human-readable class names in integer-mapping order
# (BENIGN=0, then alphabetical — matches map_cicids2017_labels)
CICIDS2017_CLASS_NAMES: List[str] = [
    "BENIGN",                      # 0
    "Bot",                         # 1
    "DDoS",                        # 2
    "DoS GoldenEye",               # 3
    "DoS Hulk",                    # 4
    "DoS Slowhttptest",            # 5
    "DoS slowloris",               # 6
    "FTP-Patator",                 # 7
    "Heartbleed",                  # 8
    "Infiltration",                # 9
    "PortScan",                    # 10
    "SSH-Patator",                 # 11
    "Web Attack – Brute Force",    # 12
    "Web Attack – Sql Injection",  # 13
    "Web Attack – XSS",            # 14
]

# Columns with known infinite/NaN issues in CICIDS2017
CICIDS2017_PROBLEMATIC_COLUMNS: List[str] = [
    "Flow Bytes/s",
    "Flow Packets/s",
]

# UNSW-NB15 Dataset Constants
# Chapter 3, Section 3.3.1
# 49 features, 9 attack families

UNSW_NB15_TARGET_COLUMN: str = "attack_cat"
UNSW_NB15_BINARY_LABEL_COLUMN: str = "label"
UNSW_NB15_NUM_FEATURES: int = 49

# Nine attack families in UNSW-NB15
UNSW_NB15_ATTACK_CATEGORIES: List[str] = [
    "Normal",
    "Fuzzers",
    "Analysis",
    "Backdoors",
    "DoS",
    "Exploits",
    "Generic",
    "Reconnaissance",
    "Shellcode",
    "Worms",
]

# Coarse-grained class grouping for UNSW-NB15
# Maps 10 fine-grained classes → 6 broader categories
# NOTE: map_unsw_nb15_labels() sorts alphabetically with "normal" forced to 0,
# so numeric indices are: 0=normal, 1=analysis, 2=backdoors, 3=dos,
# 4=exploits, 5=fuzzers, 6=generic, 7=reconnaissance, 8=shellcode, 9=worms.
# Grouping rationale (based on actual indices):
#   0 (Normal)      → 0 Normal
#   1 (Analysis)    → 1 Fuzzers   — port scan / traffic analysis
#   2 (Backdoors)   → 2 Exploits  — unauthorized access
#   3 (DoS)         → 3 DoS       — denial of service
#   4 (Exploits)    → 2 Exploits  — code execution
#   5 (Fuzzers)     → 1 Fuzzers   — input fuzzing
#   6 (Generic)     → 4 Generic   — benign-like bulk traffic
#   7 (Recon)       → 5 Recon     — network reconnaissance
#   8 (Shellcode)   → 2 Exploits  — code execution
#   9 (Worms)       → 2 Exploits  — self-propagating
UNSW_NB15_COARSE_MAPPING: Dict[int, int] = {
    0: 0,   # Normal → Normal
    1: 1,   # Analysis → Fuzzers
    2: 2,   # Backdoors → Exploits
    3: 3,   # DoS → DoS
    4: 2,   # Exploits → Exploits
    5: 1,   # Fuzzers → Fuzzers
    6: 4,   # Generic → Generic
    7: 5,   # Reconnaissance → Recon
    8: 2,   # Shellcode → Exploits
    9: 2,   # Worms → Exploits
}

UNSW_NB15_COARSE_CLASS_NAMES: List[str] = [
    "Normal",
    "Fuzzers",
    "Exploits",
    "DoS",
    "Generic",
    "Reconnaissance",
]

UNSW_NB15_CATEGORICAL_FEATURES: List[str] = [
    "proto",
    "service",
    "state",
]

# Sequence Builder Constants
# Chapter 3, Section 3.5.2 — Sequence Construction

DEFAULT_WINDOW_SIZE: int = 10      # Sliding window of 10 consecutive records
DEFAULT_STEP_SIZE: int = 1         # Step size for sliding window
LABEL_POSITION: str = "last"       # Label corresponds to final timestep

# Data Split Ratios
# Chapter 3, Section 3.5.2 — 70/15/15 stratified split

TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

# LSTM Model Architecture Constants
# Chapter 3, Section 3.5.3

LSTM_LAYER_1_UNITS: int = 128
LSTM_LAYER_2_UNITS: int = 64
DENSE_UNITS: int = 32
DROPOUT_RATE: float = 0.2
L2_LAMBDA: float = 0.001

# Training Constants
# Chapter 3, Section 3.5.4

DEFAULT_LEARNING_RATE: float = 0.001
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_EPOCHS: int = 100
EARLY_STOPPING_PATIENCE: int = 10
REDUCE_LR_PATIENCE: int = 5
REDUCE_LR_FACTOR: float = 0.5
MIN_LEARNING_RATE: float = 1e-6

# Hyperparameter Search Grid
# Chapter 3, Section 3.5.4 — Grid search configuration

HP_N_LSTM_LAYERS: List[int] = [1, 2, 3]
HP_LSTM_UNITS: List[int] = [32, 64, 128, 256]
HP_DROPOUT_RATES: List[float] = [0.1, 0.2, 0.3, 0.5]
HP_LEARNING_RATES: List[float] = [0.01, 0.001, 0.0001]
HP_BATCH_SIZES: List[int] = [32, 64, 128]

# Evaluation Constants
# Chapter 3, Section 3.5.5

EVALUATION_AVERAGING_MODES: List[str] = ["macro", "weighted"]
N_PERMUTATION_REPEATS: int = 10

# File Name Constants — Saved Artifacts

# Model files
BEST_MODEL_KERAS: str = "best_model.keras"
BEST_MODEL_H5: str = "best_model.h5"
FINAL_MODEL_KERAS: str = "lstm_ids_model.keras"
FINAL_MODEL_H5: str = "lstm_ids_model.h5"
MODEL_METADATA_JSON: str = "model_metadata.json"

# Preprocessing artifacts
LABEL_ENCODER_PKL: str = "label_encoder.pkl"
SCALER_PKL: str = "scaler.pkl"
FEATURE_NAMES_PKL: str = "feature_names.pkl"
METADATA_JSON: str = "metadata.json"

# Baseline model files
RANDOM_FOREST_PKL: str = "random_forest.pkl"
SVM_PKL: str = "svm.pkl"
LOGISTIC_REGRESSION_PKL: str = "logistic_regression.pkl"
BASELINE_RESULTS_JSON: str = "baseline_results.json"

# Data files
MERGED_DATASET_CSV: str = "merged_dataset.csv"
CLEANED_DATASET_CSV: str = "cleaned_dataset.csv"
ENCODED_DATASET_CSV: str = "encoded_dataset.csv"
SCALED_DATASET_CSV: str = "scaled_dataset.csv"

# Numpy processed arrays
X_TRAIN_NPY: str = "X_train.npy"
X_VAL_NPY: str = "X_val.npy"
X_TEST_NPY: str = "X_test.npy"
Y_TRAIN_NPY: str = "y_train.npy"
Y_VAL_NPY: str = "y_val.npy"
Y_TEST_NPY: str = "y_test.npy"

# Report figures
FIG_CLASS_DISTRIBUTION: str = "dataset_class_distribution.png"
FIG_CORRELATION_HEATMAP: str = "feature_correlation_heatmap.png"
FIG_PREPROCESSING_PIPELINE: str = "preprocessing_pipeline.png"
FIG_LSTM_ARCHITECTURE: str = "lstm_architecture.png"
FIG_TRAINING_ACCURACY: str = "training_accuracy_curve.png"
FIG_TRAINING_LOSS: str = "training_loss_curve.png"
FIG_CONFUSION_MATRIX: str = "confusion_matrix.png"
FIG_ROC_CURVE: str = "roc_curve.png"
FIG_MODEL_COMPARISON: str = "model_comparison_chart.png"
FIG_PRECISION_RECALL: str = "precision_recall_curve.png"
FIG_FEATURE_IMPORTANCE: str = "feature_importance.png"


def fig_name(dataset: str, fig_type: str) -> str:
    """Return a dataset-prefixed figure filename.

    Example::

        fig_name("nsl_kdd", "confusion_matrix")
        # → "confusion_matrix_nsl_kdd.png"
    """
    return f"{fig_type}_{dataset}.png"

# Report tables
TABLE_DATASET_SUMMARY: str = "dataset_summary.csv"
TABLE_HYPERPARAMETERS: str = "hyperparameters.csv"
TABLE_BASELINE_METRICS: str = "baseline_metrics.csv"
TABLE_FINAL_METRICS: str = "final_metrics.csv"

# Report metrics
METRICS_CLASSIFICATION_REPORT: str = "classification_report.txt"
METRICS_EVALUATION_RESULTS: str = "evaluation_results.json"
METRICS_ROC_AUC_SCORES: str = "roc_auc_scores.json"

# Logs
LOG_TRAINING: str = "training.log"
LOG_TRAINING_HISTORY: str = "training_history.csv"
LOG_PIPELINE: str = "pipeline.log"

# Outputs
OUT_TEST_PREDICTIONS: str = "test_predictions.csv"
OUT_NEW_PREDICTIONS: str = "new_data_predictions.csv"

# Visualization Constants

FIGURE_DPI: int = 300          # Publication-quality DPI for Chapter 4
FIGURE_SIZE: Tuple[int, int] = (12, 8)
FIGURE_SIZE_SQUARE: Tuple[int, int] = (10, 10)
FIGURE_SIZE_WIDE: Tuple[int, int] = (16, 6)
PLOT_STYLE: str = "seaborn-v0_8-whitegrid"
COLOR_PALETTE: str = "husl"
FONT_SIZE: int = 12
TITLE_FONT_SIZE: int = 14
LABEL_FONT_SIZE: int = 11

# Color scheme for 5 NSL-KDD classes
CLASS_COLORS: List[str] = [
    "#2ecc71",   # Normal  — green
    "#e74c3c",   # DoS     — red
    "#3498db",   # Probe   — blue
    "#f39c12",   # R2L     — orange
    "#9b59b6",   # U2R     — purple
]

# Supported Dataset Identifiers

SUPPORTED_DATASETS: List[str] = ["nsl_kdd", "cicids2017", "unsw_nb15"]

# Mapping from dataset identifier to display name
DATASET_DISPLAY_NAMES: Dict[str, str] = {
    "nsl_kdd": "NSL-KDD",
    "cicids2017": "CICIDS2017",
    "unsw_nb15": "UNSW-NB15",
}

# Supported Model Identifiers

SUPPORTED_MODELS: List[str] = [
    "lstm",
    "rnn",
    "random_forest",
    "svm",
    "logistic_regression",
]

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "lstm": "LSTM",
    "rnn": "Standard RNN",
    "random_forest": "Random Forest",
    "svm": "Support Vector Machine",
    "logistic_regression": "Logistic Regression",
}

# Logging Constants

LOG_FORMAT: str = "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL: str = "INFO"