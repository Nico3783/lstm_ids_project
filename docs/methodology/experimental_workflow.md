# Experimental Workflow

**Project:** Design and Implementation of a Deep Learning Intrusion Detection System Using Long Short-Term Memory (LSTM)  
**Institution:** Federal University of Technology, Akure (FUTA), Nigeria  
**Chapter Reference:** Chapter 3 — Materials and Methods

---

## Overview

The experimental workflow is structured into five sequential phases as described in Chapter 3, Section 3.2 (Research Design). Each phase is reproducible, logged, and produces artefacts that feed directly into the next stage.

```
Phase 1: Data Acquisition & EDA
        ↓
Phase 2: Preprocessing & Feature Engineering
        ↓
Phase 3: Model Design & Implementation
        ↓
Phase 4: Training & Optimisation
        ↓
Phase 5: Evaluation & Reporting
```

---

## Phase 1 — Data Acquisition and Exploratory Analysis

**Script:** `python -m src.data.download --dataset nsl_kdd`  
**Notebook:** `notebooks/01_data_exploration.ipynb`  
**Chapter Reference:** Section 3.5.1

### Datasets Acquired

| Dataset | Source | Records | Features | Classes |
|---------|--------|---------|----------|---------|
| NSL-KDD | University of New Brunswick | 125,973 (train) + 22,544 (test) | 41 | 5 |
| CICIDS2017 | Canadian Institute for Cybersecurity | ~2.8M | 80 | 15 |
| UNSW-NB15 | Australian Centre for Cyber Security | ~2.54M | 49 | 10 |

### EDA Steps

1. **Dataset shape inspection** — rows, columns, dtypes, memory usage
2. **Class distribution analysis** — count and percentage per attack category
3. **Missing value audit** — per-column missing counts and ratios
4. **Infinite value detection** — flow rate columns in CICIDS2017
5. **Duplicate row identification** — count and percentage
6. **Categorical feature cardinalities** — unique values per categorical column
7. **Feature correlation heatmap** — Pearson correlation, top-30 features
8. **Feature distribution histograms** — top-16 features by variance
9. **Validation report generation** — 10 quality checks per split

### EDA Outputs

All figures saved to `reports/figures/` at 300 DPI:
- `dataset_class_distribution.png`
- `feature_correlation_heatmap.png`
- `feature_distributions_nsl_kdd.png`
- `missing_values_nsl_kdd.png`

---

## Phase 2 — Preprocessing and Feature Engineering

**Script:** `python run_pipeline.py` (stages 3–6)  
**Notebook:** `notebooks/02_data_preprocessing.ipynb`, `notebooks/03_sequence_generation.ipynb`  
**Chapter Reference:** Section 3.5.2

### Preprocessing Pipeline (8 Steps)

| Step | Operation | Module |
|------|-----------|--------|
| 1 | Drop irrelevant columns (`difficulty`, `_split`) | `preprocessing.drop_irrelevant_columns` |
| 2 | Replace infinite values with NaN | `preprocessing.handle_missing_and_infinite` |
| 3 | Mean imputation (continuous features) | `preprocessing.handle_missing_and_infinite` |
| 4 | Mode imputation (categorical features) | `preprocessing.handle_missing_and_infinite` |
| 5 | Remove duplicate rows | `preprocessing.remove_duplicates` |
| 6 | Map raw attack types → 5-class integer labels | `preprocessing.map_nsl_kdd_labels` |
| 7 | One-hot encode categorical features | `preprocessing.encode_categorical_features` |
| 8 | Min-Max scale to [0, 1] (train-fitted only) | `preprocessing.fit_scaler` / `apply_scaler` |

### Sequence Construction

- **Method:** Sliding window over ordered traffic records
- **Window size:** 10 (selected from hyperparameter search)
- **Step size:** 1 (overlapping windows, maximum temporal coverage)
- **Label assignment:** Class of the final timestep in each window
- **Output shape:** `(N−9, 10, n_features)` for `N` input records

### Data Splitting

- **Ratios:** 70% train / 15% validation / 15% test
- **Method:** Stratified sampling (preserves class proportions)
- **Validation role:** Hyperparameter tuning and early stopping only
- **Test role:** Held out until final evaluation — never seen during training or tuning

### Artifacts Saved

| File | Location | Purpose |
|------|----------|---------|
| `X_train.npy`, `X_val.npy`, `X_test.npy` | `data/processed/` | Feature sequences per split |
| `y_train.npy`, `y_val.npy`, `y_test.npy` | `data/processed/` | Labels per split |
| `scaler.pkl` | `data/processed/`, `models/final/` | Fitted MinMaxScaler |
| `label_encoder.pkl` | `data/processed/`, `models/final/` | Class name mapping |
| `feature_names.pkl` | `data/processed/`, `models/final/` | Ordered feature list |
| `metadata.json` | `data/processed/`, `models/final/` | Dataset statistics snapshot |

---

## Phase 3 — Model Design and Implementation

**Module:** `src/models/lstm_model.py`  
**Chapter Reference:** Section 3.5.3

### LSTM Architecture

```
Input (batch, 10, n_features)
    │
    ▼
LSTM(128, return_sequences=True, tanh/sigmoid)
    │
    ▼
Dropout(0.2)
    │
    ▼
LSTM(64, return_sequences=False, tanh/sigmoid)
    │
    ▼
Dropout(0.2)
    │
    ▼
Dense(32, ReLU, L2 λ=0.001)
    │
    ▼
BatchNormalization
    │
    ▼
Dense(n_classes, Softmax)
```

### Compilation

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | Adam | Adaptive LR converges faster than SGD |
| Learning rate | 0.001 | Standard initial LR for Adam |
| Loss | Categorical Cross-Entropy | Multi-class classification |
| Metrics | Accuracy, Precision, Recall | Standard IDS evaluation metrics |

### Baseline Models (for Chapter 4 comparison)

| Model | Key Parameters | Class Weighting |
|-------|---------------|-----------------|
| Random Forest | 100 trees, unlimited depth | Balanced (inverse frequency) |
| Support Vector Machine | RBF kernel, C=1.0, γ=scale | Balanced |
| Logistic Regression | lbfgs solver, multinomial | Balanced |
| Standard RNN | 64 units, SimpleRNN | N/A |

---

## Phase 4 — Training and Optimisation

**Script:** `python train.py --dataset nsl_kdd`  
**Notebook:** `notebooks/05_lstm_training.ipynb`  
**Chapter Reference:** Section 3.5.4

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Maximum epochs | 100 |
| Batch size | 64 |
| Early stopping monitor | `val_loss` |
| Early stopping patience | 10 epochs |
| Best weights restoration | Yes (`restore_best_weights=True`) |
| Learning rate reduction | Factor=0.5, patience=5, min_lr=1e-6 |

### Class Imbalance Strategy

Inverse-frequency class weighting applied during training:

```
weight_c = n_samples / (n_classes × count_c)
```

This encourages the model to attend to minority classes (R2L, U2R) rather than optimising only for the dominant Normal and DoS classes.

### Hyperparameter Search Grid (Chapter 3, Section 3.5.4)

| Hyperparameter | Search Values |
|----------------|---------------|
| LSTM layers | 1, 2, 3 |
| Units per layer | 32, 64, 128, 256 |
| Dropout rate | 0.1, 0.2, 0.3, 0.5 |
| Learning rate | 0.01, 0.001, 0.0001 |
| Batch size | 32, 64, 128 |

**Objective metric:** Validation accuracy (maximise)

### Callbacks Active During Training

1. `EarlyStopping` — halts training when `val_loss` fails to improve for 10 epochs
2. `ModelCheckpoint` — saves best model to `models/checkpoints/best_model.keras`
3. `ReduceLROnPlateau` — halves learning rate after 5 epochs of no improvement
4. `CSVLogger` — writes epoch metrics to `reports/logs/training_history.csv`
5. `TensorBoard` — event files written to `reports/logs/tensorboard/`

---

## Phase 5 — Evaluation and Reporting

**Script:** `python evaluate.py` and `python compare_models.py`  
**Notebook:** `notebooks/06_model_evaluation.ipynb`, `notebooks/07_results_analysis.ipynb`  
**Chapter Reference:** Section 3.5.5

### Evaluation Protocol

All final performance measurements are made on the **held-out test set only**. The validation set was used exclusively during training for early stopping and hyperparameter tuning decisions.

### Metrics Computed

| Metric | Averaging | Purpose |
|--------|-----------|---------|
| Accuracy | — | Overall classification rate |
| Precision | Macro + Weighted | Attack detection precision |
| Recall | Macro + Weighted | Attack detection recall |
| F1-Score | Macro + Weighted | Harmonic mean (primary Chapter 4 metric) |
| ROC-AUC | Macro OvR | Discriminative ability across classes |
| Confusion Matrix | Per-class | Misclassification pattern analysis |

### Chapter 4 Outputs

All the following are auto-generated by `run_pipeline.py` or `python compare_models.py`:

**Figures** (saved to `reports/figures/`, 300 DPI):
- Training accuracy and loss curves
- Confusion matrix (normalised)
- ROC curves with per-class AUC
- Precision-Recall curves
- Model comparison bar chart
- Feature importance bar chart
- LSTM architecture diagram
- Preprocessing pipeline diagram
- System architecture diagram

**Tables** (saved to `reports/tables/`):
- `dataset_summary.csv` — dataset statistics
- `hyperparameters.csv` — best configuration
- `baseline_metrics.csv` — all model comparison
- `final_metrics.csv` — per-class classification report

**JSON Metrics** (saved to `reports/metrics/`):
- `evaluation_results.json` — all scalar metrics per model
- `roc_auc_scores.json` — per-class AUC values
- `classification_report.txt` — full sklearn report

---

## Reproducibility Checklist

To exactly reproduce all results:

- [ ] Python 3.9 with dependencies from `requirements.txt`
- [ ] Random seed = 42 (set via `set_global_seed(42)` in `run_pipeline.py`)
- [ ] NSL-KDD dataset from official UNB source (same version)
- [ ] `config.yaml` unmodified from repository
- [ ] Run: `python run_pipeline.py --dataset nsl_kdd`

Preprocessing artifacts (`scaler.pkl`, `label_encoder.pkl`, `feature_names.pkl`,
`metadata.json`) are saved alongside the model so that inference always uses
the exact same transformations as training.

---

## Execution Order

```bash
# 1. Setup environment
chmod +x setup.sh && ./setup.sh
source venv/bin/activate

# 2. Download NSL-KDD
python -m src.data.download --dataset nsl_kdd

# 3. Run full pipeline
python run_pipeline.py --dataset nsl_kdd

# 4. (Optional) Standalone training with tuning
python train.py --dataset nsl_kdd --tune

# 5. Evaluate saved model
python evaluate.py --dataset nsl_kdd

# 6. Compare all models
python compare_models.py --dataset nsl_kdd

# 7. Predict on new data
python predict.py --input data/sample/sample_input.csv
```
