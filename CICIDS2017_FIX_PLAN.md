# CICIDS2017 LSTM Failure — Root Cause Analysis & Fix Plan

## Problem Statement

LSTM model achieves ~1% accuracy on CICIDS2017 (15 classes), worse than random (6.67%). Early stops at epoch 12. Same architecture works on NSL-KDD (5 classes). The model is fundamentally failing to learn.

## Root Causes (ranked by severity)

### 1. CRITICAL: Class Weight Cap Too High
- **File:** `src/training/class_weights.py:71`
- **Current:** `max_weight = float(np.sqrt(n_samples))` → ~1,327 for CICIDS2017
- **Impact:** A sample from class 8 (11 samples) has loss multiplied by 1,327. This overwhelms the gradient signal from the majority class (BENIGN, 83%). The loss landscape is so distorted that Adam cannot navigate it.
- **Literature:** Class weight caps of 10-20 are standard. Max 50 for extreme cases.
- **Fix:** Change to `max_weight = float(min(20.0, np.sqrt(n_samples)))`. Make it configurable in `config.yaml`.

### 2. CRITICAL: Imputation Bug (NaN Not Filled)
- **File:** `src/data/preprocessing.py:163`
- **Current:** `df[col].fillna(fill_val, inplace=True)` fails silently with pandas Copy-on-Write
- **Evidence:** Pipeline log shows "Missing values: 5734 → 5734" — no change
- **Impact:** 5,734 NaN values propagate through MinMaxScaler into training data. NaN sequences corrupt LSTM gradient updates.
- **Fix:** Replace with `df[col] = df[col].fillna(fill_val)` (assignment, not inplace).

### 3. HIGH: Model Architecture Too Small
- **File:** `src/models/lstm_model.py:46-205`
- **Current:** LSTM(128, 64) + Dense(32) = ~158K params for 15 classes × 78 features
- **Impact:** The Dense(32) layer compresses 64 LSTM outputs into 32 dimensions before 15-class output — severe information bottleneck.
- **Fix:** For CICIDS2017: LSTM(256, 128) + Dense(64) = ~500K params. Add dataset-specific model config.

### 4. HIGH: No Gradient Clipping
- **File:** `src/models/lstm_model.py:191-194`
- **Current:** Adam optimizer without `clipnorm`
- **Impact:** Class-weight-induced gradient spikes destabilize Adam's internal state (momentum estimates).
- **Fix:** Add `clipnorm=1.0` to Adam optimizer.

### 5. HIGH: Early Stopping Too Aggressive
- **File:** `config.yaml:186-190`
- **Current:** patience=10, min_delta=0.0001, ReduceLR factor=0.5 patience=5
- **Impact:** ReduceLR halves LR at epoch 7, again at epoch 12. Early stopping kills training at epoch 12. Model barely gets to train.
- **Fix:** patience=20, min_delta=0.00001, ReduceLR patience=8.

### 6. MEDIUM: Scaler Data Leakage
- **File:** `src/data/preprocessing.py:758-759`
- **Current:** MinMaxScaler fitted on ENTIRE dataset before splitting
- **Impact:** Scaling ranges include test data information. Methodology violation for thesis.
- **Fix:** Fit scaler on training data only (after Stage 3 split), transform val/test with training scaler.

### 7. MEDIUM: Flow-Rate Features Corrupted
- **File:** `src/data/preprocessing.py:99-180`
- **Current:** Mean imputation of inf values in Flow Bytes/s, Flow Packets/s
- **Impact:** These features span 0 to millions. Mean imputation + MinMax scaling compresses all values near 0.
- **Fix:** Apply `np.log1p()` to flow-rate features before scaling.

### 8. MEDIUM: Rare Classes Unlearnable
- **Current:** Classes 8 (11 samples), 9 (36), 13 (21) out of 2.5M records
- **Impact:** Even with capped weights, 11-36 samples cannot train an LSTM to generalize.
- **Options:**
  a) Merge ultra-rare classes into broader categories (e.g., group by attack family)
  b) Exclude classes with <100 samples (reduce to 12 classes)
  c) Use SMOTE for rare classes (but SMOTE on sequences is complex)
  d) Accept that these classes will have poor recall and document it

## Fix Implementation Order

### Phase 1: Data Pipeline Fixes (run_pipeline.py + preprocessing.py)
1. Fix imputation bug (fillna inplace → assignment)
2. Drop all-NaN rows before imputation
3. Add log-scaling for flow-rate features
4. Fix scaler leakage (fit on train only)
5. Restore human-readable class names in metadata

### Phase 2: Model & Training Fixes (config.yaml + lstm_model.py + class_weights.py)
6. Cap class weights at 20 (configurable)
7. Add gradient clipping (clipnorm=1.0)
8. Increase model capacity for CICIDS2017 (256/128 LSTM, 64 Dense)
9. Tune early stopping (patience=20, min_delta=0.00001)

### Phase 3: Re-run on Colab
10. Re-run stages 1-9 on Colab with all fixes
11. Verify training reaches 20+ epochs
12. Verify accuracy improves beyond random baseline

## Files to Modify

| File | Changes |
|------|---------|
| `src/data/preprocessing.py` | Fix imputation bug, add log-scaling, drop all-NaN rows, fix scaler leakage |
| `src/training/class_weights.py` | Cap max_weight at 20 (configurable) |
| `src/models/lstm_model.py` | Add clipnorm=1.0 to Adam optimizer |
| `config.yaml` | Add CICIDS2017-specific model config, tune early stopping |
| `run_pipeline.py` | Fix scaler fitting order (fit after split) |

## Expected Outcomes

- **Training:** Model should reach 20-50 epochs before early stopping
- **Accuracy:** Should exceed random baseline (6.67%) significantly
- **Macro F1:** Should improve from ~0.07 to 0.3-0.5 range
- **Validation:** Loss should decrease consistently, not plateau at epoch 2
