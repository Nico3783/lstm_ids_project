# Chapter 4 — Results and Evaluation (Analysis Framework)

**Project:** Deep Learning IDS Using LSTM
**Status:** Framework ready — requires NSL-KDD and CICIDS2017 Colab results to populate

---

## 4.1 Training Convergence Analysis (UNSW-NB15 — Available)

The LSTM model was trained for 100 epochs with early stopping patience=10 and
ReduceLROnPlateau patience=5.

### Training Trajectory (from `reports/logs/training_history.csv`)

| Epoch | Train Acc | Val Acc | Train Loss | Val Loss | LR |
|:-----:|:---------:|:-------:|:----------:|:--------:|:---:|
| 0 | 0.556 | 0.868 | 1.173 | 0.532 | 1e-3 |
| 10 | 0.898 | 0.925 | 0.320 | 0.265 | 1e-3 |
| 20 | 0.920 | 0.922 | 0.230 | 0.261 | 1e-3 |
| 30 | 0.927 | 0.919 | 0.196 | 0.272 | 5e-4 |
| 50 | 0.938 | 0.928 | 0.158 | 0.264 | 5e-4 |
| 70 | 0.946 | 0.966 | 0.124 | 0.101 | 2.5e-4 |
| 90 | 0.949 | 0.968 | 0.100 | 0.096 | 1.25e-4 |
| **98** | **0.952** | **0.970** | **0.100** | **0.097** | **1.25e-4** |
| 99 | 0.951 | 0.968 | 0.097 | 0.096 | 1.25e-4 |

**Best epoch:** 98 (val_accuracy = 0.9695, val_loss = 0.0970)

### Key Observations

1. **Rapid initial convergence:** Accuracy jumped from 55.6% to 86.8% in the first
   epoch, indicating the model quickly learned dominant class patterns.
2. **Plateau at epochs 15–40:** Validation accuracy stabilised around 92%, suggesting
   the model reached linear-separable feature capacity before deep pattern extraction.
3. **LR reduction effect:** Learning rate halved from 1e-3 → 5e-4 around epoch 35,
   then to 2.5e-4 around epoch 65, then to 1.25e-4 around epoch 85. Each reduction
   correlated with a small but measurable improvement in validation accuracy.
4. **No overfitting observed:** Training and validation curves remain closely aligned
   throughout, with validation accuracy occasionally exceeding training accuracy
   (attributed to dropout being active only during training).
5. **Generalisation gap:** Final train-val accuracy gap = 0.952 - 0.970 = -0.018
   (negative = validation better), confirming strong generalisation.

### Figure to Generate

Plot training curves (accuracy and loss vs epoch) from `training_history.csv`.
This figure already exists as `reports/figures/training_accuracy.png` and
`reports/figures/training_loss.png`.

---

## 4.2 Multi-Dataset Comparison (Pending Colab Re-runs)

### Planned Structure

| Dataset | Samples | Features | Classes | Accuracy | F1-Macro | ROC-AUC | Train Time |
|:--------|--------:|:--------:|:-------:|:--------:|:--------:|:-------:|:----------:|
| NSL-KDD | ~148k | 122 | 5 | — | — | — | — |
| UNSW-NB15 | 22,185 | 122 | 5 | 0.9667 | 0.7987 | 0.9589 | ~8 min |
| CICIDS2017 | ~283k | 80 | 15 | — | — | — | — |

**Status:** Only UNSW-NB15 results are currently available. NSL-KDD and CICIDS2017
require pipeline re-runs on Colab (T4 GPU).

### Analysis Points (to write once data is available)

1. **Accuracy variance across datasets:** Which dataset yields highest/lowest accuracy?
   Relate to dataset complexity and class overlap.
2. **F1-macro vs accuracy divergence:** Large gaps indicate severe class imbalance
   effects (expected on CICIDS2017 with 15 classes).
3. **ROC-AUC comparison:** Model's probability calibration quality across datasets.
4. **Training time scaling:** Linear scaling with dataset size, or super-linear due to
   LSTM sequential processing?

---

## 4.3 Baseline Comparison (UNSW-NB15 — Available)

### Results Table

| Model | Accuracy | Precision | Recall | F1-Macro | F1-Weighted | ROC-AUC |
|:------|:--------:|:---------:|:------:|:--------:|:-----------:|:-------:|
| **LSTM** | 0.9667 | 0.7698 | 0.8410 | **0.7987** | 0.9682 | 0.9589 |
| Random Forest | **0.9942** | **0.9345** | 0.8826 | **0.9036** | **0.9942** | **0.9998** |
| SVM | 0.4866 | 0.4928 | 0.7620 | 0.3839 | 0.3480 | 0.7726 |
| Logistic Regression | 0.9333 | 0.6834 | **0.9356** | 0.7199 | 0.9431 | 0.9941 |

### Critical Discussion Points

1. **RF outperforms LSTM on UNSW-NB15:** This is a well-documented phenomenon —
   tree-based ensembles often outperform deep learning on tabular/categorical data
   with limited sample diversity. The discussion should acknowledge this honestly
   and argue for LSTM's value proposition:
   - **Temporal dependency modelling:** LSTM captures sequential attack patterns that
     RF cannot (e.g., multi-stage attacks).
   - **Online learning adaptability:** LSTM weights can be fine-tuned incrementally;
     RF requires full retraining.
   - **Generalisation potential:** LSTM may generalise better to unseen attack types
     (zero-shot transfer via learned representations).

2. **SVM poor performance (48.66%):** The RBF kernel SVM likely suffered from:
   - High dimensionality after one-hot encoding (122 features)
   - Feature scaling mismatch (SVMs are sensitive to unscaled features)
   - Multi-class strategy (one-vs-one or one-vs-rest) degrading with 5 classes
   - Should be discussed as a methodological lesson, not a fair comparison.

3. **LR competitive accuracy but poor F1-macro:** Logistic Regression achieves 93.33%
   accuracy but only 0.7199 F1-macro, indicating it over-predicts majority classes
   and fails on minority classes (R2L, U2R). This validates the use of F1-macro as
   the primary metric.

---

## 4.4 Per-Class Error Analysis (UNSW-NB15 — Available)

### Per-Class Performance

| Class | Precision | Recall | F1 | Support | Analysis |
|:------|:---------:|:------:|:--:|:-------:|:---------|
| Normal | 0.9846 | 0.9548 | 0.9695 | 11,545 | Strong performance on majority class |
| DoS | 0.9916 | 0.9811 | 0.9863 | 7,947 | Near-perfect; DoS patterns are distinctive |
| Probe | 0.9270 | 0.9890 | 0.9570 | 2,093 | High recall, slightly lower precision (some Normal misclassified as Probe) |
| R2L | 0.6297 | 0.9467 | 0.7563 | 582 | Low precision, high recall — many false positives from other classes |
| **U2R** | **0.3158** | **0.3333** | **0.3243** | **18** | **Critical failure** — only 18 samples, insufficient for learning |

### Discussion Points

1. **U2R as the critical bottleneck:** With only 18 test samples (0.08% of data), the
   model cannot learn U2R attack signatures. This is a fundamental data limitation,
   not a model architecture flaw.
   - **Mitigation strategies:** SMOTE oversampling, few-shot learning, or transfer
     learning from related datasets.
   - **Practical impact:** U2R attacks (privilege escalation) are rare but high-impact;
     the model should be deployed as a first-stage filter with human review for
     low-confidence U2R predictions.

2. **R2L false positives:** The model classifies many non-R2L samples as R2L (precision
   0.63), possibly because R2L features overlap with Normal traffic patterns
   (e.g., login attempts). This suggests feature engineering improvements could help.

3. **DoS detection reliability:** Near-perfect DoS detection (F1=0.986) validates the
   LSTM's ability to learn volumetric attack patterns, which is the primary use case
   for most IDS deployments.

---

## 4.5 Computational Cost Analysis (Pending)

### Planned Metrics

| Metric | Value (UNSW-NB15) | Notes |
|:-------|:-----------------:|:------|
| Training time (100 epochs) | ~8 min (T4 GPU) | Needs Colab timing |
| Inference time (per sample) | ~0.3 ms | Needs measurement |
| Model size (parameters) | 180,293 | Fixed |
| Model file size | ~2.8 MB | Keras SavedModel format |
| Peak GPU memory | ~1.2 GB | Needs Colab measurement |

---

## 4.6 Figures Required (Chapter 4)

| Figure | Source | Status |
|:-------|:-------|:-------|
| Fig 4.1: Training accuracy curves | `reports/figures/training_accuracy.png` | ✅ Exists |
| Fig 4.2: Training loss curves | `reports/figures/training_loss.png` | ✅ Exists |
| Fig 4.3: Confusion matrix | `reports/figures/confusion_matrix.png` | ✅ Exists |
| Fig 4.4: ROC-AUC curves | `reports/figures/roc_curves.png` | ✅ Exists |
| Fig 4.5: Per-class F1 comparison | `reports/figures/class_distribution.png` | ✅ Exists |
| Fig 4.6: Feature importance | `reports/figures/feature_importance.png` | ✅ Exists |
| Fig 4.7: Multi-dataset comparison | **Needs generation** | ❌ Pending Colab |
| Fig 4.8: Model comparison bar chart | **Needs generation** | ❌ Pending |

---

## 4.7 Summary of What's Needed from Colab

| Task | Priority | Status |
|:-----|:--------:|:-------|
| Re-run NSL-KDD pipeline | High | ⏳ Pending |
| Re-run CICIDS2017 pipeline | Medium | ⏳ Pending |
| Time training on T4 GPU | Medium | ⏳ Pending |
| Measure inference latency | Low | ⏳ Pending |
| Generate multi-dataset comparison figure | Medium | ⏳ Pending |
| Generate model comparison bar chart | Low | ⏳ Pending |
