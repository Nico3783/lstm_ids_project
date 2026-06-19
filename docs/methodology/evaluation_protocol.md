# Evaluation Protocol

**Project:** Deep Learning IDS Using LSTM  
**Chapter Reference:** Chapter 3, Section 3.5.5 — Model Evaluation  
**Module:** `src/evaluation/`

---

## Overview

The evaluation protocol defines precisely how model performance is measured,
reported, and compared. All evaluation decisions are made on the **held-out
test set** (15% of the full dataset) that was never seen during training or
hyperparameter tuning. This strict separation is fundamental to producing honest,
publication-quality results.

> *"The held-out test set was used for all final performance assessments."*
> — Chapter 3, Section 3.5.5

---

## 1. Evaluation Principles

### 1.1 Test Set Isolation

The test set is used **once and only once** — for final evaluation after all
training and tuning decisions have been made. Evaluating on the validation set
and reporting those results as test performance would constitute data leakage.

### 1.2 Stratified Evaluation

Because NSL-KDD has severe class imbalance (U2R and R2L are minority classes),
evaluation uses **both macro-averaged and weighted-averaged** variants of all
class-level metrics. The macro average gives equal weight to each class
regardless of frequency — a fairer measure when minority class performance
matters most.

### 1.3 Comparison Fairness

All models — LSTM, Random Forest, SVM, Logistic Regression — are evaluated
on exactly the same test set with the same class label encoding. Baselines
receive the same scaled features as the LSTM (last-timestep slice of the 3-D
input, which corresponds to the feature vector whose class label is assigned
to the sequence).

---

## 2. Metrics Computed

### 2.1 Classification Metrics

All metrics computed using `sklearn.metrics` on integer-encoded labels.

| Metric | Symbol | Formula | Averaging |
|--------|--------|---------|-----------|
| Accuracy | Acc | (TP+TN) / N | — |
| Precision | P | TP / (TP+FP) | Macro, Weighted |
| Recall | R | TP / (TP+FN) | Macro, Weighted |
| F1-Score | F1 | 2·P·R / (P+R) | Macro, Weighted |

**Macro averaging** (primary metric for Chapter 4 comparison):
```
F1_macro = (1/K) × Σ F1_k    for k in 1..K classes
```

**Weighted averaging:**
```
F1_weighted = Σ (n_k / N) × F1_k
```

### 2.2 ROC-AUC

Computed using the One-vs-Rest (OvR) strategy for multi-class classification.
For each class *c*, a binary classifier is evaluated (class *c* vs all others)
and the area under its ROC curve is computed. The macro average across all
classes is reported.

```python
roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
```

**Requires:** Probability scores `y_prob` (not just hard predictions) — the LSTM
and all baseline models are configured to output probabilities.

### 2.3 Confusion Matrix

A K×K matrix where entry `C[i,j]` is the number of records of true class *i*
predicted as class *j*. The normalised version (row-normalised) shows the
**recall** per class and reveals which classes are most confused with each other.

Particularly important for NSL-KDD: U2R and R2L are rarely-seen classes and
the confusion matrix reveals whether the model correctly identifies them rather
than always predicting Normal or DoS.

### 2.4 Per-Class Metrics

Each class receives individual precision, recall, F1, and support values in
the full classification report (`reports/metrics/classification_report.txt`).

---

## 3. Evaluation Workflow

### 3.1 LSTM Evaluation

```python
# 1. Load saved model
model = load_keras_model('models/final/lstm_ids_model.keras')

# 2. Generate predictions on test set (3-D input)
y_prob = model.predict(X_test, batch_size=256, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

# 3. Compute all metrics
metrics = compute_metrics(y_test, y_pred, y_prob,
                          class_names=class_names,
                          model_name='LSTM')
```

### 3.2 Baseline Evaluation

```python
# Baseline models receive 2-D input (last timestep of each sequence)
X_test_2d = X_test[:, -1, :]   # (n_test, n_features)

y_pred_rf, y_prob_rf = model.predict(X_test_2d), model.predict_proba(X_test_2d)
metrics_rf = compute_metrics(y_test, y_pred_rf, y_prob_rf, model_name='RF')
```

### 3.3 Outputs Generated

After evaluation, the following files are automatically written:

| Output | Location | Format |
|--------|----------|--------|
| Full classification report | `reports/metrics/classification_report.txt` | Plain text |
| All model metrics | `reports/metrics/evaluation_results.json` | JSON |
| ROC-AUC scores per class | `reports/metrics/roc_auc_scores.json` | JSON |
| Confusion matrix (LSTM) | `reports/figures/confusion_matrix.png` | PNG 300 DPI |
| ROC curves | `reports/figures/roc_curve.png` | PNG 300 DPI |
| Model comparison chart | `reports/figures/model_comparison_chart.png` | PNG 300 DPI |
| Precision-Recall curves | `reports/figures/precision_recall_curve_nsl_kdd.png` | PNG 300 DPI |
| Final metrics table | `reports/tables/final_metrics.csv` | CSV |
| Baseline comparison table | `reports/tables/baseline_metrics.csv` | CSV |
| Test set predictions | `outputs/predictions/test_predictions.csv` | CSV |

---

## 4. Feature Importance Analysis

**Method:** Permutation Importance  
**Chapter Reference:** Chapter 3, Section 3.7 — Data Analysis Techniques

Permutation importance is computed post-training by repeatedly shuffling each
feature column in the test set and measuring the resulting accuracy drop.
Features that cause large accuracy drops when shuffled are considered more
important to the model's predictions.

```python
# For each feature i, repeat n_repeats times:
X_permuted = X_test.copy()
X_permuted[:, i] = X_permuted[np.random.permutation(n_test), i]
accuracy_drop = baseline_accuracy - model.score(X_permuted, y_test)
importance[i] = mean(accuracy_drop over n_repeats)
```

**Advantages over weight-based importance:**
- Model-agnostic — works with both LSTM and baseline models
- Reflects actual impact on predictions, not just parameter magnitudes
- Accounts for feature interactions

**Output:** `reports/figures/feature_importance.png`

---

## 5. Statistical Interpretation Guidelines

### 5.1 Macro F1-Score (Primary Metric)

The macro F1-score is the primary metric for Chapter 4 comparison because:
1. It treats all classes equally regardless of frequency
2. The NSL-KDD dataset has severe class imbalance
3. Detecting rare U2R and R2L attacks is as important as detecting common DoS
4. Weighted metrics can be inflated by high performance on the dominant Normal class

### 5.2 Understanding NSL-KDD Class Imbalance

| Class | Approx. Training % | Imbalance Risk |
|-------|-------------------|----------------|
| Normal | ~53% | Majority class — model biased toward this |
| DoS | ~39% | Second majority — well represented |
| Probe | ~5% | Moderate minority |
| R2L | ~1% | Severe minority |
| U2R | <0.1% | Extreme minority |

Class weighting during training mitigates this, but U2R detection is inherently
challenging. Low U2R recall in the confusion matrix does not necessarily indicate
a model failure if the per-class metrics show appropriate weight was given.

### 5.3 Interpreting ROC-AUC

- AUC = 1.0: Perfect discrimination
- AUC = 0.9–1.0: Excellent
- AUC = 0.8–0.9: Good
- AUC = 0.7–0.8: Fair
- AUC < 0.7: Poor

Expected per-class AUC for a well-trained LSTM on NSL-KDD:
- Normal: ≥ 0.99
- DoS: ≥ 0.99
- Probe: ≥ 0.97
- R2L: ≥ 0.90
- U2R: ≥ 0.80 (lower due to extreme scarcity)

---

## 6. Comparison Table Template

The following table structure is used for Chapter 4:

| Model | Accuracy | Precision (M) | Recall (M) | F1 (Macro) | F1 (Wtd) | ROC-AUC |
|-------|----------|--------------|-----------|-----------|---------|---------|
| **LSTM** | — | — | — | — | — | — |
| Random Forest | — | — | — | — | — | — |
| SVM | — | — | — | — | — | — |
| Logistic Regression | — | — | — | — | — | — |

Bold = proposed model. All values rounded to 4 decimal places.

---

## 7. Running Evaluation

### Full evaluation from saved model

```bash
python evaluate.py --dataset nsl_kdd
```

### Comparison of all models

```bash
python compare_models.py --dataset nsl_kdd
```

### Via notebook (interactive)

Open `notebooks/06_model_evaluation.ipynb` or `notebooks/07_results_analysis.ipynb`
and run all cells. All figures are saved to `reports/figures/` automatically.

### Via pipeline (end-to-end)

```bash
python run_pipeline.py --dataset nsl_kdd
```

The pipeline runs evaluation automatically after training in Stage 9.
