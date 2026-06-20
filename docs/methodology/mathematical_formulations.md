# Mathematical Formulations — Chapter 3

**Project:** Deep Learning IDS Using LSTM
**Chapter Reference:** Chapter 3, Sections 3.5.2–3.5.5
**Purpose:** Complete mathematical definitions for every formula cited in the thesis

---

## 1. Min-Max Feature Scaling (Section 3.5.2)

All continuous features are normalised to the [0, 1] range using Min-Max scaling:

$$
x'_i = \frac{x_i - x_{\min}}{x_{\max} - x_{\min}}
$$

where $x_i$ is the original value of feature $i$, $x_{\min}$ and $x_{\max}$ are the
minimum and maximum values computed **exclusively on the training set**, and $x'_i$ is
the scaled output. This ensures no information from the validation or test sets leaks
into the training process.

**Implementation:** `sklearn.preprocessing.MinMaxScaler(feature_range=(0.0, 1.0))`

---

## 2. Sliding Window Sequence Construction (Section 3.5.2)

Given a flat feature matrix $X \in \mathbb{R}^{N \times F}$ and label vector
$\mathbf{y} \in \mathbb{Z}^N$, a sliding window of width $W = 10$ and step size
$s = 1$ constructs sequences:

$$
\mathbf{X}^{(k)} = X[k : k+W, :] \in \mathbb{R}^{W \times F}
$$

$$
y^{(k)} = y_{k+W-1}
$$

where $k \in \{0, 1, \ldots, N - W\}$ and the label is assigned to the **last**
timestep in each window. The total number of sequences produced is $N - W + 1$.

**Parameter values:** $W = 10$, $s = 1$, $F$ varies by dataset
(NSL-KDD: $F \approx 122$ after one-hot encoding; CICIDS2017: $F \approx 80$;
UNSW-NB15: $F \approx 49$).

---

## 3. One-Hot Encoding (Section 3.5.2)

Categorical features with $C$ unique values are transformed into $C$ binary columns.
For a categorical feature $f$ with value $v \in \{v_1, v_2, \ldots, v_C\}$:

$$
\text{OHE}(f = v_j) = \mathbf{e}_j \in \{0, 1\}^C
$$

where $\mathbf{e}_j$ is the one-hot vector with a 1 in position $j$ and 0 elsewhere.
All $C$ columns are retained (no reference category is dropped) because the features
are nominal, not ordinal.

**NSL-KDD categorical features:** `protocol_type` ($C=3$), `service` ($C \approx 70$),
`flag` ($C \approx 11$).

---

## 4. Label Encoding (Section 3.5.2)

Multi-class labels are mapped from raw attack type strings to integer codes via a
two-step process:

1. **Raw attack type → category:** $l_{\text{raw}} \rightarrow l_{\text{cat}}$
   (e.g., `"neptune" → "DoS"`)
2. **Category → integer:** $l_{\text{cat}} \rightarrow l_{\text{int}}$

The mapping for NSL-KDD is:

| Integer $l$ | Category |
|:-----------:|:--------:|
| 0 | Normal |
| 1 | DoS |
| 2 | Probe |
| 3 | R2L |
| 4 | U2R |

---

## 5. LSTM Cell Equations (Section 3.5.3)

The Long Short-Term Memory (LSTM) unit at timestep $t$ computes the following
gating operations:

**Forget gate:**

$$
\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)
$$

**Input gate:**

$$
\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)
$$

**Candidate cell state:**

$$
\tilde{\mathbf{C}}_t = \tanh(\mathbf{W}_C [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_C)
$$

**Cell state update:**

$$
\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t
$$

**Output gate:**

$$
\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)
$$

**Hidden state:**

$$
\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t)
$$

where:
- $\sigma(\cdot)$ is the logistic sigmoid activation
- $\tanh(\cdot)$ is the hyperbolic tangent activation
- $\odot$ denotes element-wise multiplication
- $\mathbf{W}_f, \mathbf{W}_i, \mathbf{W}_C, \mathbf{W}_o$ are weight matrices
- $\mathbf{b}_f, \mathbf{b}_i, \mathbf{b}_C, \mathbf{b}_o$ are bias vectors
- $[\mathbf{h}_{t-1}, \mathbf{x}_t]$ denotes concatenation of the previous hidden
  state and current input

**Implementation note:** The codebase uses `activation="tanh"` and
`recurrent_activation="sigmoid"` in `tf.keras.layers.LSTM`, which maps directly to
the above equations.

---

## 6. Stacked LSTM Architecture (Section 3.5.3)

The proposed model uses a two-layer stacked LSTM:

**Layer 1:** LSTM with 128 units, `return_sequences=True`

$$
\mathbf{H}^{(1)} = \text{LSTM}_{128}(\mathbf{X}) \in \mathbb{R}^{W \times 128}
$$

**Dropout after Layer 1:** Rate $\rho = 0.2$

$$
\hat{\mathbf{H}}^{(1)} = \text{Dropout}(\mathbf{H}^{(1)}, \rho)
$$

**Layer 2:** LSTM with 64 units, `return_sequences=False`

$$
\mathbf{h}^{(2)} = \text{LSTM}_{64}(\hat{\mathbf{H}}^{(1)}) \in \mathbb{R}^{64}
$$

**Dropout after Layer 2:** Rate $\rho = 0.2$

$$
\hat{\mathbf{h}}^{(2)} = \text{Dropout}(\mathbf{h}^{(2)}, \rho)
$$

**Dense hidden layer:** 32 units, ReLU activation, L2 regularisation ($\lambda = 0.001$)

$$
\mathbf{z} = \text{ReLU}(\mathbf{W}_d \hat{\mathbf{h}}^{(2)} + \mathbf{b}_d)
$$

with the loss regulariser:

$$
\mathcal{L}_{\text{reg}} = \lambda \|\mathbf{W}_d\|_F^2
$$

**Batch Normalisation:**

$$
\hat{z}_j = \gamma_j \frac{z_j - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}} + \beta_j
$$

where $\mu_j$ and $\sigma_j^2$ are the mini-batch mean and variance, $\gamma_j$ and
$\beta_j$ are learnable scale and shift parameters, and $\epsilon = 10^{-3}$ is a
small constant for numerical stability.

**Output layer:** $K$ units ($K=5$ for NSL-KDD), softmax activation:

$$
\hat{y}_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}, \quad k = 1, \ldots, K
$$

---

## 7. Total Parameter Count (Section 3.5.3)

The model has 180,293 trainable parameters:

| Layer | Parameters |
|:------|----------:|
| LSTM-1 ($128$ units, input $F$) | $4 \times ((F + 128) \times 128) = 4 \times (F+128) \times 128$ |
| LSTM-2 ($64$ units, input $128$) | $4 \times ((128 + 64) \times 64) = 4 \times 192 \times 64 = 49{,}152$ |
| Dense hidden ($32$ units) | $64 \times 32 + 32 = 2{,}080$ |
| Batch Norm | $32 \times 4 = 128$ |
| Output ($K$ units) | $32 \times K + K$ |

*Exact count depends on the input feature dimension $F$ after one-hot encoding.*

---

## 8. Categorical Cross-Entropy Loss (Section 3.5.3)

For a single sample with true label $y$ (one-hot encoded) and predicted probabilities
$\hat{\mathbf{y}}$:

$$
\mathcal{L}_{\text{CE}} = -\sum_{k=1}^{K} y_k \log(\hat{y}_k)
$$

For a mini-batch of $B$ samples:

$$
\mathcal{L} = -\frac{1}{B} \sum_{b=1}^{B} \sum_{k=1}^{K} y_k^{(b)} \log(\hat{y}_k^{(b)})
$$

With L2 regularisation ($\lambda = 0.001$) on the Dense layer weights:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \|\mathbf{W}_d\|_F^2
$$

---

## 9. Class Weight Computation (Section 3.5.4)

To address class imbalance, per-class weights are computed inversely proportional to
class frequency:

$$
w_k = \frac{N}{K \cdot n_k}
$$

where $N$ is the total number of training samples, $K$ is the number of classes, and
$n_k$ is the number of samples in class $k$.

**Example (NSL-KDD approximate proportions):**

| Class | $n_k$ | $w_k$ |
|:------|-------:|-------:|
| Normal | 77,054 | 0.384 |
| DoS | 55,382 | 0.536 |
| Probe | 7,431 | 3.990 |
| R2L | 2,349 | 12.660 |
| U2R | 105 | 282.330 |

These weights are passed to the loss function during training, scaling the gradient
contribution of each sample proportionally.

---

## 10. Optimiser — Adam (Section 3.5.3)

The Adam optimiser maintains per-parameter adaptive learning rates:

$$
\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t
$$

$$
\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2
$$

$$
\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad
\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}
$$

$$
\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta \frac{\hat{\mathbf{m}}_t}
{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}
$$

with defaults: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-7}$,
$\eta = 0.001$ (initial learning rate).

---

## 11. Dropout Regularisation (Section 3.5.3)

During training, each unit in the dropout layer is zeroed with probability $\rho = 0.2$:

$$
\hat{h}_j = \frac{r_j \cdot h_j}{1 - \rho}
$$

where $r_j \sim \text{Bernoulli}(1 - \rho)$. At inference, dropout is disabled and
activations are used as-is.

---

## 12. Early Stopping (Section 3.5.4)

Training terminates when the validation loss has not improved for $P = 10$ consecutive
epochs:

$$
\text{stop at epoch } t \iff \forall\, t' \in [t-P, t): \quad
\mathcal{L}_{\text{val}}(t') \geq \mathcal{L}_{\text{val}}^{\text{best}}
$$

The best model weights (at epoch $t^* = \arg\min_{t'} \mathcal{L}_{\text{val}}(t')$)
are restored after stopping.

---

## 13. Learning Rate Reduction (Section 3.5.4)

When validation loss plateaus for 5 epochs, the learning rate is halved:

$$
\eta_{\text{new}} = \max(\eta_{\text{current}} \times 0.5, \; 10^{-6})
$$

---

## 14. Stratified Train/Val/Test Split (Section 3.5.2)

The dataset is split into training (70%), validation (15%), and test (15%) using
stratified sampling. For each class $k$, the proportion is preserved:

$$
\frac{n_k^{\text{split}}}{N^{\text{split}}} = \frac{n_k}{N}
$$

The two-step process:
1. Hold out 15% for testing (stratified by $\mathbf{y}$)
2. Split remaining 85% into 70% train / 15% val ($15/85 = 17.65\%$ of remaining)

---

## 15. Accuracy (Section 3.5.5)

$$
\text{Accuracy} = \frac{\text{TP} + \text{TN}}{N} = \frac{1}{N} \sum_{i=1}^{N}
\mathbb{1}(\hat{y}_i = y_i)
$$

---

## 16. Precision (Section 3.5.5)

**Per-class (macro):**

$$
\text{Precision}_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FP}_k}
$$

**Macro average:**

$$
\text{Precision}_{\text{macro}} = \frac{1}{K} \sum_{k=1}^{K} \text{Precision}_k
$$

**Weighted average:**

$$
\text{Precision}_{\text{weighted}} = \sum_{k=1}^{K} \frac{n_k}{N} \cdot \text{Precision}_k
$$

---

## 17. Recall (Section 3.5.5)

**Per-class:**

$$
\text{Recall}_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FN}_k}
$$

**Macro average:**

$$
\text{Recall}_{\text{macro}} = \frac{1}{K} \sum_{k=1}^{K} \text{Recall}_k
$$

---

## 18. F1-Score (Section 3.5.5)

**Per-class:**

$$
\text{F1}_k = \frac{2 \cdot \text{Precision}_k \cdot \text{Recall}_k}
{\text{Precision}_k + \text{Recall}_k}
$$

**Macro average (primary metric):**

$$
\text{F1}_{\text{macro}} = \frac{1}{K} \sum_{k=1}^{K} \text{F1}_k
$$

**Weighted average:**

$$
\text{F1}_{\text{weighted}} = \sum_{k=1}^{K} \frac{n_k}{N} \cdot \text{F1}_k
$$

---

## 19. ROC-AUC — One-vs-Rest (Section 3.5.5)

For each class $k$, a binary classifier is constructed (class $k$ vs all others).
The ROC curve plots TPR vs FPR at varying thresholds, and the AUC is computed:

$$
\text{AUC}_k = \int_0^1 \text{TPR}_k(t) \, d\text{FPR}_k(t)
$$

where:

$$
\text{TPR}_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FN}_k}, \quad
\text{FPR}_k = \frac{\text{FP}_k}{\text{FP}_k + \text{TN}_k}
$$

**Macro average:**

$$
\text{AUC}_{\text{macro}} = \frac{1}{K} \sum_{k=1}^{K} \text{AUC}_k
$$

Requires predicted probabilities $\hat{\mathbf{y}}$, not hard predictions.

---

## 20. Confusion Matrix (Section 3.5.5)

The $K \times K$ confusion matrix $\mathbf{C}$ has entries:

$$
C_{ij} = \sum_{l=1}^{N} \mathbb{1}(y_l = i \;\wedge\; \hat{y}_l = j)
$$

Row-normalised (showing per-class recall):

$$
\hat{C}_{ij} = \frac{C_{ij}}{\sum_{j'=1}^{K} C_{ij'}} = \frac{C_{ij}}{n_i}
$$

---

## 21. Permutation Feature Importance (Section 3.7)

For each feature $j$, repeated $R = 10$ times:

$$
\text{Importance}_j = \frac{1}{R} \sum_{r=1}^{R}
\left( \text{Accuracy}_{\text{base}} - \text{Accuracy}_{\text{permuted}}^{(j,r)} \right)
$$

where $\text{Accuracy}_{\text{permuted}}^{(j,r)}$ is the accuracy after randomly
shuffling feature $j$ in the test set.

---

## References

- Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436–444.
- Kingma, D. P. & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR 2015*.
- Ioffe, S. & Szegedy, C. (2015). Batch normalization. *ICML 2015*.
- Srivastava, N. et al. (2014). Dropout: A simple way to prevent neural networks from overfitting. *JMLR*, 15(1), 1929–1958.
- Tavallaee, M. et al. (2009). A detailed analysis of the KDD CUP 99 data set. *IEEE SISST*.
