# Data Preprocessing Steps

**Project:** Deep Learning IDS Using LSTM  
**Chapter Reference:** Chapter 3, Section 3.5.2 — Data Preprocessing Pipeline  
**Module:** `src/data/preprocessing.py`

---

## Overview

The preprocessing pipeline transforms raw, heterogeneous dataset files into clean,
encoded, scaled, and temporally structured input tensors ready for LSTM training.
Each step is applied consistently across all three datasets (NSL-KDD, CICIDS2017,
UNSW-NB15) with dataset-specific parameters where necessary.

The pipeline is implemented as a series of discrete, composable functions that
each accept a DataFrame and return a transformed DataFrame. This design supports
inspection at any intermediate stage via the interim CSV files saved to
`data/interim/`.

---

## Step 0 — Data Loading

**Function:** `src/data/loaders.load_dataset()`  
**Input:** Raw TXT/CSV files from `data/raw/`  
**Output:** Merged pandas DataFrame

### NSL-KDD Specific

The NSL-KDD files are headerless CSVs. Column names are assigned from the
`NSL_KDD_COLUMNS` constant (43 columns: 41 features + label + difficulty).

```python
columns = [
    'duration', 'protocol_type', 'service', 'flag',
    'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
    'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login',
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]
```

Training and test sets are concatenated into a single merged DataFrame with a
`_split` column tracking origin (`"train"` or `"test"`).

**Saved:** `data/interim/merged_dataset.csv`

---

## Step 1 — Drop Irrelevant Columns

**Function:** `drop_irrelevant_columns(df, dataset='nsl_kdd')`

| Column | Reason for dropping |
|--------|---------------------|
| `difficulty` | Meta-label recording classification difficulty — not a network feature |
| `_split` | Loader tracking column — no predictive value |

```python
df = df.drop(columns=['difficulty', '_split'], errors='ignore')
```

---

## Step 2 — Handle Infinite Values

**Function:** `handle_missing_and_infinite()`  
**Chapter 3:** *"Missing and infinite values — particularly prevalent in CICIDS2017
— were handled through mean imputation for continuous features."*

Infinite values (`numpy.inf`, `-numpy.inf`) arise in CICIDS2017 when flow duration
is zero (causing division by zero in flow rate features such as `Flow Bytes/s`
and `Flow Packets/s`). These are replaced with `NaN` before imputation.

```python
df.replace([np.inf, -np.inf], np.nan, inplace=True)
```

---

## Step 3 — Missing Value Imputation

**Strategy:** As specified in Chapter 3, Section 3.5.2:
- **Continuous (numeric) features:** Mean imputation
- **Categorical (object) features:** Mode imputation

```python
# Continuous
for col in numeric_cols:
    df[col].fillna(df[col].mean(), inplace=True)

# Categorical
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)
```

**Why mean/mode (not median/KNN)?** The Chapter 3 methodology specifies mean for
continuous features because NSL-KDD contains no missing values in the numeric
columns — these operations are primarily relevant for CICIDS2017 and UNSW-NB15
where missing data is more prevalent.

---

## Step 4 — Remove Duplicate Rows

**Function:** `remove_duplicates(df)`

```python
df = df.drop_duplicates().reset_index(drop=True)
```

Duplicate network connection records inflate training statistics and can bias
the model toward memorising repeated patterns rather than generalising.

---

## Step 5 — Label Mapping (NSL-KDD)

**Function:** `map_nsl_kdd_labels(df)`  
**Chapter 3:** *"Label Encoding. The target variable was encoded as integers:
0 for normal traffic and ascending integers for each attack category."*

### Raw Attack Type → 5-Class Category → Integer

| Integer | Category | Raw Attack Types |
|---------|----------|-----------------|
| 0 | Normal | normal |
| 1 | DoS | back, land, neptune, pod, smurf, teardrop, apache2, mailbomb, processtable, udpstorm |
| 2 | Probe | ipsweep, nmap, portsweep, satan, mscan, saint |
| 3 | R2L | ftp_write, guess_passwd, imap, multihop, phf, spy, warezclient, warezmaster, sendmail, named, snmpgetattack, snmpguess, xlock, xsnoop, httptunnel |
| 4 | U2R | buffer_overflow, loadmodule, perl, rootkit, ps, sqlattack, xterm, worm |

### Two-Step Mapping Process

```python
# Step 1: Raw string → category string
df['label_category'] = df['label'].str.lower().map(ATTACK_TO_CATEGORY)

# Step 2: Category string → integer
df['label'] = df['label_category'].map(CATEGORY_TO_INT)
```

Records with unrecognised attack types are dropped with a warning log.

---

## Step 6 — One-Hot Encoding

**Function:** `encode_categorical_features(df, dataset='nsl_kdd')`  
**Chapter 3:** *"Protocol type, service, and flag fields in NSL-KDD are categorical;
they were transformed into binary representations via one-hot encoding."*

The three categorical features in NSL-KDD are:

| Feature | Cardinality | Example values |
|---------|-------------|----------------|
| `protocol_type` | 3 | tcp, udp, icmp |
| `service` | ~70 | http, ftp, smtp, telnet, ... |
| `flag` | ~11 | SF, S0, REJ, RSTO, ... |

```python
df = pd.get_dummies(df, columns=['protocol_type', 'service', 'flag'],
                    drop_first=False, dtype=float)
```

`drop_first=False` is used intentionally — Chapter 3 specifies that ordinal
assumptions should not be imposed on nominal variables. Dropping the first
dummy column would implicitly create a reference category, which is inappropriate
for a non-ordinal feature.

**After encoding:** The feature count expands from 38 numeric + 3 categorical = 41
raw features to approximately 41 encoded features (exact count depends on
the number of unique values observed in the dataset split).

---

## Step 7 — Feature/Target Separation

**Function:** `separate_features_target(df, dataset='nsl_kdd')`

```python
y = df['label'].astype(int)           # Target vector: (N,) int64
X = df.drop(columns=['label'])        # Feature matrix: (N, n_features) float32
feature_names = X.columns.tolist()    # Ordered feature name list
```

The `feature_names` list is saved as `feature_names.pkl` and is critical for
inference — it ensures new data columns are reordered to match the exact column
order used during training.

---

## Step 8 — Min-Max Feature Scaling

**Function:** `fit_scaler(X_train)` then `apply_scaler(X_split, scaler)`  
**Chapter 3:** *"All continuous features were normalised using Min-Max scaling,
mapping values to the [0, 1] range."*

### Formula

```
X_scaled = (X - X_min) / (X_max - X_min)
```

### Data Leakage Prevention

The scaler is **fitted exclusively on the training split** and then applied
to the validation and test splits. This is the critical leakage-prevention
step described in Chapter 3:

> *"Scaling parameters — minimum and maximum values — were computed exclusively
> on the training set and applied to validation and test sets, preventing any
> form of data leakage."*

```python
scaler = MinMaxScaler(feature_range=(0.0, 1.0))
scaler.fit(X_train)           # Fit on training data only
X_train_scaled = scaler.transform(X_train)
X_val_scaled   = scaler.transform(X_val)    # Apply — no re-fitting
X_test_scaled  = scaler.transform(X_test)   # Apply — no re-fitting
```

**Why this matters:** If the scaler were fitted on all data (including validation
and test), the test set statistics would leak into the training process through
the scaling parameters — inflating reported performance.

---

## Step 9 — Sequence Construction (Sliding Window)

**Function:** `build_sequences(X, y, window_size=10, step_size=1, label_position='last')`  
**Module:** `src/data/sequence_builder.py`  
**Chapter 3:** *"A sliding window of width 10 was applied across the ordered
traffic records, constructing input sequences where each sequence contains
10 consecutive network connections."*

### Algorithm

```python
for start in range(0, n_samples - window_size + 1, step_size):
    end = start + window_size
    X_seq[seq_idx] = X[start:end]      # 10 consecutive records
    y_seq[seq_idx] = y[end - 1]        # Label of final record (label='last')
```

### Input/Output Shapes

| Array | Shape | Dtype |
|-------|-------|-------|
| Input X (flat) | (N, n_features) | float32 |
| Input y (flat) | (N,) | int64 |
| Output X_seq | (N−9, 10, n_features) | float32 |
| Output y_seq | (N−9,) | int64 |

For NSL-KDD with N=148,517 merged records:
- Output sequences: 148,508
- Memory: ~148,508 × 10 × 41 × 4 bytes ≈ 243 MB

### Window Size Justification

Window size of 10 was selected through hyperparameter search (Chapter 3,
Section 3.5.4): *"which showed diminishing returns for longer windows given
the feature characteristics of the datasets used."*

---

## Step 10 — Stratified Train/Val/Test Split

**Function:** `split_sequences(X_seq, y_seq, train=0.70, val=0.15, test=0.15)`  
**Module:** `src/data/split.py`

```python
# Step 1: Split off test set (15%)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_seq, y_seq, test_size=0.15, stratify=y_seq, random_state=42
)

# Step 2: Split trainval into train (70%) and val (15%)
val_relative = 0.15 / (0.70 + 0.15)   # = 0.1765
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=val_relative, stratify=y_trainval, random_state=42
)
```

**Stratification** preserves the original class proportions in each split,
ensuring the rare U2R and R2L classes appear in all three partitions.

---

## Saved Interim Files

Each intermediate stage saves a CSV to `data/interim/` for inspection:

| File | Stage | Description |
|------|-------|-------------|
| `merged_dataset.csv` | After loading | Raw merged train+test |
| `cleaned_dataset.csv` | After steps 1–4 | After dropping, imputing, de-duplication |
| `encoded_dataset.csv` | After steps 5–6 | After label mapping and OHE |
| `scaled_dataset.csv` | After step 8 | After MinMax scaling (with label column) |

---

## Preprocessing Artifacts (Saved to disk)

| Artifact | File | Used For |
|----------|------|----------|
| Fitted scaler | `scaler.pkl` | Applying same scale to new data at inference |
| Label encoder | `label_encoder.pkl` | Decoding integer predictions to class names |
| Feature names | `feature_names.pkl` | Column alignment at inference time |
| Metadata | `metadata.json` | Dataset statistics, n_classes, class names |
| Numpy arrays | `X_train.npy` … `y_test.npy` | Direct loading for training/evaluation |
