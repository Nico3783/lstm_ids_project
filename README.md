# Deep Learning Intrusion Detection System Using LSTM

**Final Year Project — Department of Cybersecurity**
**Federal University of Technology, Akure (FUTA), Nigeria**
**Project Implementation By: Kayode Nicholas Oluwatimileyin**

---

## Project Overview

This repository contains the complete implementation of a **Deep Learning-based Network Intrusion Detection System (IDS)** using **Long Short-Term Memory (LSTM)** neural networks. The system classifies network traffic as normal or malicious, detecting four categories of attacks:

| Class | Attack Category | Description |
|-------|----------------|-------------|
| 0 | Normal | Legitimate network traffic |
| 1 | DoS | Denial of Service attacks |
| 2 | Probe | Reconnaissance and scanning |
| 3 | R2L | Remote to Local attacks |
| 4 | U2R | User to Root privilege escalation |

The system addresses five key research gaps identified in the literature: temporal modelling depth, cross-dataset generalisation, class imbalance handling, overfitting management, and contextual relevance to the Nigerian cybersecurity environment.

---

## Features

- **Multi-dataset Support** — NSL-KDD (primary), CICIDS2017, UNSW-NB15
- **End-to-End Pipeline** — From raw data to evaluated model in one command
- **Advanced LSTM Architecture** — Stacked LSTM with dropout, L2 regularisation, and batch normalisation
- **Class Imbalance Handling** — Inverse-frequency class weights + optional SMOTE
- **Baseline Comparison** — Random Forest, SVM, Logistic Regression, and standard RNN
- **Hyperparameter Tuning** — Grid search over architecture and training parameters
- **Publication-Quality Figures** — All Chapter 4 visualisations auto-generated at 300 DPI
- **Fully Reproducible** — Seeded random states, saved preprocessing artifacts, config-driven
- **Academically Aligned** — Implementation matches methodology described in Chapters 1–3

---

## Project Structure

```
lstm_ids_project/README.md                    # This file
lstm_ids_project/requirements.txt             # All dependencies
lstm_ids_project/config.yaml                  # Central configuration (all hyperparameters)
lstm_ids_project/setup.sh                     # One-command environment setup
lstm_ids_project/run_pipeline.py              # Full end-to-end pipeline runner
lstm_ids_project/train.py                     # Standalone model training script
lstm_ids_project/evaluate.py                  # Standalone evaluation script
lstm_ids_project/predict.py                   # Prediction on new data
lstm_ids_project/compare_models.py            # Baseline vs LSTM comparison

lstm_ids_project/data/raw/nsl_kdd/             # NSL-KDD dataset files
lstm_ids_project/data/raw/cicids2017/          # CICIDS2017 dataset files
lstm_ids_project/data/raw/unsw_nb15/           # UNSW-NB15 dataset files
lstm_ids_project/data/interim/                 # Intermediate processing stages
lstm_ids_project/data/processed/               # Final numpy arrays ready for training
lstm_ids_project/data/sample/                  # Sample input/output for demo

lstm_ids_project/notebooks/                   # Jupyter notebooks for exploration

lstm_ids_project/src/config/                  # Configuration loader
lstm_ids_project/src/data/                    # Data pipeline modules
lstm_ids_project/src/models/                  # LSTM and baseline model definitions
lstm_ids_project/src/training/                # Training loop, callbacks, tuning
lstm_ids_project/src/evaluation/              # Metrics, reports, visualisations
lstm_ids_project/src/visualization/           # Plot generation
lstm_ids_project/src/deployment/              # Inference and prediction
lstm_ids_project/src/utils/                   # Logging, paths, serialization helpers

lstm_ids_project/models/checkpoints/          # Per-epoch checkpoints
lstm_ids_project/models/final/                # Final production model
lstm_ids_project/models/baselines/            # Saved baseline models

lstm_ids_project/reports/figures/             # PNG plots (300 DPI)
lstm_ids_project/reports/tables/              # CSV metric tables
lstm_ids_project/reports/metrics/             # JSON evaluation results
lstm_ids_project/reports/logs/                # Training and pipeline logs

lstm_ids_project/outputs/                     # Prediction exports
lstm_ids_project/docs/                        # Architecture diagrams, screenshots
lstm_ids_project/tests/                       # Unit tests
lstm_ids_project/scripts/                     # Shell utility scripts 
```

---

## Installation

### Prerequisites
- Python 3.9 (recommended) or Python 3.10+
- pip 23+
- git
- 8GB+ RAM recommended
- GPU optional (CUDA-compatible for faster training)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/nico3783/lstm_ids_project.git
cd lstm_ids_project
```

### Step 2 — Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Automated Setup (Alternative)

```bash
chmod +x setup.sh
./setup.sh
```

---

## Dataset Setup

### NSL-KDD (Primary Dataset)

1. Visit: https://www.unb.ca/cic/datasets/nsl.html
2. Download `NSL-KDD.zip`
3. Extract and place files:

```
data/raw/nsl_kdd/KDDTrain+.txt
data/raw/nsl_kdd/KDDTest+.txt
data/raw/nsl_kdd/KDDTrain+_20Percent.txt
data/raw/nsl_kdd/field_names.csv
```

Or use the automated downloader:

```bash
python -m src.data.download --dataset nsl_kdd
```

### CICIDS2017 (Optional)

1. Visit: https://www.unb.ca/cic/datasets/ids-2017.html
2. Download all CSV files
3. Place in `data/raw/cicids2017/`

### UNSW-NB15 (Optional)

1. Visit: https://research.unsw.edu.au/projects/unsw-nb15-dataset
2. Download training and testing CSV files
3. Place in `data/raw/unsw_nb15/`

---

## Usage

### Run the Full Pipeline (Recommended)

Executes all stages automatically: data loading → preprocessing → sequence building → baseline training → LSTM training → evaluation → report generation.

```bash
python run_pipeline.py
```

To run on a specific dataset:

```bash
python run_pipeline.py --dataset nsl_kdd
python run_pipeline.py --dataset cicids2017
python run_pipeline.py --dataset unsw_nb15
```

### Train the LSTM Model Only

```bash
python train.py --dataset nsl_kdd --config config.yaml
```

### Evaluate a Saved Model

```bash
python evaluate.py --model models/final/lstm_ids_model.keras --dataset nsl_kdd
```

### Compare All Models

```bash
python compare_models.py --dataset nsl_kdd
```

### Predict on New Data

```bash
python predict.py --input data/sample/sample_input.csv --output outputs/predictions/
```

### Hyperparameter Tuning

Enable in `config.yaml` (`hyperparameter_tuning.enabled: true`), then:

```bash
python train.py --tune --dataset nsl_kdd
```

---

## Results

After running the full pipeline, all results are saved to `reports/`:

| Output | Location |
|--------|----------|
| Training accuracy/loss curves | `reports/figures/training_accuracy_curve.png` |
| Confusion matrix | `reports/figures/confusion_matrix.png` |
| ROC curves | `reports/figures/roc_curve.png` |
| Model comparison chart | `reports/figures/model_comparison_chart.png` |
| Classification report | `reports/metrics/classification_report.txt` |
| Evaluation results (JSON) | `reports/metrics/evaluation_results.json` |
| Final metrics table | `reports/tables/final_metrics.csv` |
| Training history | `reports/logs/training_history.csv` |

### Expected Performance (NSL-KDD)

Based on the literature review and methodology described in Chapter 3:

| Metric | Expected Range |
|--------|---------------|
| Accuracy | ≥ 97% |
| Macro F1-Score | ≥ 0.93 |
| ROC-AUC | ≥ 0.98 |

---

## Reproducibility

All experiments use a global random seed (`seed: 42` in `config.yaml`). To reproduce results exactly:

1. Use the same dataset files (same version from official sources)
2. Use the pinned versions in `requirements.txt`
3. Ensure `config.yaml` is unmodified
4. Run `python run_pipeline.py`

Preprocessing artifacts (scaler, label encoder, feature names) are saved to `data/processed/` and `models/final/` so that the exact same transformations are applied to new data.

---

## Architecture

The LSTM model architecture (Chapter 3, Section 3.5.3):

```
Input: (batch, 10, n_features)
    │
    ▼
LSTM(128 units, return_sequences=True, tanh/sigmoid, dropout=0.2)
    │
    ▼
LSTM(64 units, return_sequences=False, tanh/sigmoid, dropout=0.2)
    │
    ▼
Dense(32 units, ReLU, L2 λ=0.001)
    │
    ▼
BatchNormalization
    │
    ▼
Dense(n_classes, Softmax)
    │
    ▼
Output: Class probabilities
```

**Optimizer:** Adam (lr=0.001) | **Loss:** Categorical Cross-Entropy | **Max Epochs:** 100 | **Batch:** 64

---

## Academic Relevance

This project addresses a practical cybersecurity problem within the Nigerian context. It:

- Benchmarks against three internationally recognised IDS datasets
- Uses stratified evaluation that prevents optimistic bias
- Applies class weighting to handle severe class imbalance (U2R, R2L minority classes)
- Compares LSTM against classical ML baselines (Random Forest, SVM, Logistic Regression)
- Generates all figures and tables required for a complete Chapter 4 presentation
- Is aligned with the methodology documented in Chapters 1–3 of the project report

---

## Citation

```
Kayode T.N (2026). Design and Implementation of a Deep Learning Intrusion 
Detection System Using Long Short-Term Memory. Final Year Project, Department of 
Cybersecurity, Federal University of Technology, Akure (FUTA), Nigeria.
```

---

## License

This project is developed for academic purposes at FUTA. All datasets are used under their respective academic research licences.